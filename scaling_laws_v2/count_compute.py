"""Training compute and parameter counts per configuration (design.md §9).

    python scaling_laws_v2/count_compute.py OUT.csv RUN_DIR [RUN_DIR ...]

RUN_DIR is one draw of a run (outputs/<run>/draw1). For each, the model and datamodule are rebuilt
from train/resolved_config.yaml and one real training batch goes through forward and backward
under FlopCounterMode. Writes one row per run:

    flops_step   forward + backward FLOPs per patient, as torch counts them (attention, cross-
                 attention, the frozen Qwen forward pass for on-the-fly documents)
    flops_faiss  exact inner-product search, 2 · rows · dim per patient (FAISS runs outside torch);
                 0 for patient only
    C_train      (flops_step + flops_faiss) · D · epochs, the run's own training compute; a warm-
                 started run also used its patient-only checkpoint's (add that row's C_train)
    N            trained parameters except the code embedding table (the law's model size)
    N_trained    every trained parameter
    N_frozen     frozen parameters used per step (the Qwen3 token table or backbone); the
                 precomputed document vectors are data, not parameters

Encoding the corpus once (DOCS=pre) costs 2 · 0.6B · 125,848 · 256 ≈ 3.9 × 10¹⁶ FLOPs, shared by
every run; it is reported separately, not in C_train.
"""

import csv
import sys
from pathlib import Path

import polars as pl
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.flop_counter import FlopCounterMode

from medrap.train.factory import instantiate_training_module

EPOCHS = {3000: 7, 10000: 5, 30000: 4, 100000: 3, 153220: 3}  # run.sh
DOC_ROWS, DOC_DIM = 125_848, 1024


def count(run_dir: Path) -> dict:
    cfg = OmegaConf.load(run_dir / "train" / "resolved_config.yaml")
    if "path" in cfg.retrieval_encoder:  # precomputed table: map it instead of reading 66 GB
        cfg.retrieval_encoder.mmap = True
    cfg.training.module.init_encoder_from = None
    module = instantiate_training_module(cfg).cuda().train()
    module.log = module.log_dict = lambda *a, **k: None
    datamodule = instantiate(cfg.training.datamodule)
    datamodule.setup("fit")
    batch = module.transfer_batch_to_device(next(iter(datamodule.train_dataloader())), torch.device("cuda"), 0)
    with FlopCounterMode(display=False) as counter:
        module._run_supervised_step(batch, stage="train").backward()
    size = batch.code.shape[0]

    params = dict(module.named_parameters())
    trained = {n: p for n, p in params.items() if p.requires_grad}
    embedding = sum(
        m.weight.numel() for m in module.model.encoder.modules() if isinstance(m, torch.nn.Embedding)
    )
    retrieval = cfg.get("retriever") is not None and "patient_only" not in run_dir.parent.name
    d = len(pl.read_parquet(Path(cfg.training.datamodule.mt_labels_dir) / "train.parquet"))
    flops_step = counter.get_total_flops() / size
    flops_faiss = 2 * DOC_ROWS * DOC_DIM if retrieval else 0
    return {
        "run": run_dir.parent.name,
        "D": d,
        "flops_step": flops_step,
        "flops_faiss": flops_faiss,
        "C_train": (flops_step + flops_faiss) * d * EPOCHS.get(d, 3),
        "N": sum(p.numel() for p in trained.values()) - embedding,
        "N_trained": sum(p.numel() for p in trained.values()),
        "N_frozen": sum(p.numel() for n, p in params.items() if not p.requires_grad),
    }


if __name__ == "__main__":
    out, runs = Path(sys.argv[1]), [Path(r) for r in sys.argv[2:]]
    rows = []
    for r in runs:
        rows.append(count(r))
        print(rows[-1], flush=True)
        torch.cuda.empty_cache()
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
