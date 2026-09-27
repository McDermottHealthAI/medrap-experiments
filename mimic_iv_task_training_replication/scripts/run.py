r"""Run one fixed five-epoch PR113 fit.

Setup in a Python 3.12 environment (from this experiment directory):
    python -m pip install -r requirements.txt
    python -m pip uninstall -y faiss-cpu
    python -m pip install faiss-gpu-cu12==1.13.2

The GPU wheel above supports Linux x86_64, CUDA 12 and Volta through Ada GPUs.
For unsupported GPUs such as Blackwell, replace the last command with installation
of a FAISS 1.13.2 wheel built for your GPU and CUDA version:
https://github.com/facebookresearch/faiss/blob/v1.13.2/INSTALL.md
The pinned MedRAP dependency installs faiss-cpu; remove it before installing the GPU
build. Reinstalling requirements restores that CPU dependency, so repeat the replacement.

The original run used FAISS 1.13.2 (CUDA 12.8, Blackwell), Torch 2.10.0+cu128,
and Lightning 2.6.1. Prepared inputs must match the recorded cohort and document bank.

Example (from this experiment directory):
    CUDA_VISIBLE_DEVICES=0 python scripts/run.py --arm real --seed 42 --task-seed 101 \
        --data-root /authorized/default_preprocess --retrieval-db /authorized/textbooks \
        --output /authorized/runs/task101_seed42_real

Repeat for task seeds 101/202/303, training seeds 42/43/44, and patient_only/real/random.
Use one process per available GPU. Keep patient outputs on authorized storage; logging is offline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

# No external experiment logging or model downloads are needed for prepared inputs.
os.environ["WANDB_MODE"] = "offline"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
for key in [
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "POLARS_MAX_THREADS",
]:
    os.environ[key] = "2"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import lightning as L
import medrap
import numpy as np
import polars as pl
import torch
from hydra import compose, initialize_config_dir
from hydra_zen import instantiate
from medrap.train.factory import instantiate_training_module
from omegaconf import OmegaConf
from sklearn.metrics import roc_auc_score


def save_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def digest_state(state):
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def metrics(y, logits):
    """Arrays have shape (patients, tasks); macro AUC uses all supported tasks."""
    p = torch.sigmoid(torch.from_numpy(logits)).numpy()
    bce = np.logaddexp(0, logits.astype("float64")) - y * logits
    auc = [
        float(roc_auc_score(y[:, t], logits[:, t]))
        if np.unique(y[:, t]).size == 2
        else None
        for t in range(y.shape[1])
    ]
    return {
        "rows": len(y),
        "auroc": auc,
        "macro_auroc": float(np.mean([x for x in auc if x is not None])),
        "bce": float(bce.mean()),
        "per_task_bce": bce.mean(0).tolist(),
        "mean_probability": p.mean(0).tolist(),
        "prevalence": y.mean(0).tolist(),
    }


def build_config(arm, seed, output, data_root, retrieval_db, task_seed):
    tasks = data_root / f"task_generation_30d_seed{task_seed}/tasks"
    vocab_indices = pl.read_parquet(
        data_root / "tensorized/metadata/codes.parquet", columns=["code/vocab_index"]
    )["code/vocab_index"]
    vocab = int(vocab_indices.max()) + 1
    overrides = [
        "encoder=rope",
        f"encoder.vocab_size={vocab}",
        "encoder.embedding_dim=128",
        "query_projector=sequence_mean_1024",
        "query_projector.in_dim=128",
        "retriever=hf_dataset",
        f"retriever.dataset_path={retrieval_db}",
        "retriever.doc_ids_column=null",
        "retriever.k=4",
        "retriever.cache_payloads=true",
        "retriever.payload_cache_device=cuda",
        "retriever.device=0",
        "retrieval_encoder=token_feature",
        "retrieval_encoder.vocab_size=151936",
        "retrieval_encoder.embedding_dim=64",
        "pooling=masked_mean",
        "head=linear",
        "head.out_dim=25",
        "training/task=multitask_binary",
        "training.task.num_tasks=25",
        "training/datamodule=meds_multitask",
        f"training.datamodule.config.tensorized_cohort_dir={data_root / 'tensorized'}",
        f"training.datamodule.config.task_labels_dir={tasks}",
        f"training.datamodule.mt_labels_dir={tasks}",
        "training.datamodule.num_tasks=25",
        "training.datamodule.config.max_seq_len=256",
        "training.datamodule.batch_size=32",
        "training.datamodule.num_workers=8",
        "training.datamodule.pin_memory=true",
        "training/trainer=lightning_default",
        "training.trainer.max_epochs=5",
        "training.trainer.accelerator=gpu",
        "training.trainer.devices=1",
        "+training.trainer.num_sanity_val_steps=0",
        "+training.trainer.limit_val_batches=1.0",
        "training.module.lr=0.001",
        "training.module.warmup_steps=200",
        "training.module.validation_auroc_log_per_task=true",
        f"seed={seed}",
        f"output_dir={output}",
    ]
    if arm == "patient_only":
        overrides += [
            "fusion=passthrough",
            "head.in_dim=128",
            "training/loss=multitask_binary_bce",
            "marginalized_retrieval=false",
        ]
    else:
        overrides += [
            "fusion=cross_attention_perdoc_medium",
            "head.in_dim=256",
            "training/loss=multitask_binary_bce_marginalized",
            "training.loss.num_tasks=25",
            "marginalized_retrieval=true",
            "marginalized_output_mode=binary",
            f"retriever.ablation_mode={'random_docs' if arm == 'random' else 'none'}",
        ]
    with initialize_config_dir(
        config_dir=str(Path(medrap.__file__).parent / "conf"), version_base=None
    ):
        cfg = compose(config_name="_train", overrides=overrides)
    # Same output-path binding as medrap.cli._bind_trainer_paths.
    cfg.training.trainer.logger.save_dir = str(output / "loggers")
    for callback in cfg.training.trainer.callbacks:
        if "dirpath" in callback:
            callback.dirpath = str(output / "checkpoints")
    return cfg


class Exposure(L.Callback):
    """Count processed examples and preserve RNG state in package checkpoints."""

    def __init__(self):
        self.examples = 0

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs
        if not torch.isfinite(loss).all():
            raise RuntimeError("Nonfinite training loss")
        self.examples += len(batch.code)

    def on_save_checkpoint(self, trainer, module, checkpoint):
        checkpoint["replication_rng"] = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all(),
            "sampler": trainer.train_dataloader.sampler.generator.get_state(),
        }


def check_batch(module, batch, arm):
    """Check seeded replay and patient-only independence from document inputs."""
    module.eval()
    with torch.inference_mode():
        torch.manual_seed(123456)
        first = module(batch).logits
        torch.manual_seed(123456)
        repeat = module(batch).logits
        assert torch.equal(first, repeat), "Seeded forward replay failed"
        assert torch.isfinite(first).all()
        if arm == "patient_only":
            old = module.model.retriever.ablation_mode
            module.model.retriever.ablation_mode = "random_docs"
            swapped = module(batch).logits
            module.model.retriever.ablation_mode = old
            assert torch.equal(first, swapped), "Patient-only depends on documents"
    module.train()


def evaluate(module, loader, output, split, tasks):
    from lightning.fabric.utilities.apply_func import move_data_to_device
    from torch.utils.data import SequentialSampler

    assert isinstance(loader.sampler, SequentialSampler)
    L.seed_everything(900001 if split == "tuning" else 900002, workers=True)
    module.eval()
    ys = []
    scores = []
    with torch.inference_mode():
        for batch in loader:
            batch = move_data_to_device(batch, module.device)
            logits = module(batch).logits
            y = module.task.extract_targets(batch)
            assert torch.isfinite(y).all() and torch.isfinite(logits).all()
            ys.append(y.cpu().numpy())
            scores.append(logits.float().cpu().numpy())
    y = np.concatenate(ys)
    logits = np.concatenate(scores)
    assert len(y) == len(loader.dataset)
    ids = np.array([int(item[0]) for item in loader.dataset.index], dtype=np.int64)
    assert len(np.unique(ids)) == len(ids)
    # Verify that loader targets match source labels in prediction order.
    labels = pl.read_parquet(tasks / f"{split}.parquet")
    aligned = pl.DataFrame({"subject_id": ids}).join(
        labels, on="subject_id", how="left", maintain_order="left"
    )
    assert np.array_equal(
        y, aligned.select([f"task_{t}" for t in range(25)]).to_numpy()
    )
    np.savez_compressed(
        output / f"{split}_predictions.npz", subject_id=ids, y=y, logits=logits
    )
    result = metrics(y, logits)
    save_json(output / f"{split}_metrics.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--arm", choices=["patient_only", "real", "random"], required=True
    )
    parser.add_argument("--seed", type=int, choices=[42, 43, 44], required=True)
    parser.add_argument("--task-seed", type=int, choices=[101, 202, 303], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Default-preprocessed cohort with tensorized/ and task_generation_30d_seed*/tasks/.",
    )
    parser.add_argument(
        "--retrieval-db",
        type=Path,
        required=True,
        help="Unfiltered prepared textbook HF dataset with retrieval.faiss.",
    )
    args = parser.parse_args()
    os.umask(0o077)
    args.output = args.output.resolve()
    args.data_root = args.data_root.resolve()
    args.retrieval_db = args.retrieval_db.resolve()
    tasks = args.data_root / f"task_generation_30d_seed{args.task_seed}/tasks"
    metadata = json.loads((tasks / "metadata.json").read_text())
    expected = {
        "num_tasks": 25,
        "horizon_days": 30.0,
        "min_history_days": 1.0,
        "seed": args.task_seed,
        "code_selection": "random",
        "anchor_strategy": "uniform_event",
        "duration_distribution": "fixed",
    }
    assert all(metadata[key] == value for key, value in expected.items()), (
        "Task panel settings differ"
    )
    assert (args.retrieval_db / "retrieval.faiss").is_file()
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = build_config(
        args.arm,
        args.seed,
        args.output,
        args.data_root,
        args.retrieval_db,
        args.task_seed,
    )
    OmegaConf.save_json(cfg, args.output / "config.yaml", resolve=True)
    L.seed_everything(args.seed, workers=True)
    module = instantiate_training_module(cfg)
    init = {
        name: digest_state(getattr(module.model, name).state_dict())
        for name in [
            "encoder",
            "query_projector",
            "retrieval_encoder",
            "fusion",
            "head",
        ]
    }
    save_json(
        args.output / "initialization.json",
        {
            "hashes": init,
            "parameters": sum(p.numel() for p in module.parameters()),
            "source": str(medrap.__file__),
        },
    )
    dm = instantiate(cfg.training.datamodule)
    dm.setup("fit")
    train = dm.train_dataloader()
    val = dm.val_dataloader()
    generator = torch.Generator().manual_seed(args.seed)
    train.generator = generator
    train.sampler.generator = generator
    # Reset RNG streams after the forward checks.
    batch = next(iter(train))
    from lightning.fabric.utilities.apply_func import move_data_to_device

    module.cuda()
    batch = move_data_to_device(batch, module.device)
    check_batch(module, batch, args.arm)
    L.seed_everything(args.seed, workers=True)
    generator.manual_seed(args.seed)
    trainer = instantiate(cfg.training.trainer)
    Path(trainer.logger.log_dir).mkdir(parents=True, exist_ok=True)
    exposure = Exposure()
    trainer.callbacks.append(exposure)
    start = time.time()
    trainer.fit(module, train_dataloaders=train, val_dataloaders=val)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    save_json(
        args.output / "training.json",
        {
            "seconds": elapsed,
            "steps": trainer.global_step,
            "examples": exposure.examples,
            "epochs": trainer.current_epoch,
            "gpu": str(torch.cuda.get_device_properties(0).uuid),
        },
    )
    # Save the fixed-budget endpoint: Lightning last.ckpt may remain at an earlier epoch.
    last = args.output / "checkpoints/final.ckpt"
    assert not last.exists()
    trainer.save_checkpoint(last)
    ckpt = torch.load(last, map_location="cpu", weights_only=False)
    assert ckpt["global_step"] == trainer.global_step
    final_state_hash = digest_state(module.state_dict())
    assert digest_state(ckpt["state_dict"]) == final_state_hash
    module.load_state_dict(ckpt["state_dict"])
    module.cuda()
    assert exposure.examples == 5 * len(train.dataset), "Incomplete training exposure"
    results = {"tuning": evaluate(module, val, args.output, "tuning", tasks)}
    dm.setup("test")
    results["held_out"] = evaluate(
        module, dm.test_dataloader(), args.output, "held_out", tasks
    )
    save_json(
        args.output / "complete.json",
        {
            "results": results,
            "seconds": time.time() - start,
            "final_state_sha256": final_state_hash,
        },
    )


if __name__ == "__main__":
    main()
