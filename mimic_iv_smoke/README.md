# mimic_iv_smoke

Smoke test for the post-refactor MedRAP structure (subpackage reorg into
`model/`/`train/`/`prepare_retrieval/`/`preprocess/`, flat CLI entrypoints
`medrap-train`/`medrap-eval`/`medrap-prepare-retrieval-dataset`/`medrap-preprocess`).
Rebuilds the same MIMIC-IV pipeline as [`mimic_iv/`](../mimic_iv/) against the new CLI,
and additionally exercises the new `medrap-preprocess` rare-code/sparse-subject
filtering stage before tensorization, which the original `mimic_iv/` scripts never
needed (they read from a lab-shared, already-tensorized cohort directly).

Task labels are **not** regenerated here — `scripts/train.sh` reuses the existing
multi-task labels already produced by `mimic_iv/scripts/prepare_multi_task_labels_slurm.sh`,
since label prep is unrelated to the MedRAP refactor.

## Setup

```bash
pip install -r requirements.txt
```

## Scripts

Run in this order:

1. **`scripts/prepare_retrieval.sh`** — build the HF retrieval corpus and FAISS index
   (`medrap-prepare-retrieval-dataset`).
2. **`scripts/prepare_meds_data.sh`** — filter the shared raw MIMIC-IV MEDS cohort
   (`medrap-preprocess`), then tensorize the filtered output for PyTorch (`MTD_preprocess`).
3. **`scripts/train.sh`** — train the RoPE + cross-attention multitask model
   (`medrap-train`) on the tensorized output from step 2, using the existing
   `mimic_iv` task labels.

All three forward extra arguments as Hydra overrides, e.g.:

```bash
sbatch scripts/train.sh training.trainer.max_epochs=20
```
