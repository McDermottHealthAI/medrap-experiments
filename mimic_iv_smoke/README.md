# mimic_iv_smoke

Smoke test for the post-refactor MedRAP structure (subpackage reorg into
`model/`/`train/`/`prepare_retrieval/`/`preprocess/`, flat CLI entrypoints
`medrap-train`/`medrap-eval`/`medrap-prepare-retrieval-dataset`/`medrap-preprocess`).
Runs the same MIMIC-IV pipeline as [`mimic_iv/`](../mimic_iv/) against the new CLI,
reusing the existing lab-shared tensorized cohort and multi-task labels directly.

## Setup

```bash
pip install -r requirements.txt
```

## Scripts

Run in this order:

1. **`scripts/prepare_retrieval.sh`** — build the HF retrieval corpus and FAISS index
   (`medrap-prepare-retrieval-dataset`).
2. **`scripts/train.sh`** — train the RoPE + cross-attention multitask model
   (`medrap-train`), reading the existing tensorized cohort from
   `/groups/mm6677_gp/data/MIMIC_MEDS/MEDS_cohort/processed` and the existing
   `mimic_iv` task labels from `../mimic_iv/data/mt_labels/top25_7d`.

Both scripts forward extra arguments as Hydra overrides, e.g.:

```bash
sbatch scripts/train.sh training.trainer.max_epochs=20
```
