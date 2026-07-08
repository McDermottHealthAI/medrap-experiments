# mimic_iv_smoke

Smoke test for the post-refactor MedRAP structure (subpackage reorg into
`model/`/`train/`/`prepare_retrieval/`/`preprocess/`, flat CLI entrypoints
`medrap-train`/`medrap-eval`/`medrap-prepare-retrieval-dataset`/`medrap-preprocess`).
Runs the same MIMIC-IV pipeline as [`mimic_iv/`](../mimic_iv/) against the new CLI,
reusing the existing lab-shared tensorized cohort directly but regenerating
multi-task labels via the new `medrap-preprocess` / task-generation stage.

## Setup

```bash
uv sync
```

This creates `.venv/` and installs `medrap` along with all its dependencies,
including `torch` and the `nvidia-*-cu12` CUDA packages from PyPI.

## Scripts

Run in this order:

1. **`scripts/prepare_retrieval.sh`** — build the HF retrieval corpus and FAISS index
   (`medrap-prepare-retrieval-dataset`).
2. **`scripts/generate_tasks.sh`** — generate multi-task binary code-occurrence
   labels (`medrap-preprocess`), pointed at the raw MEDS cohort with
   `tensorized_dir` set to the existing tensorized cohort so only the
   task-generation stage runs. Writes to `data/tasks_gen/tasks/`.
3. **`scripts/train.sh`** — train the RoPE + cross-attention multitask model
   (`medrap-train`), reading the existing tensorized cohort from
   `/groups/mm6677_gp/data/MIMIC_MEDS/MEDS_cohort/processed` and the task
   labels generated in step 2.

All three scripts forward extra arguments as Hydra overrides, e.g.:

```bash
sbatch scripts/train.sh training.trainer.max_epochs=20
```
