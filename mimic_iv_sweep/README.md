# mimic_iv_sweep

Systematic sweep over task scale, architecture variants, and hyperparameters
on MIMIC-IV, building on results from [`mimic_iv/`](../mimic_iv/) and
[`mimic_iv_smoke/`](../mimic_iv_smoke/).

All scripts use the post-refactor flat CLI (`medrap-train`, `medrap-preprocess`,
`medrap-prepare-retrieval-dataset`) and reuse the lab-shared tensorized cohort
at `/groups/mm6677_gp/data/MIMIC_MEDS/MEDS_cohort/processed`.

## Setup

```bash
cd mimic_iv_sweep
uv sync
```

This creates `.venv/` and installs `medrap` and all dependencies (including
`torch` and `nvidia-*-cu12` CUDA packages) from PyPI.

## Sweep design

### Phase 1 — Task-count sweep (`sweep_task_count.sh`)

Fixed architecture (RoPE encoder + cross-attention medium fusion, k=4, lr=1e-3,
3 epochs). Varies N ∈ {1, 2, 4, 8, 16, 32} to measure how jointly
training more prediction targets affects per-task AUROC.

N is kept small: task codes are sampled from a long-tailed clinical
vocabulary, and only a handful of codes (e.g. Blood Pressure, Weight, BMI)
have enough in-window positive volume for a stable per-task AUROC given the
current anchor-sampling scheme (see `scripts/check_task_balance.py`) --
requesting a large N mostly adds low-count, noise-dominated tasks rather
than signal.

### Phase 2 — Architecture ablation (`sweep_architecture.sh`)

Fixed N ∈ {1, 8, 32}. Compares three architectures:

| Variant | Description |
| --- | --- |
| `patient_only` | RoPE encoder → masked-mean pooling → head; no retrieval |
| `retrieval` | RoPE + cross-attention medium fusion, k=4 retrieved docs |
| `marginalized` | Same as `retrieval` but with `multitask_binary_bce_marginalized` loss |

### Phase 3 — Hyperparameter sweep (`sweep_hparams.sh`)

Fixed N=8, RoPE + cross-attention, 3 epochs. Full grid:
k ∈ {4, 8, 16, 32} × lr ∈ {1e-4, 1e-3, 3e-3} → 12 jobs.

## Running the sweep

### Step 1 — Build the retrieval index (once)

```bash
cd mimic_iv_sweep
sbatch scripts/prepare_retrieval.sh
```

### Step 2 — Generate task labels for each N (once per N)

```bash
sbatch scripts/generate_labels.sh          # all 6 N values in parallel
# or a subset, e.g. only N=8 and N=32:
sbatch --array=3,5 scripts/generate_labels.sh
```

Label N values and their array indices:

| Index | N |
| --- | --- |
| 0 | 1 |
| 1 | 2 |
| 2 | 4 |
| 3 | 8 |
| 4 | 16 |
| 5 | 32 |

### Step 3 — Run the sweeps

```bash
sbatch scripts/sweep_task_count.sh
sbatch scripts/sweep_architecture.sh
sbatch scripts/sweep_hparams.sh
```

Each script accepts extra arguments forwarded to `medrap-train` as Hydra
overrides:

```bash
sbatch scripts/sweep_task_count.sh training.trainer.max_epochs=5
```

## Output layout

```
mimic_iv_sweep/
├── data/
│   ├── retrieval_db/          # FAISS index + HF dataset (prepare_retrieval.sh)
│   └── tasks/
│       ├── n1/tasks/          # {train,tuning,held_out}.parquet + metadata
│       ├── n2/tasks/
│       └── ...
├── logs/                      # SLURM stdout/stderr per array job
└── outputs/
    ├── task_count/
    │   ├── n1/                # checkpoints + W&B run
    │   ├── n2/
    │   └── ...
    ├── architecture/
    │   ├── patient_only_n8/
    │   ├── retrieval_n8/
    │   ├── marginalized_n8/
    │   └── ...
    └── hparams/
        ├── k4_lr1e-4/
        ├── k8_lr1e-3/
        └── ...
```

All runs are logged to W&B under the `medrap` project with run names prefixed
`sweep-task-count-*`, `sweep-arch-*`, and `sweep-hparams-*`.
