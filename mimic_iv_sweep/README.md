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

### Phase 4 — Marginalized retrieval, random tasks (`sweep_marginalized_n1248.sh`)

`marginalized` architecture only (RoPE + per-document cross-attention +
`marginalized_retrieval=true` + `multitask_binary_bce_marginalized` loss),
N ∈ {1, 2, 4, 8}, on task labels generated with the medrap task sampler after
[MedRAP#92](https://github.com/McDermottHealthAI/MedRAP/pull/92) — task codes
sampled uniformly at random from the train split, no positive-rate/count
filtering. Logs per-task AUROC every validation pass
(`training.module.validation_auroc_log_per_task=true`), not just the mean.

This is a first pass to see whether marginalized retrieval trains cleanly on
unfiltered random tasks before deciding how to extend the comparison (e.g.
adding a matched `patient_only` arm). `sweep_architecture.sh`'s `marginalized`
runs at N=25/100 (see W&B `sweep-arch-marginalized-*`) currently show AUROC in
the 0.45–0.64 range — worse than or barely above the `patient_only`/`retrieval`
arms at the same N — so treat this phase as investigative, not a settled
result.

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
sbatch scripts/sweep_marginalized_n1248.sh
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
