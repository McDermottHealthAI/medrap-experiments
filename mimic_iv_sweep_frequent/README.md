# mimic_iv_sweep_frequent

Most-frequent-code variant of [`mimic_iv_sweep`](../mimic_iv_sweep/)'s Phase 4
(`sweep_marginalized_n1248.sh`). Everything is identical to that run
(architecture, hyperparameters, N ∈ {1, 2, 4, 8, 16, 32, 64, 128}, random prediction times)
**except how task codes are chosen**: instead of sampling uniformly at random
from the train-split vocabulary (which, per
[`mimic_iv_sweep`'s README](../mimic_iv_sweep/README.md#phase-1--task-count-sweep-sweep_task_countsh),
overwhelmingly lands on rare/degenerate codes), this variant picks the `N`
codes with the **highest distinct-subject count** in the train split (not raw
event-row count, which over-weights codes measured repeatedly on a small
subject subset).

Requires `medrap` built from
[`McDermottHealthAI/MedRAP@2c5fc5a`](https://github.com/McDermottHealthAI/MedRAP/commit/2c5fc5a668bf17394a58c49a4949ab6764ef0bc4)
(`experiment/marginalized-binary-plus-task-gen`, not yet merged to `main`),
pinned in `pyproject.toml`. That commit merges two not-yet-merged MedRAP
branches so both features are available together in one env:

- `feat/task-gen-most-frequent-codes` — adds `code_selection=random|most_frequent`
    to `medrap-preprocess`. `most_frequent` ranks codes by distinct-subject
    count in the train split, not raw event-row count -- a code measured
    repeatedly on a small subject subset (e.g. hourly ICU labs) can dominate
    row count while still being near-zero prevalence in the per-subject
    labels this pipeline produces.
- [`fix/marginalized-binary-output-mode`](https://github.com/McDermottHealthAI/MedRAP/pull/93)
    (MedRAP PR #93) — adds `marginalized_output_mode=categorical|binary` to
    `RetrievalAugmentedModel`. See `sweep_marginalized_binary_n1248.sh` below.

`experiment/marginalized-binary-plus-task-gen` is a merge-only integration
branch for this experiment's pin -- it isn't itself a PR under review; once
both `feat/task-gen-most-frequent-codes` and PR #93 land on `main`, drop
back to a main-tracking pin.

## Setup

```bash
cd mimic_iv_sweep_frequent
uv sync
```

## How to run

Reuses [`mimic_iv_sweep`](../mimic_iv_sweep/)'s retrieval index
(`../mimic_iv_sweep/data/retrieval_db`) rather than rebuilding it — the
retrieval corpus (`MedRAG/textbooks`) doesn't depend on task-code selection.
Run `mimic_iv_sweep`'s `prepare_retrieval.sh` first if that index doesn't
exist yet.

```bash
sbatch ../mimic_iv_sweep/scripts/prepare_retrieval.sh   # once, if data/retrieval_db doesn't exist yet
cd mimic_iv_sweep_frequent
sbatch scripts/generate_labels_n1248_frequent.sh        # N=1,2,4,8,16,32,64,128 task labels, most-frequent codes
sbatch scripts/sweep_patient_only_n1248.sh               # no retrieval, N=1,2,4,8,16,32,64,128
sbatch scripts/sweep_retrieval_n1248.sh                  # non-marginalized retrieval, N=1,2,4,8,16,32,64,128
sbatch scripts/sweep_marginalized_n1248.sh               # marginalized retrieval, categorical (buggy), N=1,2,4,8,16,32,64,128
sbatch scripts/sweep_marginalized_binary_n1248.sh        # marginalized retrieval, binary (MedRAP#93 fix), N=1,2,4,8,16,32,64,128
```

All four run the same task labels and hyperparameters (3 epochs, lr=1e-3,
batch size 32, `max_seq_len=256`) — only the architecture/output mode
differs, mirroring `../mimic_iv_sweep/scripts/sweep_architecture.sh`'s three
arms plus one MedRAP#93 variant:

- `sweep_patient_only_n1248.sh` — `patient_only`: RoPE encoder → masked-mean
    pooling → `LinearHead(B,N)` directly, no retriever/fusion at all
    (`fusion=passthrough`). Doesn't touch `data/retrieval_db` — no
    `prepare_retrieval.sh` prerequisite. Isolates how much the patient's own
    EHR sequence alone can do on these tasks.
- `sweep_retrieval_n1248.sh` — `retrieval`: single pooled cross-attention
    fusion over `k=4` retrieved docs, `training/loss=multitask_binary_bce`.
    Runs the same, non-marginalized architecture as the older
    `mt25-rope-cross-attn` runs, just on frequent-code labels instead of
    `mt25`'s pre-`MedRAP#92` positive-rate/count-filtered ones.
- `sweep_marginalized_n1248.sh` — `marginalized`, `marginalized_output_mode=categorical`
    (the current MedRAP default): per-document fusion +
    `marginalized_retrieval=true` + `multitask_binary_bce_marginalized` loss.
    Routes `logits` through `RetrievalAugmentedModel`'s per-document
    softmax-over-tasks path (`_marginal_class_probabilities`) -- correct for
    a single mutually-exclusive `C`-way task, but wrong for `N` independent
    binary tasks, which is suspected to produce the invalid AUROC seen in
    this PR's `marginalized-frequent-n{N}` results (bimodal near-0/near-1
    per-task AUROC).
- `sweep_marginalized_binary_n1248.sh` — identical to
    `sweep_marginalized_n1248.sh` plus one override,
    `marginalized_output_mode=binary`
    ([MedRAP PR #93](https://github.com/McDermottHealthAI/MedRAP/pull/93)):
    marginalizes each task's sigmoid probability independently over
    documents instead of forcing all tasks to compete via softmax. This is
    the same fix a colleague's unmerged branch used to get a genuinely good,
    uniform marginalized AUROC (0.916 mean, no bimodal collapse) on the
    older `mt25-rope-cross-attn-marginalized` W&B run -- this script tests
    it on frequent-code labels before PR #93 merges to `main`.

`sweep_patient_only_n1248.sh` and `sweep_retrieval_n1248.sh` never touch the
marginalized softmax-over-tasks path at all, so together with
`sweep_marginalized_binary_n1248.sh` they give three same-labels,
same-retrieval-index reference points unconfounded by that question — worth
running before drawing conclusions from `sweep_marginalized_n1248.sh`'s
(categorical) AUROC.

Each script accepts extra `medrap-train`/`medrap-preprocess` Hydra
overrides, same as `mimic_iv_sweep`'s scripts.

### Checking task balance

```bash
python scripts/check_task_balance.py data/tasks/n8/tasks
```

Prints per-task, per-split positive rates and flags degenerate (single-class)
or out-of-band tasks. Worth running before `sweep_marginalized_n1248.sh` to
confirm frequent-code selection actually produced usable positive rates,
unlike the random-task variant's N=1/2/4/8 labels.

## Output layout

```
mimic_iv_sweep_frequent/
├── data/
│   └── tasks/
│       ├── n1/tasks/          # {train,tuning,held_out}.parquet + metadata
│       ├── n2/tasks/
│       └── ...
├── logs/                      # SLURM stdout/stderr per array job
└── outputs/
    ├── patient_only_n1248/
    │   ├── n1/                # checkpoints + W&B run (no retrieval)
    │   ├── n2/
    │   └── ...
    ├── retrieval_n1248/
    │   ├── n1/                # checkpoints + W&B run (non-marginalized)
    │   ├── n2/
    │   └── ...
    ├── marginalized_n1248/
    │   ├── n1/                # checkpoints + W&B run (marginalized, categorical)
    │   ├── n2/
    │   └── ...
    └── marginalized_binary_n1248/
        ├── n1/                # checkpoints + W&B run (marginalized, binary -- MedRAP#93)
        ├── n2/
        └── ...
```

(No `data/retrieval_db/` here — shared from `../mimic_iv_sweep/data/retrieval_db`.)

All runs are logged to W&B under the `medrap` project, run names prefixed
`patient-only-frequent-n{N}-*`, `retrieval-frequent-n{N}-*`,
`marginalized-frequent-n{N}-*`, and `marginalized-binary-frequent-n{N}-*`
respectively, alongside the random-task variant's `marginalized-n{N}-*` for
direct comparison of `val/auroc/mean`, `val/auroc/n_valid_tasks` vs
`n_tasks`, and `train/loss`.
