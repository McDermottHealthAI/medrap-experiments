# mimic_iv_sweep_frequent

Most-frequent-code variant of [`mimic_iv_sweep`](../mimic_iv_sweep/)'s Phase 4
(`sweep_marginalized_n1248.sh`). Everything is identical to that run
(architecture, hyperparameters, N ∈ {1, 2, 4, 8}, random prediction times)
**except how task codes are chosen**: instead of sampling uniformly at random
from the train-split vocabulary (which, per
[`mimic_iv_sweep`'s README](../mimic_iv_sweep/README.md#phase-1--task-count-sweep-sweep_task_countsh),
overwhelmingly lands on rare/degenerate codes), this variant picks the `N`
codes with the **highest distinct-subject count** in the train split (not raw
event-row count, which over-weights codes measured repeatedly on a small
subject subset).

Requires `medrap` built from
[`McDermottHealthAI/MedRAP@425a321`](https://github.com/McDermottHealthAI/MedRAP/commit/425a321268331a31cbdfa50fc60b3b0555e97284)
(`feat/task-gen-most-frequent-codes`, not yet merged to `main`) — adds
`code_selection=random|most_frequent` to `medrap-preprocess`; pinned in
`pyproject.toml`. `most_frequent` ranks codes by distinct-subject count in
the train split, not raw event-row count -- a code measured repeatedly on a
small subject subset (e.g. hourly ICU labs) can dominate row count while
still being near-zero prevalence in the per-subject labels this pipeline
produces.

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
sbatch scripts/generate_labels_n1248_frequent.sh        # N=1,2,4,8 task labels, most-frequent codes
sbatch scripts/sweep_patient_only_n1248.sh               # no retrieval, N=1,2,4,8
sbatch scripts/sweep_retrieval_n1248.sh                  # non-marginalized retrieval, N=1,2,4,8
sbatch scripts/sweep_marginalized_n1248.sh               # marginalized retrieval, N=1,2,4,8
```

All three run the same task labels, hyperparameters (3 epochs, lr=1e-3,
batch size 32, `max_seq_len=256`), and `training/loss=multitask_binary_bce`
(except `sweep_marginalized_n1248.sh`, which uses the marginalized loss) —
only the architecture differs, mirroring
`../mimic_iv_sweep/scripts/sweep_architecture.sh`'s three arms:

- `sweep_patient_only_n1248.sh` — `patient_only`: RoPE encoder → masked-mean
    pooling → `LinearHead(B,N)` directly, no retriever/fusion at all
    (`fusion=passthrough`). Doesn't touch `data/retrieval_db` — no
    `prepare_retrieval.sh` prerequisite. Isolates how much the patient's own
    EHR sequence alone can do on these tasks.
- `sweep_retrieval_n1248.sh` — `retrieval`: single pooled cross-attention
    fusion over `k=4` retrieved docs. Runs the same, non-marginalized
    architecture as the older `mt25-rope-cross-attn` runs, just on
    frequent-code labels instead of `mt25`'s pre-`MedRAP#92`
    positive-rate/count-filtered ones.
- `sweep_marginalized_n1248.sh` — `marginalized`: per-document fusion +
    `marginalized_retrieval=true` + `multitask_binary_bce_marginalized` loss.
    `marginalized_retrieval=true` routes `logits` through
    `RetrievalAugmentedModel`'s per-document softmax-over-tasks path
    (`_marginal_class_probabilities`), which is suspected of producing
    invalid AUROC for independent multi-task targets (see this PR's
    `marginalized-frequent-n{N}` results: bimodal near-0/near-1 per-task
    AUROC). The other two scripts never touch that path, so together they
    give a same-labels, same-retrieval-index reference point unconfounded by
    that open question — run and check them before drawing conclusions from
    `sweep_marginalized_n1248.sh`'s AUROC.

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
    └── marginalized_n1248/
        ├── n1/                # checkpoints + W&B run (marginalized)
        ├── n2/
        └── ...
```

(No `data/retrieval_db/` here — shared from `../mimic_iv_sweep/data/retrieval_db`.)

All runs are logged to W&B under the `medrap` project, run names prefixed
`patient-only-frequent-n{N}-*`, `retrieval-frequent-n{N}-*`, and
`marginalized-frequent-n{N}-*` respectively, alongside the random-task
variant's `marginalized-n{N}-*` for direct comparison of `val/auroc/mean`,
`val/auroc/n_valid_tasks` vs `n_tasks`, and `train/loss`.
