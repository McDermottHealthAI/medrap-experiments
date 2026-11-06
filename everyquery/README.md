# EveryQuery with retrieval

Retrieval-augmented [EveryQuery](https://github.com/payalchandak/EveryQuery) on MIMIC-IV with the
medical-textbook corpus. Two configurations: **#1 patient only** (EveryQuery as is) and **#2
retrieval + patient**. Details in [`design.md`](design.md).

| File | What it does |
|---|---|
| [`EVERYQUERY.md`](EVERYQUERY.md) | how EveryQuery works: model, training, loss, evaluation (no retrieval) |
| [`design.md`](design.md) | the design, for review |
| [`run.sh`](run.sh) | one step: `preprocess`, `labels DRAW`, `eval_grid`, `encode_docs`, `train CONFIG SIZE LR DRAW` |
| [`submit.sh`](submit.sh) | submits `data`, `docs`, `pilot` or `stage0` to SLURM; skips finished and queued work; refuses to train without data |
| [`encode_docs.py`](encode_docs.py) | document key index (and contextual Qwen3 vectors per passage, kept for a richer encoding later) |
| [`pyproject.toml`](pyproject.toml) | environment: EveryQuery pinned to `13d7df8`, torch 2.10, FAISS, datasets, simple-scaling-laws |

## EveryQuery today (#1)

```mermaid
flowchart TB
    P["Patient events<br/>last ≤ 256 codes before the prediction time"]
    Q["Window tokens<br/>W0, C0 A0, W1, … , W4"]
    P --> S
    Q --> S
    S["One stream: [ patient events | W0 C0 A0 … W4 ]<br/>B × (S + 13) tokens"]
    S --> L["Causal Llama<br/>12 layers, hidden 384"]
    L --> H["Hidden state at each W token<br/>B × 5 × 384"]
    H --> R["× code-embedding table<br/>one logit per code"]
    R --> O["Logits B × 5 × V"]
    O --> Loss["BCE against the labels<br/>every code in every window"]
```

## With the retriever (#2)

```mermaid
flowchart TB
    P["Patient events<br/>last ≤ 256 codes"] --> LP["Causal Llama over the patient events"]
    LP --> HL["Hidden state at the last patient event<br/>B × 384"]
    HL --> QP["Query: linear 384 → 1024<br/>trained"]
    QP --> F["FAISS search over textbook keys<br/>top K, score s_k = query · key / √1024"]
    DB[("Textbook passages<br/>one 1024-d key each")] --> F
    F --> DT["One token per passage<br/>linear(key_k) + s_k · u<br/>B × K × 384"]
    DT --> S
    P --> S
    Q["Window tokens<br/>W0, C0 A0, … , W4"] --> S
    S["One stream: [ patient events | K passage tokens | W0 C0 A0 … W4 ]<br/>B × (S + K + 13) tokens"]
    S --> L["Causal Llama<br/>same backbone"]
    L --> H["Hidden state at each W token<br/>B × 5 × 384"]
    H --> R["× code-embedding table<br/>one logit per code"]
    R --> O["Logits B × 5 × V"]
    O --> Loss["BCE against the labels<br/>same loss as #1"]
    classDef added fill:#e7ecfd,stroke:#3b5bdb,color:#1d1d1f
    class LP,HL,QP,F,DB,DT added
```

Blue boxes are added by retrieval; the rest is EveryQuery as above. Each retrieved passage is one
token: its key embedding projected to 384, plus its search score times a learned vector `u`. The
score term is what trains the query (nothing flows back through the search). The loss and
evaluation are unchanged; the only setting is K. Retrieval runs on L40S GPUs (FAISS has no H100
kernels). Details in [`design.md`](design.md) §2.

## Status

| Piece | Status |
|---|---|
| Data: preprocessing, training labels (3 draws), evaluation grid (1,000 specs) | ready to run |
| Stage 0: patient-only configuration search (18 runs) | ready to run |
| Document key index | done |
| Retrieval + patient model (#2), one token per passage | next |
| Stage 1 K sweep, paired analysis | after #2 is implemented |

## Running on the cluster

```bash
cd everyquery
uv sync                      # .venv: EveryQuery + torch 2.10 (CUDA 12) + FAISS

# Data (CPU): preprocess, then 3 training-label draws and the evaluation grid after it
./submit.sh data

# Document key index (GPU, independent of the above); reads ../scaling_laws/data/retrieval_db
./submit.sh docs

# After the data jobs: one pilot run to time training, then Stage 0
./submit.sh pilot
./submit.sh stage0
```

Each training job trains, predicts on the evaluation grid and evaluates:

```
outputs/<CONFIG>_<SIZE>_lr<LR>/draw<DRAW>/
    checkpoints/                    best (by tuning loss) and last; kept
    loggers/lightning_logs/…/metrics.csv
    resolved_config.yaml
    predictions.parquet             one probability per grid row
    metrics*                        AUROC per query spec with bootstrap CIs
```

Paths can be overridden with `MEDS_DIR`, `RETRIEVAL_DB` and `N_TRAIN` (see [`run.sh`](run.sh)).
Logs go to CSV; W&B is off.
