# Extreme-starved capacity floor retrieval (N=25, 30d)

Status: complete. 6 configs (patient_only x 3 data fractions, marginalized
x 3 data fractions), 5 draws each, val + test split.

Scripts: [`sweep_patient_only_extreme_starved_n25_30d.sh`](../../mimic_iv_sweep/scripts/sweep_patient_only_extreme_starved_n25_30d.sh),
[`sweep_marginalized_binary_learned_linear_extreme_starved_n25_30d.sh`](../../mimic_iv_sweep/scripts/sweep_marginalized_binary_learned_linear_extreme_starved_n25_30d.sh),
and `_train50pct`/`_train5pct` variants of both, plus matching
`eval_test_*` scripts (all in `mimic_iv_sweep/scripts/`).

## Task

Multi-task binary classification on MIMIC-IV: N=25 randomly-sampled
diagnostic codes, predict whether each occurs within a 30-day window after
a prediction anchor time. 5 independent random draws of the 25 task codes.
3 training epochs per run.

## Model config

**Patient encoder** (`TimeDeltaRoPEPatientEncoder`): token embedding
(`nn.Embedding(vocab_size, D)`) → time-delta rotary self-attention
(position = cumulative log time gap between events, not sequence index) →
transformer blocks (self-attention + GELU feed-forward) → final LayerNorm.

| Param | Value |
| --- | --- |
| `encoder.embedding_dim` (D) | 4 |
| `encoder.num_heads` | 1 |
| `encoder.num_layers` | 1 |
| `encoder.ff_dim` | 8 |
| `max_seq_len` | 8 |

**`patient_only`** (`fusion=passthrough`): masked-mean pool over the
sequence → `Linear(4, 25)`. No retrieval; predictions come only from the
patient's own event history.

**`marginalized`**: query projector → retriever → fusion → head, then
marginalized over retrieved documents.

- Query projector: `sequence_mean_1024`, `in_dim=4` -- mean-pools the
  patient encoder's own hidden state, then a trainable `Linear(4, 1024)`.
- Retriever: `HFDatasetRetriever`, FAISS search over 125k passages from
  `MedRAG/textbooks`, pre-embedded with a frozen `Qwen3-Embedding-0.6B`
  model. Returns the top `K=4` documents per patient.
- Retrieval encoder: `TokenFeatureRetrievalEncoder`, a learned
  `nn.Embedding(151936, 64)` applied to retrieved-document tokens.
- Fusion: `PerDocCrossAttentionFusion`, 2 cross-attention layers,
  `d_model=256`, 8 heads, `ff_dim=512`, one 256-dim output per retrieved
  document.
- Head: `Linear(256, 25)` applied per document.
- `marginalized_output_mode=binary`: each of the 25 tasks' sigmoid
  probability is marginalized over the K=4 documents independently,
  weighted by softmax over the K documents' retrieval scores.

## Data config

Training-set size is varied at 3 levels; validation (`tuning.parquet`) and
test (`held_out.parquet`) splits are the same full size at every level.
`train.parquet` is subsampled uniformly at random by row, with a fixed
seed per draw; `patient_only` and `marginalized` train on the identical
subsampled label file for a given (draw, fraction).

| Train fraction | Approx. train rows (of ~153,195 full) |
| --- | --- |
| 100% | 153,195 |
| 50% | 76,598 |
| 5% | 7,660 |

Retrieval corpus is not subsampled at any data level -- full 125k passages
throughout.

Tensorized cohort: `/groups/mm6677_gp/zzw2102/data/MIMIC_MEDS_restored/processed`.

## Results

Val: `val/auroc/mean`, computed once at the end of fit against the tuning
split, `retriever.k=4` for `marginalized` (as trained). Test:
`test/auroc/mean`, `eval_mode=test` scoring each checkpoint's
`checkpoints/last.ckpt` against the held-out split; for `marginalized`,
test uses `retriever.k=1, ablation_mode=none` (top-1 retrieved document
only, not the trained K=4 marginalization). `patient_only` has no
retrieval, so val and test use the same config throughout.

### Summary

| Train fraction | patient_only (val) | patient_only (test) | marginalized (val, k=4) | marginalized (test, top1) | Δ (test) | Draws won (test) |
| --- | --- | --- | --- | --- | --- | --- |
| 100% | 0.7911 ± 0.0314 | 0.7846 ± 0.0312 | 0.8286 ± 0.0300 | 0.8201 ± 0.0301 | +0.0355 | 5/5 |
| 50% | 0.7506 ± 0.0333 | 0.7486 ± 0.0391 | 0.8051 ± 0.0221 | 0.8051 ± 0.0259 | +0.0564 | 5/5 |
| 5% | 0.5255 ± 0.0164 | 0.5334 ± 0.0123 | 0.6829 ± 0.0472 | 0.6590 ± 0.0513 | +0.1257 | 5/5 |

### Per draw, 100% data

| Draw | patient_only (val) | patient_only (test) | marginalized (val, k=4) | marginalized (test, top1) | Δ (val) | Δ (test) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.8032 | 0.8138 | 0.8425 | 0.8519 | +0.0393 | +0.0381 |
| 2 | 0.8034 | 0.7716 | 0.8150 | 0.7897 | +0.0116 | +0.0181 |
| 3 | 0.7991 | 0.8080 | 0.8299 | 0.8361 | +0.0308 | +0.0281 |
| 4 | 0.8199 | 0.8002 | 0.8733 | 0.8445 | +0.0534 | +0.0443 |
| 5 | 0.7300 | 0.7294 | 0.7825 | 0.7784 | +0.0525 | +0.0490 |

### Per draw, 50% data

| Draw | patient_only (val) | patient_only (test) | marginalized (val, k=4) | marginalized (test, top1) | Δ (val) | Δ (test) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.7729 | 0.7851 | 0.8198 | 0.8286 | +0.0469 | +0.0435 |
| 2 | 0.7302 | 0.7136 | 0.7893 | 0.7860 | +0.0591 | +0.0724 |
| 3 | 0.7636 | 0.7818 | 0.8061 | 0.8211 | +0.0425 | +0.0393 |
| 4 | 0.7896 | 0.7726 | 0.8366 | 0.8261 | +0.0470 | +0.0535 |
| 5 | 0.6965 | 0.6901 | 0.7738 | 0.7635 | +0.0773 | +0.0734 |

### Per draw, 5% data

| Draw | patient_only (val) | patient_only (test) | marginalized (val, k=4) | marginalized (test, top1) | Δ (val) | Δ (test) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.5238 | 0.5422 | 0.7113 | 0.7080 | +0.1875 | +0.1658 |
| 2 | 0.5043 | 0.5287 | 0.6166 | 0.5832 | +0.1123 | +0.0545 |
| 3 | 0.5263 | 0.5421 | 0.6989 | 0.7069 | +0.1726 | +0.1648 |
| 4 | 0.5546 | 0.5428 | 0.7460 | 0.6839 | +0.1914 | +0.1411 |
| 5 | 0.5186 | 0.5111 | 0.6416 | 0.6132 | +0.1230 | +0.1021 |
