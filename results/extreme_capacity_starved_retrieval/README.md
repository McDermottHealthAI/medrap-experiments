# Capacity sweep retrieval (N=25, 30d)

Status: complete. Two experiments: (1) a clean 6-point capacity-only
sweep with context length held fixed, (2) an extreme capacity+context cut
crossed with 3 training-data levels.

Scripts: `mimic_iv_sweep/scripts/sweep_{patient_only,marginalized_binary_learned_linear}_capacity_sweep_n25_30d_dim{4,8,16,32,64}.sh`
and matching `eval_test_*` scripts (experiment 1); `sweep_{patient_only,marginalized_binary_learned_linear}_extreme_starved_n25_30d{,_train50pct,_train5pct}.sh`
and matching `eval_test_*` scripts (experiment 2). All in
`mimic_iv_sweep/scripts/`.

## Task

Multi-task binary classification on MIMIC-IV: N=25 randomly-sampled
diagnostic codes, predict whether each occurs within a 30-day window after
a prediction anchor time. 5 independent random draws of the 25 task codes.
3 training epochs per run, 100% training data unless noted.

## Shared model architecture

**Patient encoder** (`TimeDeltaRoPEPatientEncoder`): token embedding
(`nn.Embedding(vocab_size, D)`) → time-delta rotary self-attention
(position = cumulative log time gap between events, not sequence index) →
transformer blocks (self-attention + GELU feed-forward) → final LayerNorm.

**`patient_only`** (`fusion=passthrough`): masked-mean pool over the
sequence → `Linear(D, 25)`. No retrieval; predictions come only from the
patient's own event history.

**`marginalized`**: query projector → retriever → fusion → head, then
marginalized over retrieved documents.

- Query projector: `sequence_mean_1024`, `in_dim=D` -- mean-pools the
  patient encoder's own hidden state, then a trainable `Linear(D, 1024)`.
- Retriever: `HFDatasetRetriever`, FAISS search over 125k passages from
  `MedRAG/textbooks`, pre-embedded with a frozen `Qwen3-Embedding-0.6B`
  model. Returns the top `K=4` documents per patient (not subsampled at
  any point in either experiment).
- Retrieval encoder: `TokenFeatureRetrievalEncoder`, a learned
  `nn.Embedding(151936, 64)` applied to retrieved-document tokens.
- Fusion: `PerDocCrossAttentionFusion`, 2 cross-attention layers,
  `d_model=256`, 8 heads, `ff_dim=512`, one 256-dim output per retrieved
  document.
- Head: `Linear(256, 25)` applied per document.
- `marginalized_output_mode=binary`: each of the 25 tasks' sigmoid
  probability is marginalized over the K=4 documents independently,
  weighted by softmax over the K documents' retrieval scores.

Tensorized cohort (both experiments): `/groups/mm6677_gp/zzw2102/data/MIMIC_MEDS_restored/processed`.

Val: `val/auroc/mean`, computed once at the end of fit against the tuning
split, `retriever.k=4` for `marginalized`. Test: `test/auroc/mean`,
`eval_mode=test` scoring each checkpoint's `checkpoints/last.ckpt` against
the held-out split; for `marginalized`, test uses top-1-only retrieval
(`retriever.k=1, ablation_mode=none`) in experiment 2, and the trained
`k=4` marginalization in experiment 1 (noted per table).

---

## Experiment 1: clean capacity-only sweep

Encoder capacity is varied while `max_seq_len` is held fixed at 256
(full patient history) throughout -- unlike experiment 2 below, this
isolates parameter/representational capacity from context-length cutoff.
`ff_dim = 2 x embedding_dim` at every level (matches the convention used
everywhere else in this repo).

| `embedding_dim` (D) | `num_heads` | `num_layers` | `ff_dim` | `max_seq_len` |
| --- | --- | --- | --- | --- |
| 4 | 1 | 1 | 8 | 256 |
| 8 | 1 | 1 | 16 | 256 |
| 16 | 1 | 1 | 32 | 256 |
| 32 | 1 | 1 | 64 | 256 |
| 64 | 1 | 1 | 128 | 256 |
| 128 | 4 | 2 | 256 | 256 |

`marginalized` test-split numbers below use `retriever.k=4` (the trained
marginalization), not top-1.

### Summary

| D | patient_only (val) | patient_only (test) | marginalized (val, k=4) | marginalized (test, k=4) | Δ (test) | Draws won (test) |
| --- | --- | --- | --- | --- | --- | --- |
| 4 | 0.8372 ± 0.0294 | 0.8199 ± 0.0238 | 0.8597 ± 0.0287 | 0.8523 ± 0.0216 | +0.0324 | 5/5 |
| 8 | 0.8511 ± 0.0273 | 0.8385 ± 0.0263 | 0.8657 ± 0.0268 | 0.8569 ± 0.0207 | +0.0184 | 5/5 |
| 16 | 0.8666 ± 0.0265 | 0.8562 ± 0.0226 | 0.8747 ± 0.0299 | 0.8681 ± 0.0251 | +0.0119 | 5/5 |
| 32 | 0.8777 ± 0.0341 | 0.8640 ± 0.0230 | 0.8864 ± 0.0326 | 0.8764 ± 0.0256 | +0.0124 | 5/5 |
| 64 | 0.8907 ± 0.0281 | 0.8804 ± 0.0209 | 0.8886 ± 0.0192 | 0.8838 ± 0.0217 | +0.0034 | 4/5 |
| 128 | 0.9037 ± 0.0246 | 0.9003 ± 0.0186 | 0.8968 ± 0.0273 | 0.8909 ± 0.0216 | -0.0094 | 1/5 |

### Per draw, D=4

| Draw | patient_only (val) | patient_only (test) | marginalized (val) | marginalized (test) | Δ (test) |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.8774 | 0.8450 | 0.8654 | 0.8663 | +0.0213 |
| 2 | 0.8307 | 0.8155 | 0.8507 | 0.8465 | +0.0310 |
| 3 | 0.8558 | 0.8382 | 0.8607 | 0.8614 | +0.0232 |
| 4 | 0.8324 | 0.8240 | 0.9056 | 0.8742 | +0.0502 |
| 5 | 0.7894 | 0.7771 | 0.8161 | 0.8131 | +0.0360 |

### Per draw, D=8

| Draw | patient_only (val) | patient_only (test) | marginalized (val) | marginalized (test) | Δ (test) |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.8806 | 0.8565 | 0.8770 | 0.8597 | +0.0032 |
| 2 | 0.8503 | 0.8307 | 0.8622 | 0.8498 | +0.0191 |
| 3 | 0.8552 | 0.8569 | 0.8663 | 0.8765 | +0.0196 |
| 4 | 0.8688 | 0.8582 | 0.9030 | 0.8775 | +0.0193 |
| 5 | 0.8008 | 0.7901 | 0.8202 | 0.8212 | +0.0311 |

### Per draw, D=16

| Draw | patient_only (val) | patient_only (test) | marginalized (val) | marginalized (test) | Δ (test) |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.8824 | 0.8678 | 0.8753 | 0.8782 | +0.0104 |
| 2 | 0.8666 | 0.8528 | 0.8618 | 0.8533 | +0.0005 |
| 3 | 0.8722 | 0.8748 | 0.8756 | 0.8816 | +0.0068 |
| 4 | 0.8946 | 0.8719 | 0.9264 | 0.8998 | +0.0279 |
| 5 | 0.8171 | 0.8137 | 0.8343 | 0.8275 | +0.0138 |

### Per draw, D=32

| Draw | patient_only (val) | patient_only (test) | marginalized (val) | marginalized (test) | Δ (test) |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.9026 | 0.8670 | 0.8847 | 0.8763 | +0.0093 |
| 2 | 0.8635 | 0.8503 | 0.8822 | 0.8607 | +0.0104 |
| 3 | 0.8695 | 0.8800 | 0.8891 | 0.8927 | +0.0127 |
| 4 | 0.9261 | 0.8942 | 0.9393 | 0.9133 | +0.0191 |
| 5 | 0.8269 | 0.8283 | 0.8365 | 0.8390 | +0.0107 |

### Per draw, D=64

| Draw | patient_only (val) | patient_only (test) | marginalized (val) | marginalized (test) | Δ (test) |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.8982 | 0.8854 | 0.8773 | 0.8901 | +0.0047 |
| 2 | 0.8721 | 0.8648 | 0.8840 | 0.8675 | +0.0027 |
| 3 | 0.8895 | 0.8955 | 0.8913 | 0.8922 | -0.0033 |
| 4 | 0.9385 | 0.9071 | 0.9235 | 0.9159 | +0.0088 |
| 5 | 0.8553 | 0.8494 | 0.8667 | 0.8532 | +0.0038 |

### Per draw, D=128 (full capacity)

| Draw | patient_only (val) | patient_only (test) | marginalized (val) | marginalized (test) | Δ (test) |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.8923 | 0.9060 | 0.9020 | 0.8932 | -0.0128 |
| 2 | 0.9095 | 0.8997 | 0.8802 | 0.8782 | -0.0215 |
| 3 | 0.9030 | 0.9056 | 0.8947 | 0.8977 | -0.0079 |
| 4 | 0.9446 | 0.9237 | 0.9444 | 0.9252 | +0.0015 |
| 5 | 0.8691 | 0.8668 | 0.8628 | 0.8604 | -0.0064 |

---

## Experiment 2: extreme capacity+context cut x data scarcity

Both `embedding_dim` and `max_seq_len` are cut together (bundled, unlike
experiment 1), crossed with 3 training-data levels.

| Param | Value |
| --- | --- |
| `encoder.embedding_dim` | 4 |
| `encoder.num_heads` | 1 |
| `encoder.num_layers` | 1 |
| `encoder.ff_dim` | 8 |
| `max_seq_len` | 8 |

Training-set size subsampled uniformly at random by row (fixed seed per
draw); `patient_only` and `marginalized` train on the identical
subsampled label file for a given (draw, fraction). Validation and test
splits are full-size at every data level.

| Train fraction | Approx. train rows (of ~153,195 full) |
| --- | --- |
| 100% | 153,195 |
| 50% | 76,598 |
| 5% | 7,660 |

`marginalized` test-split numbers below use top-1-only retrieval
(`retriever.k=1, ablation_mode=none`), not the trained k=4 marginalization.

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
