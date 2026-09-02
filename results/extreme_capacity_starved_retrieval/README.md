# Capacity sweep retrieval (N=25, 30d)

Status: complete. Three experiments: (1) a clean 6-point capacity-only
sweep with context length held fixed, (2) an extreme capacity+context cut
crossed with 3 training-data levels (percentage-based), (3) the same
extreme capacity+context cut crossed with 4 absolute training-set sizes
(100/1,000/10,000/100,000 rows), with loss reported alongside AUROC.

Scripts: `mimic_iv_sweep/scripts/sweep_{patient_only,marginalized_binary_learned_linear}_capacity_sweep_n25_30d_dim{4,8,16,32,64}.sh`
and matching `eval_test_*` scripts (experiment 1); `sweep_{patient_only,marginalized_binary_learned_linear}_extreme_starved_n25_30d{,_train50pct,_train5pct}.sh`
and matching `eval_test_*` scripts (experiment 2); `sweep_{patient_only,marginalized_binary_learned_linear}_extreme_starved_n25_30d_trainN{100,1000,10000,100000}.sh`
and matching `eval_test_*` scripts (experiment 3). All in
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

---

## Experiment 3: extreme capacity x absolute training-set size (with loss)

Same capacity cut as Experiment 2 (`embedding_dim=4, num_heads=1,
num_layers=1, ff_dim=8, max_seq_len=8`), but the data axis is now exact
absolute training-row counts instead of percentages, and loss is reported
alongside AUROC. `train.parquet` is subsampled uniformly at random by row,
fixed seed per (level, draw), via `subsample_train_labels.py --n-rows`;
`patient_only` and `marginalized` train on the identical subsampled label
file per (draw, level).

| N (train rows) | `max_epochs` | `warmup_steps` |
| --- | --- | --- |
| 100 | 20 | 15 |
| 1,000 | 10 | 60 |
| 10,000 | 5 | 200 |
| 100,000 | 3 | 200 |

Epoch/warmup schedule is capped, not fully step-matched to the ~14,360
total optimizer steps of Experiment 1/2's full-data runs (which would need
~4,790 epochs at N=100) -- see the schedule table for exact per-level
values. `marginalized` test-split numbers use top-1-only retrieval
(`retriever.k=1, ablation_mode=none`), matching Experiment 2's convention.

### Summary

| N | patient_only (val AUROC) | patient_only (val loss) | patient_only (test AUROC) | patient_only (test loss) | marginalized (val AUROC) | marginalized (val loss) | marginalized (test AUROC) | marginalized (test loss) | Δ AUROC (test) | Draws won |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 100 | 0.5075 ± 0.0060 | 0.6474 ± 0.0015 | 0.5206 ± 0.0131 | 0.6476 ± 0.0015 | 0.5107 ± 0.0250 | 0.0685 ± 0.0205 | 0.5183 ± 0.0137 | 0.0685 ± 0.0203 | -0.0023 | 2/5 |
| 1,000 | 0.5115 ± 0.0118 | 0.3282 ± 0.0055 | 0.5326 ± 0.0078 | 0.3283 ± 0.0054 | 0.5358 ± 0.0283 | 0.0646 ± 0.0189 | 0.5580 ± 0.0284 | 0.0646 ± 0.0188 | +0.0254 | 4/5 |
| 10,000 | 0.5457 ± 0.0296 | 0.0713 ± 0.0180 | 0.5561 ± 0.0340 | 0.0714 ± 0.0178 | 0.7413 ± 0.0310 | 0.0562 ± 0.0159 | 0.7297 ± 0.0430 | 0.0562 ± 0.0158 | +0.1737 | 5/5 |
| 100,000 | 0.7709 ± 0.0334 | 0.0532 ± 0.0151 | 0.7652 ± 0.0343 | 0.0533 ± 0.0149 | 0.8182 ± 0.0283 | 0.0510 ± 0.0146 | 0.8109 ± 0.0286 | 0.0510 ± 0.0145 | +0.0457 | 5/5 |

### Per draw, N=100

| Draw | patient_only (val AUROC) | patient_only (val loss) | patient_only (test AUROC) | patient_only (test loss) | marginalized (val AUROC) | marginalized (val loss) | marginalized (test AUROC) | marginalized (test loss) | Δ (test AUROC) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.5087 | 0.6451 | 0.5230 | 0.6452 | 0.4978 | 0.0953 | 0.5072 | 0.0956 | -0.0158 |
| 2 | 0.4962 | 0.6494 | 0.5262 | 0.6497 | 0.4717 | 0.0347 | 0.5039 | 0.0355 | -0.0223 |
| 3 | 0.5106 | 0.6474 | 0.5117 | 0.6471 | 0.5160 | 0.0638 | 0.5369 | 0.0624 | +0.0252 |
| 4 | 0.5136 | 0.6469 | 0.5402 | 0.6475 | 0.5468 | 0.0655 | 0.5327 | 0.0665 | -0.0075 |
| 5 | 0.5087 | 0.6484 | 0.5018 | 0.6486 | 0.5214 | 0.0830 | 0.5108 | 0.0824 | +0.0090 |

### Per draw, N=1,000

| Draw | patient_only (val AUROC) | patient_only (val loss) | patient_only (test AUROC) | patient_only (test loss) | marginalized (val AUROC) | marginalized (val loss) | marginalized (test AUROC) | marginalized (test loss) | Δ (test AUROC) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.5157 | 0.3368 | 0.5310 | 0.3369 | 0.5420 | 0.0875 | 0.5754 | 0.0876 | +0.0444 |
| 2 | 0.4881 | 0.3198 | 0.5367 | 0.3201 | 0.4842 | 0.0334 | 0.5069 | 0.0339 | -0.0298 |
| 3 | 0.5201 | 0.3264 | 0.5341 | 0.3260 | 0.5695 | 0.0608 | 0.5813 | 0.0596 | +0.0472 |
| 4 | 0.5180 | 0.3288 | 0.5421 | 0.3292 | 0.5485 | 0.0602 | 0.5792 | 0.0614 | +0.0371 |
| 5 | 0.5157 | 0.3293 | 0.5188 | 0.3291 | 0.5347 | 0.0808 | 0.5471 | 0.0807 | +0.0283 |

### Per draw, N=10,000

| Draw | patient_only (val AUROC) | patient_only (val loss) | patient_only (test AUROC) | patient_only (test loss) | marginalized (val AUROC) | marginalized (val loss) | marginalized (test AUROC) | marginalized (test loss) | Δ (test AUROC) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.5742 | 0.0933 | 0.6156 | 0.0935 | 0.7710 | 0.0701 | 0.7766 | 0.0705 | +0.1610 |
| 2 | 0.4992 | 0.0409 | 0.5256 | 0.0417 | 0.7138 | 0.0299 | 0.6845 | 0.0307 | +0.1589 |
| 3 | 0.5556 | 0.0671 | 0.5627 | 0.0660 | 0.7535 | 0.0530 | 0.7713 | 0.0509 | +0.2086 |
| 4 | 0.5749 | 0.0698 | 0.5557 | 0.0707 | 0.7722 | 0.0529 | 0.7425 | 0.0536 | +0.1868 |
| 5 | 0.5244 | 0.0852 | 0.5207 | 0.0851 | 0.6960 | 0.0752 | 0.6739 | 0.0752 | +0.1532 |

### Per draw, N=100,000

| Draw | patient_only (val AUROC) | patient_only (val loss) | patient_only (test AUROC) | patient_only (test loss) | marginalized (val AUROC) | marginalized (val loss) | marginalized (test AUROC) | marginalized (test loss) | Δ (test AUROC) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.7806 | 0.0642 | 0.7924 | 0.0649 | 0.8359 | 0.0617 | 0.8439 | 0.0627 | +0.0515 |
| 2 | 0.7802 | 0.0288 | 0.7445 | 0.0297 | 0.8060 | 0.0273 | 0.7885 | 0.0281 | +0.0440 |
| 3 | 0.7747 | 0.0495 | 0.7936 | 0.0477 | 0.8197 | 0.0486 | 0.8175 | 0.0465 | +0.0239 |
| 4 | 0.8101 | 0.0503 | 0.7883 | 0.0511 | 0.8568 | 0.0475 | 0.8363 | 0.0481 | +0.0480 |
| 5 | 0.7090 | 0.0732 | 0.7072 | 0.0730 | 0.7728 | 0.0701 | 0.7684 | 0.0699 | +0.0612 |

### Positive-count caveat

At the two smallest levels, many of the 25 task codes have zero
training-set positives:

| N | Draw | # tasks with 0 positives (of 25) | min positives across tasks |
| --- | --- | --- | --- |
| 100 | 1 | 15 | 0 |
| 100 | 2 | 18 | 0 |
| 100 | 3 | 14 | 0 |
| 100 | 4 | 15 | 0 |
| 100 | 5 | 11 | 0 |
| 1,000 | 1 | 2 | 0 |
| 1,000 | 2 | 10 | 0 |
| 1,000 | 3 | 0 | 1 |
| 1,000 | 4 | 6 | 0 |
| 1,000 | 5 | 2 | 0 |
| 10,000 | 1-4 | 0 | 1 (draws 1,2,4), 4 (draw 3) |
| 10,000 | 5 | 1 | 0 |
| 100,000 | all | 0 | 4-51 |

`patient_only` and `marginalized` train on the identical subsampled label
file for a given (draw, N), so both architectures face the same
positive-count handicap per task.

### Note: loss/AUROC divergence at N=100

At N=100, `patient_only` test loss (0.6476, near `ln(2)=0.693`, the loss
of uninformative 50/50 predictions) is far higher than `marginalized` test
loss (0.0685), despite both architectures having near-chance test AUROC
(0.5206 vs. 0.5183). Given the high fraction of zero-positive tasks at this
level (see caveat table above), this reflects `patient_only` producing
high-entropy predictions while `marginalized` produces confident-but-still
uninformative ones -- low loss here does not imply better discrimination.
