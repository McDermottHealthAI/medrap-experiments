# Extreme-starved capacity floor retrieval (N=25, 30d)

Status: complete. 6 configs (patient_only x 3 data fractions, marginalized
x 3 data fractions), 5 draws each, val + test split.

Scripts: [`sweep_patient_only_extreme_starved_n25_30d.sh`](../../mimic_iv_sweep/scripts/sweep_patient_only_extreme_starved_n25_30d.sh),
[`sweep_marginalized_binary_learned_linear_extreme_starved_n25_30d.sh`](../../mimic_iv_sweep/scripts/sweep_marginalized_binary_learned_linear_extreme_starved_n25_30d.sh),
and `_train50pct`/`_train5pct` variants of both (`mimic_iv_sweep/scripts/`).

## Task

Multi-task binary classification: N=25 randomly-sampled diagnostic codes,
predict occurrence within a 30-day window after a prediction anchor time.
5 draws, 3 training epochs.

## Config

| Param | Full | Starved (`capacity_starved_retrieval`) | Extreme-starved (this experiment) |
| --- | --- | --- | --- |
| `encoder.embedding_dim` | 128 | 16 | 4 |
| `encoder.num_heads` | 4 | 1 | 1 |
| `encoder.num_layers` | 2 | 1 | 1 |
| `encoder.ff_dim` | 256 | 32 | 8 |
| `max_seq_len` | 256 | 32 | 8 |

`patient_only` (`fusion=passthrough`): masked-mean pool → `Linear(4, 25)`.
No retrieval.

`marginalized`: `query_projector=sequence_mean_1024` (`in_dim=4`) →
`HFDatasetRetriever` (FAISS, K=4, full 125k-passage `MedRAG/textbooks`
corpus) → `TokenFeatureRetrievalEncoder` → `PerDocCrossAttentionFusion`
(`d_model=256`) → `Linear(256, 25)` → `marginalized_output_mode=binary`.

Training-data axis: 3 levels, reusing the subsampled label sets from
`results/data_scarcity_retrieval/`.

| Train fraction | Approx. train rows |
| --- | --- |
| 100% | ~153,195 |
| 50% | 76,598 |
| 5% | 7,660 |

Tensorized cohort: `/groups/mm6677_gp/zzw2102/data/MIMIC_MEDS_restored/processed`
(restored copy; original path's `processed/` directory is absent).

## Results, val split (`val/auroc/mean`)

| Data | patient_only | marginalized |
| --- | --- | --- |
| 100% | 0.7911 ± 0.0314 | 0.8286 ± 0.0300 |
| 50% | 0.7506 ± 0.0333 | 0.8051 ± 0.0221 |
| 5% | 0.5255 ± 0.0164 | 0.6829 ± 0.0472 |

### Per draw

**100% data**

| Draw | patient_only | marginalized |
| --- | --- | --- |
| 1 | 0.8032 | 0.8425 |
| 2 | 0.8034 | 0.8150 |
| 3 | 0.7991 | 0.8299 |
| 4 | 0.8199 | 0.8733 |
| 5 | 0.7300 | 0.7825 |

**50% data**

| Draw | patient_only | marginalized |
| --- | --- | --- |
| 1 | 0.7729 | 0.8198 |
| 2 | 0.7302 | 0.7893 |
| 3 | 0.7636 | 0.8061 |
| 4 | 0.7896 | 0.8366 |
| 5 | 0.6965 | 0.7738 |

**5% data**

| Draw | patient_only | marginalized |
| --- | --- | --- |
| 1 | 0.5238 | 0.7113 |
| 2 | 0.5043 | 0.6166 |
| 3 | 0.5263 | 0.6989 |
| 4 | 0.5546 | 0.7460 |
| 5 | 0.5186 | 0.6416 |

## Results, test split (`test/auroc/mean`, `checkpoints/last.ckpt`, `eval_mode=test`)

| Data | patient_only | marginalized | Δ (marginalized − patient_only) | Draws won |
| --- | --- | --- | --- | --- |
| 100% | 0.7846 ± 0.0312 | 0.8204 ± 0.0301 | +0.0358 | 5/5 |
| 50% | 0.7486 ± 0.0391 | 0.8050 ± 0.0256 | +0.0564 | 5/5 |
| 5% | 0.5334 ± 0.0123 | 0.6607 ± 0.0535 | +0.1273 | 5/5 |

### Per draw

**100% data**

| Draw | patient_only | marginalized | Δ |
| --- | --- | --- | --- |
| 1 | 0.8138 | 0.8515 | +0.0377 |
| 2 | 0.7716 | 0.7900 | +0.0184 |
| 3 | 0.8080 | 0.8364 | +0.0284 |
| 4 | 0.8002 | 0.8456 | +0.0454 |
| 5 | 0.7294 | 0.7785 | +0.0491 |

**50% data**

| Draw | patient_only | marginalized | Δ |
| --- | --- | --- | --- |
| 1 | 0.7851 | 0.8273 | +0.0422 |
| 2 | 0.7136 | 0.7866 | +0.0730 |
| 3 | 0.7818 | 0.8221 | +0.0403 |
| 4 | 0.7726 | 0.8258 | +0.0532 |
| 5 | 0.6901 | 0.7636 | +0.0735 |

**5% data**

| Draw | patient_only | marginalized | Δ |
| --- | --- | --- | --- |
| 1 | 0.5422 | 0.7137 | +0.1715 |
| 2 | 0.5287 | 0.5811 | +0.0524 |
| 3 | 0.5421 | 0.7073 | +0.1652 |
| 4 | 0.5428 | 0.6879 | +0.1451 |
| 5 | 0.5111 | 0.6135 | +0.1024 |
