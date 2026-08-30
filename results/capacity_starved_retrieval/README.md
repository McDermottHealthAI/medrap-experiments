# Capacity-starved `patient_only` vs. marginalized retrieval (N=25, 30d)

Status: complete. 15 val jobs (5 draws x 3 architectures), 15 test-split
jobs, 60 larger-k jobs (learned-linear, k=32/64/128), 10 top1-only jobs
(learned-linear + qwen3_text), plus random-doc/null-doc ablations (separate
README, linked below).

Scripts: [`sweep_patient_only_capacity_starved_n25_30d.sh`](../../mimic_iv_sweep/scripts/sweep_patient_only_capacity_starved_n25_30d.sh),
[`sweep_marginalized_binary_qwen3_text_capacity_starved_n25_30d.sh`](../../mimic_iv_sweep/scripts/sweep_marginalized_binary_qwen3_text_capacity_starved_n25_30d.sh),
[`sweep_marginalized_binary_learned_linear_capacity_starved_n25_30d.sh`](../../mimic_iv_sweep/scripts/sweep_marginalized_binary_learned_linear_capacity_starved_n25_30d.sh)
(training, `mimic_iv_sweep/`); [`mimic_iv_null_random_doc_ablations/scripts/`](../../mimic_iv_null_random_doc_ablations/README.md)
(test-split repro, top1-only eval, random/null-doc ablations).

## Task

Multi-task binary classification: N=25 randomly-sampled diagnostic codes,
predict occurrence within a 30-day window after a prediction anchor time.
5 draws, 3 training epochs.

## Config

**Patient encoder** (`TimeDeltaRoPEPatientEncoder`): token embedding
(`nn.Embedding(vocab_size, D)`) → time-delta RoPE self-attention
(position = cumulative log time gap between events, not sequence index) →
`num_layers` transformer blocks (self-attention + GELU FFN, `ff_dim`) →
final LayerNorm. Identical architecture in both configurations; only size
differs between full-capacity and starved.

| Param | Full capacity | Starved (this experiment) |
| --- | --- | --- |
| `encoder.embedding_dim` | 128 | 16 |
| `encoder.num_heads` | 4 | 1 |
| `encoder.num_layers` | 2 | 1 |
| `encoder.ff_dim` | 256 | 32 |
| `max_seq_len` | 256 | 32 |

**`patient_only`** (`fusion=passthrough`): masked-mean pool over the
sequence → `Linear(D, 25)`. No retrieval.

**`marginalized`**: query projector → `HFDatasetRetriever` (FAISS top
K=4 over 125k `MedRAG/textbooks` passages, pre-embedded with frozen
`Qwen3-Embedding-0.6B`) → `TokenFeatureRetrievalEncoder`
(`nn.Embedding(151936, 64)` on retrieved-doc tokens) →
`PerDocCrossAttentionFusion` (2 cross-attention layers, `d_model=256`, 8
heads, `ff_dim=512`, one 256-dim output per retrieved doc) →
`Linear(256, 25)` applied per doc → `marginalized_output_mode=binary`
(per-task sigmoid probability marginalized over K, weighted by softmax
over the K retrieval scores).

Two query projectors:

- **`qwen3_text`**: patient's recent event codes rendered as text,
  embedded with the same frozen `Qwen3-Embedding-0.6B` used for the
  corpus. Not a function of the patient encoder's hidden state --
  unaffected by the capacity cut.
- **`sequence_mean_1024`** ("learned-linear"): mean-pools the patient
  encoder's own hidden state, then a trainable `Linear(D, 1024)`.
  Capacity-cut along with the rest of the model (`D`: 128 → 16).

## Results, val split (`val/auroc/mean`, k=4)

| Draw | patient_only | learned-linear | qwen3_text |
| --- | --- | --- | --- |
| 1 | 0.8565 | 0.8583 | 0.8725 |
| 2 | 0.8202 | 0.8400 | 0.8525 |
| 3 | 0.8534 | 0.8567 | 0.8638 |
| 4 | 0.8805 | 0.9066 | 0.9204 |
| 5 | 0.7904 | 0.8023 | 0.8160 |
| **mean ± std** | **0.8402 ± 0.0352** | **0.8528 ± 0.0378** | **0.8650 ± 0.0392** |

Δ (marginalized − patient_only): learned-linear +0.0125 ± 0.0104 (5/5
draws), qwen3_text +0.0248 ± 0.0119 (5/5 draws).

Full-capacity `patient_only` 30d mean (from `mimic_iv_sweep/README.md`):
0.8949 ± 0.0220.

## Results, larger k (learned-linear, val split)

Scope: learned-linear only. `qwen3_text`'s live frozen-Qwen3 forward pass
is ~3.5x slower per job regardless of k, so the k-sweep was not repeated
for it.

| Draw | patient_only | k=32 | k=64 | k=128 |
| --- | --- | --- | --- | --- |
| 1 | 0.8565 | 0.8772 | 0.8586 | 0.8854 |
| 2 | 0.8202 | 0.8421 | 0.8423 | 0.8451 |
| 3 | 0.8534 | 0.8620 | 0.8599 | 0.8581 |
| 4 | 0.8805 | 0.9156 | 0.9071 | 0.9086 |
| 5 | 0.7904 | 0.8087 | 0.8085 | 0.8146 |

| k | Δ (marginalized − patient_only), mean ± std | draws won |
| --- | --- | --- |
| 4 | +0.0125 ± 0.0104 | 5/5 |
| 32 | +0.0209 ± 0.0095 | 5/5 |
| 64 | +0.0151 ± 0.0104 | 5/5 |
| 128 | +0.0222 ± 0.0100 | 5/5 |

## Results, test split (`test/auroc/mean`, k=4)

`eval_mode=test` scores each checkpoint's `checkpoints/last.ckpt` on the
MEDS `held_out` split (no retraining).

| Draw | patient_only | learned-linear | qwen3_text |
| --- | --- | --- | --- |
| 1 | 0.8635 | 0.8581 | 0.8723 |
| 2 | 0.8098 | 0.8276 | 0.8335 |
| 3 | 0.8503 | 0.8596 | 0.8653 |
| 4 | 0.8715 | 0.8881 | 0.8934 |
| 5 | 0.7950 | 0.8050 | 0.8224 |
| **mean ± std** | **0.8380 ± 0.0338** | **0.8477 ± 0.0317** | **0.8574 ± 0.0292** |

| Query projector | Val Δ (mean ± std, draws won) | Test Δ (mean ± std, draws won) |
| --- | --- | --- |
| learned-linear | +0.0125 ± 0.0104, 5/5 | +0.0096 ± 0.0092, 4/5 |
| qwen3_text | +0.0248 ± 0.0119, 5/5 | +0.0193 ± 0.0074, 5/5 |

## Results, top1-only inference (test split, `retriever.k=1`, `ablation_mode=none`)

Same checkpoints, evaluated with only the single top-retrieved document
instead of marginalizing over K=4 (valid for any checkpoint regardless of
trained k -- `PerDocCrossAttentionFusion` and
`marginalized_output_mode=binary` are both K-agnostic).

| Draw | learned-linear (k=1) | qwen3_text (k=1) |
| --- | --- | --- |
| 1 | 0.8500 | 0.8659 |
| 2 | 0.8240 | 0.8216 |
| 3 | 0.8428 | 0.8605 |
| 4 | 0.8819 | 0.8904 |
| 5 | 0.7985 | 0.8178 |
| **mean ± std** | **0.8394 ± 0.0277** | **0.8512 ± 0.0277** |

Δ (k=1 − k=4 real docs): learned-linear −0.0083 (5/5 draws lower),
qwen3_text −0.0062 (5/5 draws lower).

## Results, random-doc / null-doc ablations (test split, k=4)

Full tables: [`results/random_doc_null_doc_ablations_mimic_iv/README.md`](../random_doc_null_doc_ablations_mimic_iv/README.md).

| Variant | patient_only | real docs | random docs | null docs |
| --- | --- | --- | --- | --- |
| learned_linear | 0.8380 ± 0.0302 | 0.8477 ± 0.0287 | 0.8126 ± 0.0415 | 0.7859 ± 0.0376 |
| qwen3_text | 0.8380 ± 0.0302 | 0.8574 ± 0.0260 | 0.8473 ± 0.0264 | 0.8087 ± 0.0721 |

NWICU replication: [`results/random_doc_null_doc_ablations_nwicu/README.md`](../random_doc_null_doc_ablations_nwicu/README.md).
