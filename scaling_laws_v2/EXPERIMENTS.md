# Experiment log

Goal and rules: [`GOAL.md`](GOAL.md). Newest round last. `Δ` = L(patient only) − L(config), % of
patient only's test loss; positive means retrieval is better.

## Round 1: Stage 0 and the Stage 1 ladder (lr 1e-3, K 4, E0)

- **Hypothesis**: v1 loses to patient only because of the padding bug, the train/test K mismatch or the fusion design.
- **Result**: v1 Δ −12% to −23% at 153K. Testing at K = 4 recovers about a third of that; the masked mean alone changes little. The gated fusion reaches Δ ≈ 0, but its gate stays at ~0.01, so it does not read the documents (not allowed by the rules).
- **Also**: the gradient norm is ~0.01 and clipping never triggers; both losses are correct; v1 is worse on training loss too (it underfits, not overfits); the query gradient is ~1e-4, against ~1e-2 for the encoder; 45–80% of test patients share one top passage.
- **Decision**: keep the padding and K fixes; drop the gate; find what in the always-read v1 fusion stops it from fitting.

## Round 2: no averaging over documents (K 1, no null document)

- **Hypothesis**: averaging the per-document probabilities blurs predictions.
- **Result**: v1 with random documents at 64x2x4, 153K gives test loss 0.02479 at K 1 with no null, 0.02452 at K 4, against patient only's 0.02164. Still 15% worse.
- **Decision**: averaging is not the cause.

## Round 3: v1 fusion ablations (D 153K, lr 1e-3, 3 draws)

- **Hypothesis**: the documents drown the patient signal at the start (zero-init), the encoder learns
  while being fed noise (warm start), or the search returns the same passages for everyone (centered keys).
- **Result** (topK test loss; Δ vs patient only at lr 1e-3, positive = better):

  | | 64x2x4 | Δ % | 256x4x8 | Δ % |
  |---|---|---|---|---|
  | patient only, lr 1e-3 | 0.02164 | | 0.02183 | |
  | patient only, best lr (Stage 0) | 0.02127 | | 0.02077 | |
  | v1_masked | 0.02456 | −13.4 | 0.02579 | −18.1 |
  | + zero-init | | −15.8 | | −18.3 |
  | + centered keys | | −12.6 | | −15.3 |
  | **+ warm encoder** | 0.02110 | **+2.5** (3/3 draws) | 0.02140 | **+2.0** (3/3) |
  | + all three | 0.02105 | +2.7 | 0.02145 | +1.8 |
  | warm, trained on random documents | 0.02111 | +2.5 | 0.02147 | +1.7 |

  Paired randomK − topK (×1e-4) for the warm model: 0.69 ± 0.75 (64x2x4), 2.66 ± 3.05 (256x4x8); positive in every draw, but the interval includes 0.
- **Reading**: the cold start is the cause of the harm. With a trained encoder, reading documents no
  longer hurts, and zero-init and centered keys add nothing. But the warm model gets 3 more epochs than
  the baseline, and the random-documents model gains just as much, so the gain is the extra training,
  not the documents. G2 fails: the documents are nearly unused. At 256x4x8 the warm model is still 3%
  worse than patient only at its best learning rate.
- **Decision**: keep the warm start. Run the fair baseline (patient only continued from the same
  checkpoint for the same epochs, `./submit.sh warm`) and the D 10K cells, then test whether the reader
  can use documents at all (positive control).

## Round 4: warm start against its fair baseline (lr 1e-3, 3 draws)

- **Hypothesis**: Round 3's +2% is the 3 extra epochs, not the documents.
- **Baseline**: patient only continued from the same checkpoint for the same epochs (`patient_only_warm`):
  0.02499 / 0.02113 (64x2x4, D 10K / 153K), 0.02461 / 0.02128 (256x4x8). This alone is 1.5–2.6% better than
  the paired baseline used before, so the earlier +2% was the extra training.
- **Result** (`Δ` ×1e-4 vs `patient_only_warm`, 95% interval; positive = retrieval better):

  | | 64/10K | 64/153K | 256/10K | 256/153K |
  |---|---|---|---|---|
  | warm, topK | −0.63 ± 1.52 | +0.30 ± 1.38 | −2.50 ± 3.13 | −1.25 ± 2.47 |
  | warm, randomK at test | −2.80 ± 2.23 | −0.39 ± 1.03 | −4.11 ± 6.13 | −3.91 ± 2.17 |
  | warm, trained on random documents | −1.02 ± 2.18 | +0.21 ± 1.48 | −1.69 ± 2.35 | −1.94 ± 1.14 |

- **G1 (no harm)**: met; every topK interval includes 0.
- **G2 (uses the documents)**: half met. The retrieved documents beat random ones at test in all 4 cells
  (by 0.7–2.7 ×1e-4), but a model trained on random documents does as well, so what it reads is not
  knowledge yet.
- **G3 (helps)**: not met; `Δ` ≈ 0.
- **Decision**: the warm start is the new default (and its baseline is `patient_only_warm`). The reader no
  longer harms; the question is now whether it *can* gain from a document. Next: the positive control.

## Round 5: contextual documents, all tokens of all layers (D 153K, warm start, 3 draws)

- **Hypothesis** (advisor): the documents add nothing because the fusion reads context-free word
  pieces. Run Qwen over each document, take all layers, and attend through them.
- **Change**: `DOCS=attend` (MedRAP `QwenHiddenStatesRetrievalEncoder`): a frozen Qwen3-Embedding-0.6B
  runs over each retrieved passage; the fusion cross-attends over 256 tokens × 29 hidden states, each
  layer-normalised, plus a learned layer embedding. Everything else as Round 4. About 7.5–9 h per run
  on an L40S/H100, against about 2 h before.
- **Result** (`Δ` ×1e-4 vs `patient_only_warm`, 95% interval; positive = retrieval better):

  | | 64x2x4 | 256x4x8 |
  |---|---|---|
  | E0 (Round 4), topK | +0.30 ± 1.38 | −1.25 ± 2.47 |
  | **all layers, topK** | **+0.77 ± 0.65** (3/3 draws; +0.36%) | −1.10 ± 2.93 (−0.54%) |
  | all layers, random documents at test | −0.03 ± 1.19 | −3.39 ± 3.61 |
  | all layers, trained on random documents | +0.33 ± 0.56 | −1.48 ± 2.97 |

  | Uses the documents? (×1e-4, positive = retrieved better) | 64x2x4 | 256x4x8 |
  |---|---|---|
  | randomK − topK, same model, E0 | 0.69 ± 0.76 | 2.66 ± 3.06 |
  | randomK − topK, same model, all layers | 0.80 ± 1.28 | **2.29 ± 2.02** |
  | trained on random − trained on retrieved, all layers | 0.44 ± 0.88 (3/3) | **0.38 ± 0.21** (3/3) |

  Share of test patients with the most common top passage: 0.51 / 0.48 (E0) → 0.35 / 0.38 (all layers).
- **Reading**: the first positive signs, all small.
  - **G1 (no harm):** still met.
  - **G2 (uses the documents):** met at 256x4x8, where retrieved passages beat random ones at test and
    beat a model trained on random passages, both with intervals excluding 0. At 64x2x4 both point
    the same way (3/3 draws) but the intervals include 0.
  - **G3 (helps):** at 64x2x4/153K, the first `Δ` with an interval excluding 0 (+0.36%). Most of that
    comes from the architecture, since the random-documents model is +0.16%. Needs 2 of 4 cells.
  - Retrieval is also more diverse.

  The effects are about 0.2–0.4% of the loss: real, but far from useful yet.
- **Decision**: keep the contextual documents. Run the D 10K cells (submitted) to complete the 4-cell
  check. Then more documents per patient (the advisor's K 128/512); with all layers that is too
  expensive to compute on the fly, so it needs precomputed last-layer vectors or fewer layers.
- **D 10K cells** (same setup; `Δ` ×1e-4 vs `patient_only_warm`):

  | | 64x2x4 / 10K | 256x4x8 / 10K |
  |---|---|---|
  | E0 (Round 4), topK | −0.63 ± 1.52 | −2.50 ± 3.13 |
  | all layers, topK | −0.94 ± 2.40 (−0.36%) | −5.77 ± 19.16 (−2.2%) |
  | all layers, random documents at test | −3.46 ± 2.92 | −11.21 ± 33.82 |
  | all layers, trained on random documents | −0.84 ± 2.17 | −2.48 ± 2.12 |
  | randomK − topK, same model | **2.52 ± 0.86** | 5.44 ± 14.72 |

  At 256x4x8/10K one draw drives the wide interval: draw 3 is 5.5% worse than its baseline
  (0.02777 against 0.02631), draws 1–2 are 0.4–0.8% worse. With 10K patients the larger model
  overfits through the retrieved documents (it is worse than the same model on random documents in
  that draw).
- **Reading, all 4 cells**: G1 still met (every interval includes 0). G1+ not met: at 153K,
  256x4x8 is −0.54%. G2 met only where the same-model swap excludes 0 (64x2x4/10K, 256x4x8/153K);
  the model trained on random documents beats retrieval at D 10K. G3: 1 of 4 cells. Contextual
  documents help at large `D` and small `W`, and hurt at small `D`.
- **Decision**: the precomputed last-layer vectors (`DOCS=pre`, ~30 min per run, against 8 h) gave the
  same result as all layers at K 4 (64x2x4/153K: +0.20 ± 1.43; 256x4x8: −1.81 ± 2.90, 2 draws), so
  the K sweep (Round 6) uses them.

## Round 6: more documents per patient, and each W at its tuned lr

- **Hypothesis A** (advisor): with 4 candidates the retriever learns little about which passage
  helps; K 32 and 128 give it more to compare. `./submit.sh kdocs`, precomputed last layer, D 153K.
- **Hypothesis B**: 256x4x8 trails because lr 1e-3 is too high for it (its Stage 0 optimum is 3e-4);
  both arms at each W's tuned lr, warm-started from that lr's checkpoint. `./submit.sh tuned`.
- **Result A** (D 153K, v1 reader, `Δ` % vs `patient_only_warm`; in brackets the same model trained
  on random documents):

  | K | 64x2x4 | 256x4x8 |
  |---|---|---|
  | 4 | +0.10 (−0.05) | −0.64 (−0.85) |
  | 32 | +0.32 (+0.33) | −0.40 (−0.39) |
  | 128 | +0.45 (+0.43) | −0.52 (−0.73) |

  The model reads more with more documents (randomK − topK at K 32: 0.84 ± 0.36 and 3.0 ± 3.1 ×1e-4),
  but the model trained on random documents gains just as much: the gain is capacity, not knowledge.
- **Result B**: worse at the tuned lr: 64x2x4 at 3e-3 −1.6%, 256x4x8 at 3e-4 −0.74% (random documents
  −1.56% and −0.88%).
- **Decision**: drop both. K stays 4 and lr 1e-3 (the gap at 256x4x8 is not the number of documents
  or the learning rate).

## Round 7: start exactly at the patient-only model, add documents with no gate

- **Hypothesis**: the warm start copies only the encoder. `patient_only_warm`'s fresh head reads the
  trained encoder's pooled vector `h` directly; the v1 fusion's head reads it only through a fresh
  projection and two cross-attention layers, which have to relearn the patient signal. That costs
  most where data is scarce (D 10K), and it explains why random documents train as well.
- **Change**: `residual` (MedRAP `GatedDocFusion(gated=False)`): the output for document `k` is
  `h + c_k`, where `c_k` is that document's cross-attention contribution. No gate or scalar can turn
  the documents off; every document always changes the output. Control: `residual_random_docs`.
  Precomputed last layer, K 4, lr 1e-3, both W and both D. `./submit.sh residual`.
- **Result** (`Δ` % vs `patient_only_warm`): −0.22 (64/10K), **+0.14** (64/153K, 0.28 ± 0.20 ×1e-4,
  3/3 draws), −0.35 (256/10K), −0.12 (256/153K). Trained on random documents is worse in every cell
  (by 0.2–1.4 ×1e-4; interval excludes 0 at 64/153K).
- **Reading**: better than v1 at 256x4x8/153K (−0.12% against −0.64%), but this one-query reader is weak.

## Round 8: the per-code reader in residual form

- **Hypothesis**: Round 7's `residual` reads each passage with one pooled query vector, a weaker
  reader than v1's, where every patient code cross-attends to the passage. Keep v1's reader, but on
  the encoder's own states (no projection, no output norm), so the pooled output is `h` plus the
  mean of the documents' updates and the head reads the patient-only input.
- **Change**: `xres` (MedRAP `PerDocCrossAttentionFusion(residual=True)`, `d_model` = W, the
  encoder's head count, `ff_dim` 2W, 2 layers). Control: `xres_random_docs`. Precomputed last layer,
  K 4, lr 1e-3, both W and both D. `CONFIGS="xres xres_random_docs" ./submit.sh residual`.
- **Result** (`Δ` ×1e-4 vs `patient_only_warm`, 95% interval):

  | | 64/10K | 64/153K | 256/10K | 256/153K |
  |---|---|---|---|---|
  | **xres, topK** | −0.48 ± 0.86 (−0.19%) | **+0.58 ± 0.54** (+0.27%, 3/3) | −0.62 ± 3.15 (−0.25%) | **+1.61 ± 1.41** (+0.75%, 3/3) |
  | xres trained on random documents | −1.92 ± 0.44 | +0.01 ± 0.75 | −0.93 ± 2.23 | +0.67 ± 1.69 |
  | trained on random − retrieved | **1.43 ± 0.42** | **0.57 ± 0.23** | 0.31 ± 2.69 | **0.94 ± 0.76** |
  | randomK − topK, same model | **3.01 ± 0.88** | **1.95 ± 1.23** | 0.90 ± 2.39 | 1.24 ± 1.51 |

- **Goals**: G1 met (every interval includes 0 or is positive). **G1+ met** (both W positive at
  153K). **G3 met** (intervals exclude 0 at 64/153K and 256/153K). G2: retrieved documents beat the
  random-documents model in 3 of 4 cells, the first sign they carry knowledge; the same-model swap
  excludes 0 at 64x2x4 only.
- **Validation check** (settings must be chosen on validation, and xres was picked after seeing test
  results for several readers): on validation loss xres is the best reader at 256x4x8 (+0.80% at
  153K, interval excludes 0; −0.04% at 10K), but at 64x2x4 it is middling (+0.13% at 153K, interval
  includes 0). So the 256x4x8/153K gain is solid; the 64x2x4/153K one is fragile.
- **Decision**: xres is the reader. Confirm it on fresh draws 4–6 (Round 9), on which nothing has been
  chosen, before building on it.

## Round 9: xres on fresh draws

- **Hypothesis**: Round 8's gain is real, not a choice made on draws 1–3.
- **Change**: new tasks, patients and seeds (draws 4–6); patient only (cold, then warm), xres and
  xres trained on random documents, both W and both D. `DRAW_LIST="4 5 6" ./submit.sh confirm`.
- **Result** (`Δ` % vs `patient_only_warm`; ×1e-4 with 95% interval for draws 1–6 pooled):

  | | draws 1–3 | draws 4–6 | **draws 1–6** |
  |---|---|---|---|
  | 64x2x4 / 10K | −0.19% | −0.41% | **−0.73 ± 0.46** (−0.30%) |
  | 64x2x4 / 153K | +0.27% | +0.30% | **+0.60 ± 0.20** (+0.28%, 5/5; one draw re-running) |
  | 256x4x8 / 10K | −0.25% | −0.71% | **−1.17 ± 1.12** (−0.48%) |
  | 256x4x8 / 153K | +0.75% | +0.83% | **+1.72 ± 1.21** (+0.79%, 6/6) |

  Uses the documents, draws 1–6 (×1e-4): trained on random − retrieved 1.40 ± 0.36, 0.64 ± 0.16,
  0.60 ± 1.16, 1.03 ± 0.67 (same cell order); randomK − topK 3.63 ± 1.04, 1.96 ± 0.54, 1.03 ± 0.70,
  1.19 ± 0.50.
- **Reading**: the large-`D` gain replicates on fresh draws with the same size (+0.3% and +0.8%).
  G1+ and G3 hold. The documents carry knowledge in every cell (G2: the swap excludes 0 in all 4,
  the random-documents control in 3). But with 6 draws the D 10K harm is significant: G1 fails at
  10K. There the random-documents model is worse still (−0.9% and −0.7%), so the reader itself
  overfits with little data.
- **Decision**: try regularising the reader (Round 10). Start the scaling grid's patient-only runs,
  which don't depend on the reader.

## Round 10: dropout in the reader

- **Hypothesis**: the D 10K harm is the reader overfitting; dropout in the reader alone (not the
  encoder, so the warm start is unchanged) fixes it without losing the 153K gain.
- **Change**: `FUSION_DROPOUT` 0.1 and 0.3 for xres, both W and both D, draws 1–3.
  `FUSION_DROPOUT=0.1 CONFIGS=xres ./submit.sh residual`.
- **Result** (`Δ` % vs `patient_only_warm`, draws 1–3; reader dropout 0 / 0.1 / 0.3):
  64x2x4/10K −0.19 / −0.53 / −0.85; 64x2x4/153K +0.27 / +0.24 / +0.24; 256x4x8/10K −0.25 / −0.15 /
  −0.52; 256x4x8/153K +0.75 / +0.68 / +0.71.
- **Reading**: dropout does not fix D 10K and makes 64x2x4 worse, so the small-`D` cost is not the
  reader overfitting.
- **Decision**: keep dropout 0. Run the scaling grid with xres as it is; it maps `Δ` against `D`
  (where retrieval starts to help) and against compute.

## Round 11: the scaling grid

`./submit.sh grid`: `W` {64x2x4, 128x2x4, 256x4x8} × `D` {3K, 10K, 30K, 100K, 153K} × 3 draws (6 at
the Round 9 cells); per cell patient only, `patient_only_warm`, xres and xres trained on random
documents. Analysis: `grid.py` → `grid_delta.csv`, `grid_fits.csv`, `grid_compare.csv`,
`grid_delta.png`, `grid_compute.png`.

- **Δ against D** (% of patient only's test loss, 95% t-interval; positive: retrieval better):

  | `D` | 64x2x4 | 128x2x4 | 256x4x8 |
  |---|---|---|---|
  | 3K | −1.55 ± 0.40 | −2.85 ± 1.15 | −1.88 ± 1.89 |
  | 10K | −0.30 ± 0.20 | −1.50 ± 1.32 | −0.48 ± 0.44 |
  | 30K | −0.24 ± 0.58 | −0.65 ± 1.09 | −0.05 ± 2.15 |
  | 100K | **+0.21 ± 0.23** (3/3) | −0.04 ± 0.58 | **+0.82 ± 0.37** (3/3) |
  | 153K | **+0.27 ± 0.08** (6/6) | +0.13 ± 0.89 | **+0.79 ± 0.50** (6/6) |

  Trained on random documents: −2.0 to −2.9% at 3K, rising to −0.02 / −0.01 / +0.31% at 153K.
  The documents' own contribution (xres − random) grows with `D` too: about +0.3 to +0.5% at 100K
  and 153K.
- **Laws** (fit on draw-centred loss; one form for all arms): separable-power,
  `L = E + A·N^−α + B·D^−β`, is kept. It passes the quality gate for patient only (R² over cell
  means 0.994, leave-one-cell-out RMSE 2.2 × 10⁻⁴ < run SD 2.3 × 10⁻⁴) and xres (0.996; 2.2 < 2.4),
  and narrowly fails it for the random-documents arm (2.20 vs 2.18). The other four forms fail on
  `ill_conditioned_fit` or optimizer warnings. The arm comparison from the laws is inconclusive
  (P(xres better) 0.5–0.6 at large `D`): unpaired laws cannot resolve a 0.3–0.8% difference, and an
  additive law cannot represent a gain that grows with `D`. The paired cell table is the evidence.
- **Compute**: at equal training compute patient only wins. Its best cell (128x2x4, 153K) reaches
  0.02073 at 5.9 × 10¹⁴ FLOPs; xres's best (128x2x4, 153K) reaches 0.02071 at 3.8 × 10¹⁵, the same
  loss for 6.5× the compute. Most of a retrieval step projects the 5 × 256 passage vectors.
- **Reading**: retrieval is a large-data effect. It costs 1.5–3% at 3K patients, breaks even around
  30K–100K and helps from 100K on, and the documents' own contribution grows with `D`. It is the
  best model at a fixed amount of data, not at a fixed compute budget.

Also finished: retrieval only (the patient reaches the prediction only through the documents):
−5.5% / −7.0% at 10K and −15.7% / −13.9% at 153K (64x2x4 / 256x4x8) against patient only. xres at
K 32: +0.34% at 64x2x4/153K (3/3, random-documents control +0.07%), +0.88% at 256x4x8 (1 draw so far).
