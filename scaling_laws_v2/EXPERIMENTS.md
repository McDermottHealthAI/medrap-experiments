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
