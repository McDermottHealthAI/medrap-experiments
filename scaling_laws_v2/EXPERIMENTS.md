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
