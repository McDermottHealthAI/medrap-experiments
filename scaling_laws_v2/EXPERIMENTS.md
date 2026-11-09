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

## Round 3: v1 fusion ablations (running)

Zero-init cross-attention output; encoder warm-started from patient only; centered keys (mean
pairwise cosine 0.22 → 0); all three together. Each with top and random documents, `W` 64x2x4 and
256x4x8, `D` 153K, 3 draws.
