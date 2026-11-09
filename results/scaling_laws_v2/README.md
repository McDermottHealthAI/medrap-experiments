# Scaling laws v2: Stage 0 and Stage 1 results

Design: [`scaling_laws_v2/design.md`](../../scaling_laws_v2/design.md). Made by
[`ladder.py`](../../scaling_laws_v2/ladder.py) from 288 test evaluations
([`results.csv`](results.csv)). `T` = 1,000 tasks, 3 draws, K = 4 + null document.

## Stage 1: paired difference from patient only (lr 1e-3, dropout 0)

`Δ` = test loss of patient only − test loss of the configuration, as % of patient only's loss
(positive: the configuration is better). Mean over 3 paired draws; top-K test unless noted.

| Configuration | 64x2x4, 10K | 64x2x4, 153K | 256x4x8, 10K | 256x4x8, 153K |
|---|---|---|---|---|
| v1 #2, tested at top 1 (v1's test) | −6.2% | −18.7% | −8.8% | −22.6% |
| v1 #2, tested at top 4 (as trained) | −3.4% | −12.1% | −6.5% | −18.1% |
| v1 #2 + masked mean | −4.5% | −13.4% | −5.6% | −18.1% |
| v1 #4 random documents | −4.6% | −13.2% | −6.8% | −15.8% |
| **gated retrieval (fixed #2)** | **−0.45%** | **−0.16%** | **+1.3%** | **−0.3%** |
| gated, random documents | −0.67% | −0.2% | +1.3% | −0.7% |
| retrieval only | −8.0% | −16.7% | −9.7% | −17.2% |

- **The gated fusion closes the gap**: from −12% to −23% (v1) to within ±1.3% of patient only, with
  intervals that include 0 in three of four cells (64x2x4 at 153K: −0.16% ± 0.14%).
- **But it doesn't use the documents**: the gate ends at `tanh(g)` ≈ 0.001–0.02, and top-K, top-1
  and random-document tests give the same loss. The model has learned that K = 4 token-embedding
  documents don't help, and switched them off.
- Testing with the same K as training recovers about a third of v1's gap; the masked mean alone
  changes little. Most of v1's loss came from the fusion design (no way to ignore documents).
- Retrieval only is 8–17% worse than patient only, and much worse again with random documents,
  so its retrieved documents do carry patient information, but mostly as a lossy code for the
  patient: 50–78% of its test patients share the same top document (45–64% for gated retrieval).

## Stage 0: patient-only settings (D 153K)

| `W` | Best lr | Test loss | AUROC |
|---|---|---|---|
| 64x2x4 | 3e-3 | 0.02127 | 0.906 |
| 128x2x4 | 1e-3 | 0.02122 | 0.906 |
| 256x4x8 | 3e-4 | **0.02077** | 0.911 |

The best learning rate falls with size; dropout 0 vs 0.1 changes the loss by < 1%; the best
checkpoint (by validation loss) is the last one in every run, so training is not yet overfitting.
Stage 1 used lr 1e-3 for every size, which is not the best for 64x2x4 or 256x4x8.
