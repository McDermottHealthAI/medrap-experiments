# Scaling laws: results

All runs of [`scaling_laws/design.md`](../../scaling_laws/design.md):

| Config | Grid | Runs |
|---|---|---|
| #1 Patient only | 8 W × 6 D × 5 T × 5 draws | 1,200 |
| #2 Retrieval, learned query | 8 W × 6 D × 5 T × 5 draws | 1,200 |
| #3 Retrieval, Qwen3 query | W 4, 16, 64, 256x4x8 × 6 D × T 25, 625 × 5 draws | 240 |
| #4 Random documents | W 4, 16, 64 × D 1K, 10K, 100K × T 625 × 5 draws | 45 |

Metric: test loss (lower is better), mean over draws; error bars are standard errors over draws.

## Findings

- Patient only beats every retrieval configuration at every task count. #2 has lower loss in
  27 of 240 cells, all at D ≤ 10K; #3 in 14 of 240 runs.
- The gap grows with data and model size: at D = 153K #2 is worse in all 40 cells, by 0.001–0.009.
- Retrieval barely improves with encoder size (P2): its fitted floor E is 0.019–0.029 vs
  0.011–0.020 for Patient only.
- The Qwen3 query (#3) performs like the learned query (#2): mean difference +0.0009 (T 25) and
  +0.0002 (T 625).
- The retrieved documents add no knowledge (P4 control): #2 − #4 is +0.0000 on average and
  negative in 21 of 45 runs. What retrieval changes comes from the architecture (#4 − #1).
- The models do use their documents: a random document at test raises the loss of #3 in 236 of
  240 runs, by 0.002 on average. A model trained on random documents (#4) does as well as #2.

## Best cell per task count

| T | Config | Loss | AUROC | At |
|---|---|---|---|---|
| 5 | #1 | 0.0299 | 0.872 | W 128x2x4, D 153,220 |
| 5 | #2 | 0.0370 | 0.845 | W 128x2x4, D 153,220 |
| 25 | #1 | 0.0252 | 0.897 | W 128x2x4, D 153,220 |
| 25 | #2 | 0.0309 | 0.844 | W 64, D 153,220 |
| 25 | #3 | 0.0298 | 0.845 | W 256x4x8, D 153,220 |
| 125 | #1 | 0.0168 | 0.894 | W 128x2x4, D 153,220 |
| 125 | #2 | 0.0198 | 0.831 | W 64, D 153,220 |
| 625 | #1 | 0.0209 | 0.909 | W 128x2x4, D 153,220 |
| 625 | #2 | 0.0251 | 0.846 | W 32, D 153,220 |
| 625 | #3 | 0.0255 | 0.844 | W 16, D 153,220 |
| 625 | #4 | 0.0256 | 0.827 | W 64, D 100,000 |
| 2500 | #1 | 0.0206 | 0.907 | W 128x2x4, D 153,220 |
| 2500 | #2 | 0.0241 | 0.840 | W 16, D 153,220 |

## Fitted law

$L(N, D) = E + A / N^{\alpha} + B / D^{\beta}$, fitted on all runs of each configuration and T.
The #4 fit uses a 3 × 3 grid.

| Config | T | E | A | α | B | β |
|---|---|---|---|---|---|---|
| #1 | 5 | 0.0167 | 1.71 | 1.29 | 0.145 | 0.30 |
| #1 | 25 | 0.0195 | 0.0836 | 0.55 | 0.0984 | 0.26 |
| #1 | 125 | 0.0136 | 0.0178 | 0.34 | 0.0381 | 0.22 |
| #1 | 625 | 0.0113 | 0.0152 | 0.22 | 0.0317 | 0.10 |
| #1 | 2500 | 0.0105 | 0.0184 | 0.31 | 0.0289 | 0.09 |
| #2 | 5 | 0.0243 | 0.591 | 1.26 | 0.359 | 0.52 |
| #2 | 25 | 0.0285 | 0.0149 | 0.31 | 0.352 | 0.57 |
| #2 | 125 | 0.0195 | 0.0305 | 0.63 | 0.0882 | 0.47 |
| #2 | 625 | 0.0248 | 0.035 | 0.66 | 0.0424 | 0.34 |
| #2 | 2500 | 0.0235 | 0.088 | 0.82 | 0.0288 | 0.28 |
| #3 | 25 | 0.0270 | 0.0202 | 0.32 | 1.08 | 0.63 |
| #3 | 625 | 0.0248 | 0.015 | 0.46 | 0.123 | 0.45 |
| #4 | 625 | 0.0226 | 0.00622 | 0.28 | 0.0207 | 0.18 |

## Plots

### P1

Loss vs training patients D, one line per encoder size.

**T = 5**

![P1 T5](T5/P1.png)

**T = 25**

![P1 T25](T25/P1.png)

**T = 125**

![P1 T125](T125/P1.png)

**T = 625**

![P1 T625](T625/P1.png)

**T = 2500**

![P1 T2500](T2500/P1.png)

### P2

Loss vs encoder parameters N, one line per D.

**T = 5**

![P2 T5](T5/P2.png)

**T = 25**

![P2 T25](T25/P2.png)

**T = 125**

![P2 T125](T125/P2.png)

**T = 625**

![P2 T625](T625/P2.png)

**T = 2500**

![P2 T2500](T2500/P2.png)

### P3

Fitted law (dashed) through P1 and P2.

**T = 5**

![P3_D T5](T5/P3_D.png)
![P3_N T5](T5/P3_N.png)

**T = 25**

![P3_D T25](T25/P3_D.png)
![P3_N T25](T25/P3_N.png)

**T = 125**

![P3_D T125](T125/P3_D.png)
![P3_N T125](T125/P3_N.png)

**T = 625**

![P3_D T625](T625/P3_D.png)
![P3_N T625](T625/P3_N.png)

**T = 2500**

![P3_D T2500](T2500/P3_D.png)
![P3_N T2500](T2500/P3_N.png)

### P4

Retrieval − Patient only loss, matched on (W, D, draw). Below 0: retrieval better.

**T = 5**

![P4 T5](T5/P4.png)

**T = 25**

![P4 T25](T25/P4.png)

**T = 125**

![P4 T125](T125/P4.png)

**T = 625**

![P4 T625](T625/P4.png)

Control slice: #2 − #4 (knowledge) and #4 − #1 (architecture).

![P4 control T625](T625/P4_control.png)

**T = 2500**

![P4 T2500](T2500/P4.png)

### P5

IsoFLOP curves from the fitted law; ★ is the best encoder size per budget.

**T = 5**

![P5 T5](T5/P5.png)

**T = 25**

![P5 T25](T25/P5.png)

**T = 125**

![P5 T125](T125/P5.png)

**T = 625**

![P5 T625](T625/P5.png)

**T = 2500**

![P5 T2500](T2500/P5.png)

### P6

Best loss vs training compute.

**T = 5**

![P6 T5](T5/P6.png)

**T = 25**

![P6 T25](T25/P6.png)

**T = 125**

![P6 T125](T125/P6.png)

**T = 625**

![P6 T625](T625/P6.png)

**T = 2500**

![P6 T2500](T2500/P6.png)

## Definitions

- N: patient-encoder parameters without the code embedding table.
- C = 6 · N_compute · D · epochs · 256, with N_compute = trainable parameters without embedding
  tables (encoder, query projector, fusion, head).
- P5 and P6 curves are read off the fitted law, using only D within the trained range.

## Files

`runs.csv` (one row per test evaluation), `cells.csv` (means per cell), `fits.csv`, `params.csv`.
Made by [`scaling_laws/analyze.py`](../../scaling_laws/analyze.py).
