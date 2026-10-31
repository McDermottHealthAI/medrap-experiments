# Scaling laws

Code: [`scaling_laws/`](../../scaling_laws/) ([`run.sh`](../../scaling_laws/run.sh),
[`submit.sh`](../../scaling_laws/submit.sh), [`analyze.py`](../../scaling_laws/analyze.py)).

## 1. Design

### Variables

| Symbol | Meaning | Value |
|---|---|---|
| `B` | patients per batch | 32 |
| `S` | codes per patient | 256 |
| `W` | patient-encoder width | 4 to 256 (varies) |
| `D` | training patients | 1K to 153K (varies) |
| `T` | tasks | 5 to 2,500 (varies) |
| `K` | documents per patient, plus the null document | 4 + 1 in training, 1 + 1 at test |
| `S_doc` | tokens per document | 256 |

### Architecture

![Model architecture](../../scaling_laws/architecture.png)

| Component | What it is | Output |
|---|---|---|
| Patient Encoder | code embedding table + transformer (L layers, H heads, time-gap position encoding) | B × S × W |
| Mean Over Codes | average over the patient's codes | B × W |
| Head | one linear layer: one logit per task | B × T, or B × K × T per document |
| Query | mean over codes, then linear W → 1024 (#2, #4), or a frozen Qwen3 text embedding (#3) | B × 1024 |
| External Document Database | textbook passages + one null document, each with a fixed 1024-d key; top K by query · key | B × K × S_doc |
| Retrieval Encoder | frozen Qwen3 token embeddings (151,669 × 1,024) | B × K × S_doc × 1024 |
| Fusion | patient codes cross-attend to one document at a time (keys and values 1024 → 256), then mean over codes | B × K × 256 |
| Combine Documents | weighted average of the per-document predictions; weights from query · key | B × T |

### Configurations

| # | Configuration | Query | Documents |
|---|---|---|---|
| 1 | Patient only | none | none |
| 2 | Retrieval, learned query | learned from the patient encoder | top K + null |
| 3 | Retrieval, Qwen3 query | patient's last 32 codes as text, frozen Qwen3 embedding | top K + null |
| 4 | Retrieval, random documents (control) | as #2 | random passages + null, in training and at test |

#2 − #4 measures the documents' knowledge; #4 − #1 the extra architecture.

### Loss

Patient $i$, task $t$, label $y_{it}$, $\sigma$ = sigmoid, $V$ = labelled (patient, task) pairs.

#1 (`MultiTaskBCELoss`):

$$
\mathcal{L} = \frac{1}{|V|} \sum_{(i,t) \in V} -\Big[\, y_{it} \log \sigma(z_{it}) + (1 - y_{it}) \log\big(1 - \sigma(z_{it})\big) \Big]
$$

#2, #3, #4 (`MultiTaskBCEMarginalizedLoss`):

$$
w_{ik} = \operatorname{softmax}_k\!\left( \frac{q_i \cdot \kappa_{ik}}{\sqrt{1024}} \right),
\qquad
p_{it} = \sum_{k=1}^{K} w_{ik}\, \sigma(z_{ikt}),
\qquad
\mathcal{L} = \frac{1}{|V|} \sum_{(i,t) \in V} -\Big[\, y_{it} \log p_{it} + (1 - y_{it}) \log(1 - p_{it}) \Big]
$$

### Training

| Setting | Value |
|---|---|
| Optimizer | AdamW, lr 1e-3, weight decay 0.01 |
| Schedule | linear warmup, cosine decay to 0 |
| Batch size | 32 |
| Gradient clipping | 1.0 |
| Dropout | 0 in the patient encoder, 0.1 in the fusion |
| Head bias at start | each task's training base rate (log-odds) |
| Validation | every 0.2 epoch, up to 200 batches |
| Checkpoint tested | last |
| Seed | draw number (1–5) |

| `D` | Epochs | Warmup steps |
|---|---|---|
| 1K | 10 | 60 |
| 3K | 7 | 100 |
| 10K | 5 | 200 |
| 30K | 4 | 200 |
| 100K | 3 | 200 |
| 153K | 3 | 200 |

### Testing

| Evaluation | Documents | Configs |
|---|---|---|
| Test (reported) | top document + null | #1 (no documents), #2, #3; #4 with a random document |
| Random document | one random passage + null | #2, #3 |

Metric: test loss, mean over (patient, task) pairs. AUROC also reported.

### Plots

| # | Plot |
|---|---|
| P1 | Loss vs `D`, one line per `W`, one panel per configuration |
| P2 | Loss vs `N`, one line per `D`, one panel per configuration |
| P3 | Fitted law (line) with its 95% interval (band) vs `D` and `N`; cell means as dots |
| P4 | Paired law difference, retrieval − Patient only, with 95% interval; matched cell differences as dots. Control: #2 − #4 and #4 − #1 |
| P5 | Compute-optimal encoder size `N*` vs training compute, with 95% interval |
| P6 | Test loss at `N*` vs training compute, with 95% interval; cells as dots |
| P7 | Random-document − top-document test loss vs `D` and `N`, for #2 and #3 |

### Fitting

Fits, intervals, comparisons and allocation use
[simple-scaling-laws](https://github.com/mmcdermott/simple-scaling-laws) 0.2.0.

| Step | Method |
|---|---|
| Law | `separable-power`, per configuration and `T`, on test loss and on AUROC (logit scale) |
| Input | one row per run; `test_set_id` = (T, draw), shared by all configurations |
| Intervals | wild cluster bootstrap over (`N`, `D`) cells, 1,000 draws, seed 0 |
| Comparisons | paired draws: arms fitted on the same cells with the same seed (`compare`) |
| Allocation | `optimal_allocation` with a `power` cost model fitted to each configuration's `C` |

$$
L(N, D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}}
$$

$$
C = 6 \cdot N_{\text{compute}} \cdot D \cdot \text{epochs} \cdot S
$$

- `N`: patient-encoder parameters without the code embedding table.
- $N_{\text{compute}}$: trainable parameters without embedding tables (encoder, query projector,
  fusion, head).
- `C` is computed, not measured.
- Each run has one test evaluation, so test-set noise is part of the run-to-run noise
  (`no_repeated_evaluations`).
- Law-free isoFLOP profiles need runs along equal-compute curves; this grid has none.
- R²: $1 - SS_{\text{res}} / SS_{\text{tot}}$, over runs and over cell means.

## 2. Runs

| Config | `W` | `D` | `T` | Draws | Runs |
|---|---|---|---|---|---|
| #1 Patient only | 4, 8, 16, 32, 64, 128, 128x2x4, 256x4x8 | 1K, 3K, 10K, 30K, 100K, 153K | 5, 25, 125, 625, 2,500 | 5 | 1,200 |
| #2 Retrieval, learned query | 4, 8, 16, 32, 64, 128, 128x2x4, 256x4x8 | 1K, 3K, 10K, 30K, 100K, 153K | 5, 25, 125, 625, 2,500 | 5 | 1,200 |
| #3 Retrieval, Qwen3 query | 4, 16, 64, 256x4x8 | 1K, 3K, 10K, 30K, 100K, 153K | 25, 625 | 5 | 240 |
| #4 Random documents | 4, 16, 64 | 1K, 10K, 100K | 625 | 5 | 45 |
| **Total** | | | | | **2,685** |

`WxLxH` = width, layers, heads; otherwise 1 layer, 1 head. All runs finished.

## 3. Results

### Best cell per task count

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

### Fitted law: test loss

| Config | T | E [95% CI] | α [95% CI] | β [95% CI] | A | B | R² (runs) | R² (cell means) | RMSE |
|---|---|---|---|---|---|---|---|---|---|
| #1 | 5 | 0.0283 [-7.8597, 0.0323] | 1.88 [0.00, 3.00] | 0.45 [0.09, 0.87] | 60.9 | 0.581 | 0.10 | 0.74 | 0.0246 |
| #1 | 25 | 0.0215 [-3.3332, 0.0259] | 0.58 [0.02, 3.00] | 0.27 [0.00, 0.54] | 0.107 | 0.106 | 0.10 | 0.91 | 0.0141 |
| #1 | 125 | 0.0146 [0.0064, 0.0163] | 0.36 [0.17, 0.68] | 0.25 [0.09, 0.40] | 0.0207 | 0.0458 | 0.26 | 0.95 | 0.0039 |
| #1 | 625 | 0.0141 [-0.0439, 0.0182] | 0.23 [0.13, 0.37] | 0.13 [0.02, 0.25] | 0.0161 | 0.0327 | 0.58 | 0.96 | 0.0023 |
| #1 | 2500 | 0.0137 [-0.5951, 0.0182] | 0.33 [0.19, 0.52] | 0.11 [0.00, 0.23] | 0.0203 | 0.0285 | 0.90 | 0.95 | 0.0008 |
| #2 | 5 | 0.0382 [-4.8439, 0.0398] | 0.65 [0.00, 3.00] | 0.89 [0.38, 2.23] | 0.0558 | 5.87 | 0.03 | 0.88 | 0.0254 |
| #2 | 25 | 0.0307 [-2.7458, 0.0328] | 0.27 [0.00, 3.00] | 0.53 [0.00, 1.44] | 0.0139 | 0.274 | 0.03 | 0.91 | 0.0156 |
| #2 | 125 | 0.0200 [-0.7759, 0.0205] | 0.58 [0.00, 3.00] | 0.48 [0.19, 0.79] | 0.0244 | 0.0947 | 0.08 | 0.90 | 0.0041 |
| #2 | 625 | 0.0249 [0.0198, 0.0257] | 0.61 [0.06, 1.99] | 0.33 [0.12, 0.52] | 0.0277 | 0.0405 | 0.21 | 0.88 | 0.0025 |
| #2 | 2500 | 0.0236 [0.0206, 0.0245] | 0.78 [0.31, 2.01] | 0.28 [0.13, 0.45] | 0.0706 | 0.0296 | 0.69 | 0.89 | 0.0008 |
| #3 | 25 | 0.0289 [-3.4335, 0.0325] | 0.24 [0.00, 3.00] | 0.66 [0.16, 1.75] | 0.0162 | 1.4 | 0.10 | 0.98 | 0.0159 |
| #3 | 625 | 0.0248 [-0.6236, 0.0257] | 0.40 [0.00, 3.00] | 0.45 [0.20, 0.74] | 0.0123 | 0.126 | 0.36 | 0.95 | 0.0025 |
| #4 | 625 | 0.0226 [-0.5621, 0.0267] | 0.24 [0.00, 3.00] | 0.18 [0.00, 3.00] | 0.0055 | 0.0208 | 0.26 | 0.99 | 0.0024 |

### Fitted law: AUROC (fitted on the logit scale)

| Config | T | α [95% CI] | β [95% CI] | R² (runs) | R² (cell means) |
|---|---|---|---|---|---|
| #1 | 5 | 0.49 [0.18, 0.89] | 0.00 [0.00, 0.20] | 0.34 | 0.83 |
| #1 | 25 | 0.38 [0.21, 0.56] | 0.04 [0.00, 0.19] | 0.81 | 0.92 |
| #1 | 125 | 0.33 [0.21, 0.48] | 0.00 [0.00, 0.14] | 0.92 | 0.94 |
| #1 | 625 | 0.27 [0.18, 0.39] | 0.00 [0.00, 0.11] | 0.96 | 0.95 |
| #1 | 2500 | 0.36 [0.23, 0.54] | 0.00 [0.00, 0.12] | 0.94 | 0.95 |
| #2 | 5 | 0.43 [0.00, 3.00] | 0.04 [0.00, 0.24] | 0.19 | 0.74 |
| #2 | 25 | 0.23 [0.04, 0.44] | 0.07 [0.00, 0.19] | 0.51 | 0.94 |
| #2 | 125 | 0.55 [0.19, 1.12] | 0.00 [0.00, 0.12] | 0.77 | 0.93 |
| #2 | 625 | 0.69 [0.00, 2.28] | 0.00 [0.00, 0.12] | 0.85 | 0.93 |
| #2 | 2500 | 0.87 [0.00, 3.00] | 0.00 [0.00, 0.11] | 0.86 | 0.92 |
| #3 | 25 | 0.41 [0.08, 3.00] | 0.00 [0.00, 0.18] | 0.61 | 0.94 |
| #3 | 625 | 3.00 [0.16, 3.00] | 0.00 [0.00, 0.12] | 0.91 | 0.97 |
| #4 | 625 | 0.45 [0.00, 3.00] | 0.00 [0.00, 0.72] | 0.90 | 0.92 |

### Paired comparisons (test loss)

| Comparison | T | Cells | Better (95% CI < 0) | Worse (95% CI > 0) | Difference at largest cell [95% CI] | P(better) at largest cell |
|---|---|---|---|---|---|---|
| #2 − #1 | 5 | 48 | 0 | 25 | +0.0077 [+0.0046, +0.0116] (W 256x4x8, D 153,220) | 0.00 |
| #2 − #1 | 25 | 48 | 0 | 28 | +0.0056 [+0.0037, +0.0077] (W 256x4x8, D 153,220) | 0.00 |
| #2 − #1 | 125 | 48 | 1 | 34 | +0.0032 [+0.0026, +0.0038] (W 256x4x8, D 153,220) | 0.00 |
| #2 − #1 | 625 | 48 | 3 | 37 | +0.0045 [+0.0038, +0.0051] (W 256x4x8, D 153,220) | 0.00 |
| #2 − #1 | 2500 | 48 | 3 | 38 | +0.0035 [+0.0030, +0.0039] (W 256x4x8, D 153,220) | 0.00 |
| #3 − #1 | 25 | 24 | 0 | 10 | +0.0039 [-0.0004, +0.0084] (W 256x4x8, D 153,220) | 0.04 |
| #3 − #1 | 625 | 24 | 0 | 19 | +0.0039 [+0.0030, +0.0048] (W 256x4x8, D 153,220) | 0.00 |
| #3 − #2 | 25 | 24 | 0 | 2 | -0.0017 [-0.0058, +0.0022] (W 256x4x8, D 153,220) | 0.77 |
| #3 − #2 | 625 | 24 | 0 | 5 | -0.0004 [-0.0009, +0.0001] (W 256x4x8, D 153,220) | 0.91 |
| #2 − #4 | 625 | 9 | 0 | 0 | -0.0001 [-0.0024, +0.0023] (W 64, D 100,000) | 0.54 |
| #4 − #1 | 625 | 9 | 0 | 4 | +0.0029 [+0.0010, +0.0051] (W 64, D 100,000) | 0.00 |

### Fit warnings

| Warning | Count |
|---|---|
| `weakly_identified_exponent` | 27 |
| `bootstrap_nonconvergence` | 17 |
| `no_repeated_evaluations` | 13 |
| `ill_conditioned_fit` | 11 |
| `parameter_at_bound` | 8 |
| `lack_of_fit` | 2 |
| `optimizer_nonconvergence` | 1 |

## 4. Plots

### P1

#### T = 5

![P1 T5](T5/P1.png)

#### T = 25

![P1 T25](T25/P1.png)

#### T = 125

![P1 T125](T125/P1.png)

#### T = 625

![P1 T625](T625/P1.png)

#### T = 2500

![P1 T2500](T2500/P1.png)

### P2

#### T = 5

![P2 T5](T5/P2.png)

#### T = 25

![P2 T25](T25/P2.png)

#### T = 125

![P2 T125](T125/P2.png)

#### T = 625

![P2 T625](T625/P2.png)

#### T = 2500

![P2 T2500](T2500/P2.png)

### P3

#### T = 5

![P3_D T5](T5/P3_D.png)
![P3_N T5](T5/P3_N.png)

#### T = 25

![P3_D T25](T25/P3_D.png)
![P3_N T25](T25/P3_N.png)

#### T = 125

![P3_D T125](T125/P3_D.png)
![P3_N T125](T125/P3_N.png)

#### T = 625

![P3_D T625](T625/P3_D.png)
![P3_N T625](T625/P3_N.png)

#### T = 2500

![P3_D T2500](T2500/P3_D.png)
![P3_N T2500](T2500/P3_N.png)

### P4

#### T = 5

![P4 T5](T5/P4.png)

#### T = 25

![P4 T25](T25/P4.png)

#### T = 125

![P4 T125](T125/P4.png)

#### T = 625

![P4 T625](T625/P4.png)
![P4 control T625](T625/P4_control.png)

#### T = 2500

![P4 T2500](T2500/P4.png)

### P5

#### T = 5

![P5 T5](T5/P5.png)

#### T = 25

![P5 T25](T25/P5.png)

#### T = 125

![P5 T125](T125/P5.png)

#### T = 625

![P5 T625](T625/P5.png)

#### T = 2500

![P5 T2500](T2500/P5.png)

### P6

#### T = 5

![P6 T5](T5/P6.png)

#### T = 25

![P6 T25](T25/P6.png)

#### T = 125

![P6 T125](T125/P6.png)

#### T = 625

![P6 T625](T625/P6.png)

#### T = 2500

![P6 T2500](T2500/P6.png)

### P7

#### T = 5

![P7 T5](T5/P7.png)

#### T = 25

![P7 T25](T25/P7.png)

#### T = 125

![P7 T125](T125/P7.png)

#### T = 625

![P7 T625](T625/P7.png)

#### T = 2500

![P7 T2500](T2500/P7.png)

## Files

| File | Contents |
|---|---|
| `runs.csv` | one row per test evaluation |
| `cells.csv` | means per cell |
| `params.csv` | `N` and `N_compute` per configuration, `W`, `T` |
| `fits.csv` | fitted parameters with 95% intervals, R², RMSE |
| `comparisons.csv` | paired differences at every trained cell |
| `allocation.csv` | `N*`, `D*` and loss per budget |
| `warnings.csv` | every fit warning |
| `fits/*.slaw` | fitted models (`simple_scaling_laws.load`) |

Made by [`scaling_laws/analyze.py`](../../scaling_laws/analyze.py).
