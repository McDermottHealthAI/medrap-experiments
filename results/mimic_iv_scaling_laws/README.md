# Retrieval vs `patient_only` scaling laws on MIMIC-IV

**Status:**
- **Round 1 complete:** 500 runs, covering the 25-task scaling grid and the task-count ablation.
- **Round 2 running:** the 625-task scaling grid (250 more runs) and the knowledge-vs-capacity control (90 runs). Its results will be added here when it finishes.

**Code:** [`mimic_iv_scaling_laws/`](../../mimic_iv_scaling_laws/README.md). For the reasoning behind each choice, see its [`DESIGN.md`](../../mimic_iv_scaling_laws/DESIGN.md).

## TL;DR

- **Retrieval helps a lot when the patient encoder is small or training data is scarce.** At encoder
  width 4–8 with 1K–10K training patients, retrieval adds **+0.13 to +0.23 test AUROC** in every
  draw.
- **The advantage closes as the model and the data grow.** At width 64–128, no cell shows a
  significant difference in either direction. At 100K patients, both arms reach about
  0.83–0.86 AUROC at every width.
- **With many tasks, a small, consistent gain remains at every data size.** At width 32 with
  125–2,500 tasks, retrieval wins **5/5 draws in every cell**, by +0.02–0.04 even at 100K
  patients. At 25 tasks the same cells look borderline only because the choice of tasks adds
  noise.
- **Test loss shows the same picture, with a bigger small-model gap.** Both arms' test BCE is
  the same quantity. Retrieval's loss is lower in 20/30 cells, by up to 0.27 at width 4 with 1K
  patients, and the gap closes at large width or data.
- **A power-law scaling fit is not identifiable on this data.** AUROC still rises roughly
  linearly in log(patients) up to 100K with no plateau, so the curves below are descriptive.
  Their exponents shouldn't be extrapolated.
- **Open question: knowledge or capacity?** The retrieval arm's 256-d fusion block and head
  dwarf the smallest encoders, so part of the small-model gain could be extra parameters
  rather than the retrieved text. Round 2's control arms test this directly.

## Setup

| | |
|---|---|
| Cohort | MIMIC-IV MEDS, **rebuilt from the raw data** for this experiment (MEDS-transforms: `min_subjects_per_code=100`, `min_events_per_subject=10`; then meds-torch-data tensorization) |
| Tasks | Binary task: does code X occur within **30 days** of the prediction time? The prediction time is one `uniform_event` point per patient. Codes are drawn **uniformly at random** from the train split, and codes with only one class in any split are redrawn |
| Draws | 5 per cell. Each draw sets its own task codes (seed 101–505), its own subsample of training patients and its own training seed. Both arms use the same draws, so comparisons are **paired** |
| Metric | Test (`held_out`) AUROC, averaged over tasks, from `last.ckpt` |

### Arms

| | `patient_only` | `retrieval` |
|---|---|---|
| Encoder | RoPE transformer, 1 layer, 1 head, `ff_dim = 2·dim` | same |
| Between encoder and head | passthrough | learned query (mean-pool → linear → 1024-d) → FAISS top-4 from MedRAG/textbooks (Qwen3-Embedding-0.6B) → per-document cross-attention fusion (d=256) |
| Head / loss | linear, in_dim = dim / multitask BCE | linear, in_dim = 256 / BCE on probabilities marginalized over documents |

### Shared training

- **Optimization:** batch 32, lr 1e-3, cosine schedule with warmup, gradient clipping 1.0.
- **Context:** the last 256 events.
- **Schedule** (epochs / warmup steps, scaled with training size so small cells get enough steps):

| Training patients | 1K | 3K | 10K | 30K | 100K |
|---|---|---|---|---|---|
| Epochs / warmup steps | 10 / 60 | 7 / 100 | 5 / 200 | 4 / 200 | 3 / 200 |

### Experiments

| Experiment | What varies | Fixed | Runs | Status |
|---|---|---|---|---|
| **Main grid, 25 tasks** | width {4, 8, 16, 32, 64, 128} × training patients {1K, 3K, 10K, 30K, 100K} × 2 arms | 25 tasks | 300 | ✅ |
| **Task-count ablation** | tasks {5, 25, 125, 625, 2,500} × training patients × 2 arms | width 32 | 250 | ✅ |
| **Main grid, 625 tasks** | same grid as the main grid | 625 tasks (less noise) | 300 | ⏳ round 2 |
| **Knowledge-vs-capacity control** | `retrieval_random_docs` and `retrieval_null_docs` (the retrieval model trained *and* tested on documents with no knowledge) × width {4, 16, 64} × patients {1K, 10K, 100K} | 625 tasks | 90 | ⏳ round 2 |

## Results: main grid, 25 tasks

### Retrieval advantage

Each cell is retrieval − `patient_only` test AUROC, paired over the 5 draws: the mean, then
<sub>[95% CI] · draws retrieval won</sub>. **Bold** marks cells where the CI excludes 0.

| Width ↓ / patients → | 1K | 3K | 10K | 30K | 100K |
|---|---|---|---|---|---|
| dim 4 | **+0.178** <sub>[+0.135, +0.221] · 5/5</sub> | **+0.213** <sub>[+0.152, +0.273] · 5/5</sub> | **+0.163** <sub>[+0.133, +0.193] · 5/5</sub> | +0.056 <sub>[-0.011, +0.123] · 5/5</sub> | **+0.031** <sub>[+0.023, +0.039] · 5/5</sub> |
| dim 8 | **+0.151** <sub>[+0.046, +0.256] · 5/5</sub> | **+0.231** <sub>[+0.186, +0.276] · 5/5</sub> | **+0.136** <sub>[+0.002, +0.270] · 5/5</sub> | **+0.033** <sub>[+0.011, +0.055] · 5/5</sub> | **+0.036** <sub>[+0.006, +0.065] · 5/5</sub> |
| dim 16 | +0.085 <sub>[-0.031, +0.202] · 4/5</sub> | **+0.158** <sub>[+0.070, +0.246] · 5/5</sub> | **+0.019** <sub>[+0.004, +0.033] · 5/5</sub> | **+0.030** <sub>[+0.017, +0.043] · 5/5</sub> | **+0.033** <sub>[+0.023, +0.044] · 5/5</sub> |
| dim 32 | +0.053 <sub>[-0.038, +0.143] · 4/5</sub> | **+0.040** <sub>[+0.020, +0.059] · 5/5</sub> | +0.013 <sub>[-0.002, +0.027] · 4/5</sub> | **+0.028** <sub>[+0.010, +0.045] · 5/5</sub> | +0.008 <sub>[-0.004, +0.020] · 5/5</sub> |
| dim 64 | -0.032 <sub>[-0.091, +0.027] · 2/5</sub> | -0.001 <sub>[-0.028, +0.026] · 2/5</sub> | +0.017 <sub>[-0.008, +0.042] · 4/5</sub> | +0.014 <sub>[-0.001, +0.029] · 4/5</sub> | +0.003 <sub>[-0.020, +0.026] · 2/5</sub> |
| dim 128 | -0.041 <sub>[-0.094, +0.012] · 1/5</sub> | +0.005 <sub>[-0.010, +0.021] · 3/5</sub> | -0.004 <sub>[-0.021, +0.013] · 1/5</sub> | +0.005 <sub>[-0.005, +0.015] · 4/5</sub> | +0.001 <sub>[-0.011, +0.013] · 3/5</sub> |

**15 of 30 cells are significantly in retrieval's favor, and none are significantly against it.**
The gain sits in the bottom-left of the grid (small width, little data) and fades toward
zero at the top and right.

![Retrieval advantage heatmap](plots/fig3_retrieval_advantage.png)

### Raw AUROC

Mean test AUROC over 5 draws:

| Width | Arm | 1K | 3K | 10K | 30K | 100K |
|---|---|---|---|---|---|---|
| dim 4 | patient_only | 0.521 | 0.535 | 0.614 | 0.740 | 0.794 |
|  | retrieval | 0.699 | 0.747 | 0.777 | 0.796 | 0.825 |
| dim 8 | patient_only | 0.518 | 0.534 | 0.645 | 0.768 | 0.796 |
|  | retrieval | 0.669 | 0.765 | 0.781 | 0.801 | 0.831 |
| dim 16 | patient_only | 0.561 | 0.595 | 0.765 | 0.783 | 0.809 |
|  | retrieval | 0.646 | 0.753 | 0.784 | 0.813 | 0.843 |
| dim 32 | patient_only | 0.628 | 0.726 | 0.779 | 0.788 | 0.835 |
|  | retrieval | 0.681 | 0.765 | 0.791 | 0.816 | 0.843 |
| dim 64 | patient_only | 0.717 | 0.770 | 0.774 | 0.808 | 0.847 |
|  | retrieval | 0.685 | 0.769 | 0.791 | 0.822 | 0.851 |
| dim 128 | patient_only | 0.760 | 0.777 | 0.798 | 0.824 | 0.859 |
|  | retrieval | 0.719 | 0.783 | 0.793 | 0.829 | 0.861 |

Two things stand out:

- **`patient_only` improves cleanly with both width and data**, from 0.52 to 0.86.
- **`retrieval` barely changes with width at a fixed data size**, for example 0.65–0.72 at
  1K patients. The retrieval path is the same size in every cell, so it carries the small
  models, and the encoder's width matters little.

![AUROC vs training patients, by width](plots/fig1_data_scaling.png)

![AUROC vs encoder parameters, by training size](plots/fig2_capacity_scaling.png)

### Fitted scaling law

We fit `AUROC ≈ E + A·N^-α + B·D^-β` per arm on the logit scale with
[`simple-scaling-laws`](https://github.com/mmcdermott/simple-scaling-laws). `N` is the
patient-encoder parameter count and `D` the number of training patients.

| Arm | α (width) | β (data) | R² |
|---|---|---|---|
| `patient_only` | 0.0067 (95% CI 0.002–0.52) | 0.004 | 0.80 |
| `retrieval` | 0.0005 (95% CI 0.0003–1.38) | 0.239 | 0.69 |

⚠️ **Treat these as descriptive only.** Both fits put the asymptote `E` on its constraint
bound, and the width exponent's CI spans 2–3 orders of magnitude. The library flags
`weakly_identified_exponent` for both arms. Within 1K–100K patients, AUROC keeps rising about
linearly in log(D), so there is no plateau for a power law to anchor to. The fitted curves
interpolate the grid well (the lines in the figures above), but the exponents can't support
extrapolation.

## Results: test loss (25 tasks)

Both arms log the **same** test loss: binary cross-entropy of their final predicted
probability, averaged over (patient, task) pairs. For `retrieval`, that probability is the
marginal over its 4 documents, Σₖ P(docₖ | x)·σ(logitₖ). Lower is better, so here a
**negative** value means retrieval is better. The format matches the AUROC table:
<sub>[95% CI] · draws retrieval won</sub>, and **bold** = significant.

| Width ↓ / patients → | 1K | 3K | 10K | 30K | 100K |
|---|---|---|---|---|---|
| dim 4 | **-0.2694** <sub>[-0.3452, -0.1935] · 5/5</sub> | **-0.0743** <sub>[-0.0990, -0.0496] · 5/5</sub> | **-0.0177** <sub>[-0.0278, -0.0077] · 5/5</sub> | **-0.0055** <sub>[-0.0089, -0.0022] · 5/5</sub> | **-0.0016** <sub>[-0.0028, -0.0003] · 5/5</sub> |
| dim 8 | **-0.1000** <sub>[-0.1124, -0.0877] · 5/5</sub> | **-0.0278** <sub>[-0.0349, -0.0207] · 5/5</sub> | **-0.0103** <sub>[-0.0134, -0.0073] · 5/5</sub> | **-0.0036** <sub>[-0.0056, -0.0017] · 5/5</sub> | **-0.0011** <sub>[-0.0022, -0.0000] · 5/5</sub> |
| dim 16 | **-0.0350** <sub>[-0.0421, -0.0279] · 5/5</sub> | **-0.0150** <sub>[-0.0239, -0.0062] · 5/5</sub> | **-0.0055** <sub>[-0.0069, -0.0040] · 5/5</sub> | **-0.0026** <sub>[-0.0040, -0.0012] · 5/5</sub> | -0.0009 <sub>[-0.0020, +0.0001] · 5/5</sub> |
| dim 32 | **-0.0137** <sub>[-0.0199, -0.0075] · 5/5</sub> | **-0.0077** <sub>[-0.0095, -0.0060] · 5/5</sub> | **-0.0027** <sub>[-0.0037, -0.0017] · 5/5</sub> | -0.0017 <sub>[-0.0037, +0.0003] · 4/5</sub> | -0.0003 <sub>[-0.0006, +0.0001] · 4/5</sub> |
| dim 64 | **-0.0059** <sub>[-0.0081, -0.0036] · 5/5</sub> | **-0.0026** <sub>[-0.0037, -0.0016] · 5/5</sub> | -0.0010 <sub>[-0.0020, +0.0001] · 4/5</sub> | -0.0005 <sub>[-0.0010, +0.0001] · 5/5</sub> | +0.0003 <sub>[-0.0002, +0.0007] · 1/5</sub> |
| dim 128 | **-0.0019** <sub>[-0.0037, -0.0001] · 5/5</sub> | -0.0002 <sub>[-0.0009, +0.0005] · 3/5</sub> | +0.0003 <sub>[-0.0009, +0.0014] · 3/5</sub> | +0.0003 <sub>[-0.0001, +0.0006] · 1/5</sub> | +0.0003 <sub>[-0.0006, +0.0013] · 1/5</sub> |

- **Retrieval's loss is significantly lower in 20 of 30 cells and never significantly
  higher.** The pattern matches AUROC: the gap is largest for small encoders on little data,
  and it closes toward dim 64–128 and 100K patients.
- **Retrieval's loss is nearly flat** at ~0.03–0.04 across the whole grid. `patient_only`
  starts far higher (0.31 at dim 4, 1K patients) and falls until the two meet.
- **Loss also rewards calibration.** That's why the small-model gap is much larger in loss
  (−0.27) than in AUROC (+0.18). A small, overfit `patient_only` model is both less
  discriminative and badly overconfident, while averaging over 4 documents keeps
  retrieval's probabilities moderate.
- **The power law fits loss poorly** (R² 0.50 for `patient_only`, 0.06 for `retrieval`).
  The loss figures therefore show the measured means, on a log y-axis, without fitted curves.

![Test-loss difference heatmap](plots/fig3_retrieval_advantage_loss.png)

![Test loss vs training patients, by width](plots/fig1_data_scaling_loss.png)

![Test loss vs encoder parameters, by training size](plots/fig2_capacity_scaling_loss.png)

## Results: task-count ablation (width 32)

Each task count draws its own random codes. Table format as above: retrieval −
`patient_only`, paired, <sub>[95% CI] · draws won</sub>, **bold** = significant.

| Tasks ↓ / patients → | 1K | 3K | 10K | 30K | 100K |
|---|---|---|---|---|---|
| 5 tasks | **-0.047** <sub>[-0.093, -0.002] · 0/5</sub> | -0.013 <sub>[-0.067, +0.042] · 1/5</sub> | -0.007 <sub>[-0.109, +0.095] · 2/5</sub> | +0.032 <sub>[-0.024, +0.089] · 4/5</sub> | +0.003 <sub>[-0.026, +0.033] · 2/5</sub> |
| 25 tasks | +0.053 <sub>[-0.038, +0.143] · 4/5</sub> | **+0.040** <sub>[+0.020, +0.059] · 5/5</sub> | +0.013 <sub>[-0.002, +0.027] · 4/5</sub> | **+0.028** <sub>[+0.010, +0.045] · 5/5</sub> | +0.008 <sub>[-0.004, +0.020] · 5/5</sub> |
| 125 tasks | **+0.088** <sub>[+0.016, +0.161] · 5/5</sub> | **+0.155** <sub>[+0.110, +0.201] · 5/5</sub> | **+0.022** <sub>[+0.011, +0.033] · 5/5</sub> | **+0.027** <sub>[+0.020, +0.034] · 5/5</sub> | **+0.023** <sub>[+0.012, +0.034] · 5/5</sub> |
| 625 tasks | **+0.053** <sub>[+0.014, +0.091] · 5/5</sub> | **+0.173** <sub>[+0.142, +0.204] · 5/5</sub> | **+0.029** <sub>[+0.016, +0.042] · 5/5</sub> | **+0.028** <sub>[+0.017, +0.038] · 5/5</sub> | **+0.039** <sub>[+0.028, +0.050] · 5/5</sub> |
| 2,500 tasks | **+0.032** <sub>[+0.010, +0.055] · 5/5</sub> | **+0.172** <sub>[+0.144, +0.200] · 5/5</sub> | **+0.025** <sub>[+0.020, +0.030] · 5/5</sub> | **+0.023** <sub>[+0.019, +0.028] · 5/5</sub> | **+0.035** <sub>[+0.023, +0.047] · 5/5</sub> |

**More tasks mostly means less noise, not a bigger effect.** `patient_only`'s spread (std)
across draws at width 32 shrinks sharply as the task count grows:

| Tasks | 1K | 3K | 10K | 30K | 100K |
|---|---|---|---|---|---|
| 5 | 0.104 | 0.137 | 0.137 | 0.087 | 0.092 |
| 25 | 0.071 | 0.037 | 0.019 | 0.027 | 0.030 |
| 125 | 0.022 | 0.049 | 0.016 | 0.014 | 0.013 |
| 625 | 0.012 | 0.023 | 0.012 | 0.008 | 0.006 |
| 2,500 | 0.011 | 0.021 | 0.005 | 0.005 | 0.005 |

- **At 125 or more tasks, the gain is stable.** Retrieval shows a steady +0.02–0.04 from
  10K patients up, winning 5/5 draws everywhere. At 25 tasks the same effect is mostly
  hidden by task-sampling noise, which is why round 2 reruns the main grid at 625 tasks.
- **The 3K spike (+0.16 to +0.17) is mostly a `patient_only` artifact.** With 125 or more
  tasks, `patient_only` stalls at about 0.58 with 3K patients, then jumps to 0.75 at 10K.
  This likely comes from the small-data training schedule (7 epochs, warmup 100) rather
  than from retrieval, and is still under investigation.
- **The 5-task row is mostly noise.** The across-draw std is ~0.1, so five random, mostly
  rare tasks are too small a sample to read. The −0.047 at 1K is just inside significance.

![Task-count ablation](plots/fig4a_task_count_ablation.png)

![Retrieval advantage by task count](plots/fig4b_task_count_advantage.png)

## Caveats

- **Capacity confound (being tested in round 2).** The retrieval arm adds a fixed 256-d
  fusion block and head, which is far more capacity than a width-4 encoder. The
  `retrieval_random_docs` / `retrieval_null_docs` arms are trained and tested without
  useful documents, so they separate "the documents help" from "the extra layers help".
  Earlier MIMIC ablations only swapped documents at eval time on checkpoints trained with
  real documents, which can't answer this.
- **Draws mix noise sources.** Each draw changes the task codes, the training-patient
  subsample and the training seed together, so task-choice noise and optimization noise
  aren't separated.
- **Rare tasks.** Uniform random codes are mostly rare lab-value bins, with median positive
  rates around 0.5–1.7%. The smallest cells train on very few positives per task.
- **Model size counts only the encoder, and mostly its embedding table.** The table has
  65,536 × dim rows while the cohort uses ~10K codes, so `N` overstates the effective
  encoder size. The two arms also aren't parameter-matched.
- **The schedule scales with data size.** Larger training sets also get more optimizer
  steps, so the data axis isn't compute-matched.

## Reproduce

```bash
cd mimic_iv_scaling_laws && uv sync
ln -s <path>/mimic_iv_sweep/data/retrieval_db data/retrieval_db
.venv/bin/python scripts/make_null_retrieval_db.py     # null corpus for the control arm
scripts/submit_tasks.sh                                  # rebuild cohort from raw MEDS, then tasks; prints afterok:<ids>
scripts/submit_sweep.sh afterok:<ids>                    # every cell; reruns only fill in missing draws
.venv/bin/python scripts/aggregate_results.py            # fits + parquet summaries
.venv/bin/python scripts/plot_results.py --out .         # figures/ + tables/ (add --out t625 --main-tasks 625 for round 2)
.venv/bin/python scripts/make_report.py                  # report.html: self-contained overview of all of the above
```
