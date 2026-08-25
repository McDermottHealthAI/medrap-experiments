# Data-scarce `patient_only` vs. marginalized retrieval (N=25, 30d)

**Status: COMPLETE -- all 40 jobs (5 draws x 4 train-fractions x 2
architectures) finished, plus the pre-existing 100%/full-data baseline.
A second, independent axis showing the same pattern as
[`results/capacity_starved_retrieval/`](../capacity_starved_retrieval/README.md):
retrieval flips from a small inconsistent loss to a consistent, growing win
once *something* limits the base model.**

Scripts: [`subsample_train_labels.py`](../../mimic_iv_sweep/scripts/subsample_train_labels.py) +
[`sweep_patient_only_data_scarcity_n25_30d_train{50,20,10,5}pct.sh`](../../mimic_iv_sweep/scripts/) +
[`sweep_marginalized_binary_learned_linear_data_scarcity_n25_30d_train{50,20,10,5}pct.sh`](../../mimic_iv_sweep/scripts/)
(all in `mimic_iv_sweep/`, which this experiment reuses for its labels,
retrieval corpus, and cluster setup -- see that directory's `README.md`
for the full sweep infrastructure and every other experiment run against
the same MIMIC-IV cohort).

## Motivation

[`results/capacity_starved_retrieval/`](../capacity_starved_retrieval/README.md)
showed that shrinking the *model* (an ~8x parameter cut to the patient
encoder) flips retrieval from a small inconsistent loss to a small,
consistent win -- evidence that retrieved content carries real information,
but a full-capacity encoder is already expressive enough to not need it.

This experiment tests the same hypothesis along a different, orthogonal
axis: instead of shrinking the *model*, shrink the *training data*. Does
retrieval become load-bearing once the encoder doesn't have enough labeled
examples to learn the task from patient history alone, even though it's
still large enough in principle to represent it?

**The data cut**: only `train.parquet` is subsampled (uniform random by
row, fixed seed per draw, via `subsample_train_labels.py`). `tuning.parquet`
and `held_out.parquet` are copied byte-for-byte unchanged at every fraction
level, so every level is evaluated against the *exact same* validation
data -- only the training signal shrinks. `patient_only` and `marginalized`
train on the identical subsampled label file per (draw, fraction), so the
comparison stays apples-to-apples. Model capacity is **full** throughout
(same `encoder=rope` defaults as the rest of `mimic_iv_sweep`, unlike the
capacity-starved experiment) -- only training-example count varies.

| Fraction of train subjects kept | Approx. train rows (of ~153,195 full) |
| --- | --- |
| 100% (pre-existing baseline, full capacity + full data) | ~153,195 |
| 50% | 76,598 |
| 20% | 30,639 |
| 10% | 15,320 |
| 5% | 7,660 |

Same N=25 task codes, 5 draws, and seeds as every other 30d experiment in
`mimic_iv_sweep`; `query_projector=sequence_mean_1024` ("learned-linear")
only, `retriever.k=4` -- scoped to match the cheaper, faster variant used
throughout this line of experiments, not re-testing `qwen3_text` at every
level to keep the sweep tractable.

```bash
cd mimic_iv_sweep
# one-time: subsample each draw's train split (already done for this run)
python scripts/subsample_train_labels.py \
    data/tasks_zach_uniform_event_n25_30d/draw${DRAW}/tasks \
    data/tasks_zach_uniform_event_n25_30d_train${FRAC}pct/draw${DRAW}/tasks \
    --fraction 0.${FRAC} --seed <seed>

sbatch --array=0-4 scripts/sweep_patient_only_data_scarcity_n25_30d_train50pct.sh
sbatch --array=0-4 scripts/sweep_marginalized_binary_learned_linear_data_scarcity_n25_30d_train50pct.sh
# ... and the same pair for train20pct / train10pct / train5pct
```

## Results

All 5 levels (100/50/20/10/5%) are fully finished, 5 draws each. Every
number below is `val/auroc/mean`, computed once at the end of fit
(`EndOfFitValAUROCCallback`) against the full, unchanged `tuning` split.

### 100% (pre-existing full-capacity, full-data baseline)

| Draw | patient_only | marginalized (learned-linear) | Δ |
| --- | --- | --- | --- |
| 1 | [0.8923](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/paagyvs5) | [0.9020](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/zv6znbgy) | +0.0097 |
| 2 | [0.9095](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/llsm3nrh) | [0.8802](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/z0mkqk7x) | -0.0293 |
| 3 | [0.9030](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/67lvzdwm) | [0.8947](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/yf58w6k6) | -0.0083 |
| 4 | [0.9446](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/s99xd227) | [0.9444](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/s7wcvz6r) | -0.0002 |
| 5 | [0.8691](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/zcmvtx1m) | [0.8628](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/zlfwb75k) | -0.0064 |

Mean patient_only: 0.9037. Mean marginalized: 0.8968. **Δ = -0.0069 ±
0.0144, 1/5 draws won.**

### 50% train

| Draw | patient_only | marginalized (learned-linear) | Δ |
| --- | --- | --- | --- |
| 1 | [0.9040](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/hbsh7b0k) | [0.8881](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/4re52hh9) | -0.0159 |
| 2 | [0.8888](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/1nn1xfse) | [0.8811](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/2ljkgs87) | -0.0077 |
| 3 | [0.8909](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/slvk54iu) | [0.8888](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/06b0lvk3) | -0.0021 |
| 4 | [0.9240](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/iu5qocdh) | [0.9273](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/o0gpturd) | +0.0033 |
| 5 | [0.8648](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/2rny6uhs) | [0.8604](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/di8vsqs6) | -0.0044 |

Mean patient_only: 0.8945. Mean marginalized: 0.8891. **Δ = -0.0053 ±
0.0071, 1/5 draws won.**

### 20% train

| Draw | patient_only | marginalized (learned-linear) | Δ |
| --- | --- | --- | --- |
| 1 | [0.8543](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/x0ha1b3p) | [0.8910](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/api2vv1d) | +0.0367 |
| 2 | [0.8197](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/0l5rrm86) | [0.8428](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/jlh8kx9f) | +0.0231 |
| 3 | [0.8648](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/skwohikj) | [0.8676](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/ox2riuei) | +0.0028 |
| 4 | [0.8911](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/vqafih26) | [0.9022](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/hdwk2bs5) | +0.0112 |
| 5 | [0.8314](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/nw1k4va9) | [0.8316](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/x9dndotg) | +0.0002 |

Mean patient_only: 0.8522. Mean marginalized: 0.8670. **Δ = +0.0148 ±
0.0152, 5/5 draws won.**

### 10% train

| Draw | patient_only | marginalized (learned-linear) | Δ |
| --- | --- | --- | --- |
| 1 | [0.8547](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/dfg7wk4s) | [0.8677](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/lr6n6dv1) | +0.0130 |
| 2 | [0.8285](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/j2olphhq) | [0.8326](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/khgbgs8v) | +0.0041 |
| 3 | [0.8366](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/c2k93nsp) | [0.8478](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/v44e9bg7) | +0.0112 |
| 4 | [0.8364](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/w8eqs736) | [0.8651](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/skol5c7l) | +0.0287 |
| 5 | [0.7945](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/c3grmkuu) | [0.7942](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/txchkhrf) | -0.0003 |

Mean patient_only: 0.8302. Mean marginalized: 0.8415. **Δ = +0.0113 ±
0.0111, 4/5 draws won.**

### 5% train

| Draw | patient_only | marginalized (learned-linear) | Δ |
| --- | --- | --- | --- |
| 1 | [0.8392](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/bmcgrhyv) | [0.8640](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/rmobclhs) | +0.0248 |
| 2 | [0.7843](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/epskg3cb) | [0.8057](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/t119bcag) | +0.0215 |
| 3 | [0.8143](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/ofcl6xan) | [0.8321](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/y2p7thsx) | +0.0178 |
| 4 | [0.8254](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/32mne9wn) | [0.8318](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/1oreijcu) | +0.0064 |
| 5 | [0.7593](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/19oqfw26) | [0.7692](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/8dg8qzzh) | +0.0099 |

Mean patient_only: 0.8045. Mean marginalized: 0.8206. **Δ = +0.0161 ±
0.0078, 5/5 draws won.**

## Summary across all 5 levels

| Train fraction | patient_only mean AUROC | marginalized mean AUROC | Δ (mean ± std) | Draws won |
| --- | --- | --- | --- | --- |
| 100% (full data) | 0.9037 | 0.8968 | -0.0069 ± 0.0144 | 1/5 |
| 50% | 0.8945 | 0.8891 | -0.0053 ± 0.0071 | 1/5 |
| 20% | 0.8522 | 0.8670 | **+0.0148 ± 0.0152** | **5/5** |
| 10% | 0.8302 | 0.8415 | **+0.0113 ± 0.0111** | **4/5** |
| 5% | 0.8045 | 0.8206 | **+0.0161 ± 0.0078** | **5/5** |

The same crossover seen on the capacity axis happens here on the data axis:
retrieval is a small, inconsistent net negative at 100%/50% train data (1/5
draws each), then flips to a consistent, positive win at 20% and below --
and unlike the capacity-starved result, the win rate and magnitude hold
(or even tighten in std) all the way down to 5% train data, with the 5%
level giving the *lowest* variance of any level tested (± 0.0078).

## Caveat: training-positive scarcity at 10%/5%

At the most aggressive subsampling, some (draw, task) combinations have
**zero training-set positives** for some of the 25 task codes -- an
expected, not-a-bug consequence of subsampling rare binary outcomes down
to a few thousand rows:

| Fraction | Draw | Min positives (of 25 tasks) | # tasks with 0 positives |
| --- | --- | --- | --- |
| 10% | 1 | 3 | 0/25 |
| 10% | 2 | 0 | 1/25 |
| 10% | 3 | 6 | 0/25 |
| 10% | 4 | 1 | 0/25 |
| 10% | 5 | 1 | 0/25 |
| 5% | 1 | 0 | 1/25 |
| 5% | 2 | 0 | 1/25 |
| 5% | 3 | 3 | 0/25 |
| 5% | 4 | 1 | 0/25 |
| 5% | 5 | 0 | 1/25 |

Because `patient_only` and `marginalized` always train on the *identical*
subsampled label file for a given (draw, fraction), both architectures face
the same handicap per task -- the Δ comparison stays fair even though
`val/auroc/mean` for individual under-represented tasks in these draws is
noisier (and, for a 0-positive task, that task's contribution to the
aggregate AUROC is degenerate/undefined-per-task but still included in the
same way for both architectures, so it doesn't bias the comparison one way
or the other). This wasn't filtered out or worked around -- it's reported
here transparently as a limitation of pushing subsampling this aggressive,
not hidden from the aggregate metric.

## Takeaway

**A second, independent axis confirms the same conditional-benefit pattern**
found in [`results/capacity_starved_retrieval/`](../capacity_starved_retrieval/README.md):
retrieval doesn't help a model that already has enough of *something*
(capacity there, training examples here) to learn the task well on its own,
but becomes reliably useful once that resource is scarce enough to bind.
Here, the crossover point is around 20% of the full training set
(~30,639 rows) -- below that, marginalized retrieval wins on every draw
tested except one (10%, draw 5, an essentially-zero Δ of -0.0003).

Combined with the capacity result, this makes a stronger case that the
`MedRAG/textbooks` retrieval corpus carries real, usable signal: it shows
up as a consistent benefit whether the bottleneck is model expressiveness
or labeled-example count, two structurally unrelated constraints. It also
sharpens the practical takeaway for when to reach for retrieval in this
setup: not by default, but specifically in low-resource regimes -- either a
deliberately small model or a small labeled cohort -- where the base
patient encoder alone can't fully learn the task.

**What's not yet tested here** (mirroring the same open question in the
capacity-starved README): whether this benefit is retrieval *content*
specifically, or partly an implicit ensembling/regularization effect from
marginalizing over K=4 documents regardless of relevance. The same
`retriever.ablation_mode=random_docs` check proposed there would apply
here too, at the 20%/10%/5% data levels where the effect is now
consistently positive.
