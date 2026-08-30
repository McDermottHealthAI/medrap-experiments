# Data-scarce `patient_only` vs. marginalized retrieval (N=25, 30d)

Status: complete. 40 training jobs (5 draws x 4 train-fractions x 2
architectures), plus test-split repro, top1-only eval, and random/null-doc
ablations (separate README, linked below).

Scripts: [`subsample_train_labels.py`](../../mimic_iv_sweep/scripts/subsample_train_labels.py),
[`sweep_patient_only_data_scarcity_n25_30d_train{50,20,10,5}pct.sh`](../../mimic_iv_sweep/scripts/),
[`sweep_marginalized_binary_learned_linear_data_scarcity_n25_30d_train{50,20,10,5}pct.sh`](../../mimic_iv_sweep/scripts/)
(training, `mimic_iv_sweep/`); [`mimic_iv_data_scarcity_ablations/scripts/`](../../mimic_iv_data_scarcity_ablations/README.md)
(test-split repro, top1-only eval, random/null-doc ablations).

## Config

Same architecture as [`results/capacity_starved_retrieval/`](../capacity_starved_retrieval/README.md),
at **full model capacity** (`encoder=rope` defaults, no capacity cut).
Only the training-set size varies: `train.parquet` is subsampled (uniform
random by row, fixed seed per draw, via `subsample_train_labels.py`).
`tuning.parquet` and `held_out.parquet` are unchanged at every fraction, so
every level is evaluated against the same validation/test data.
`patient_only` and `marginalized` train on the identical subsampled label
file per (draw, fraction).

| Fraction of train subjects kept | Approx. train rows (of ~153,195 full) |
| --- | --- |
| 50% | 76,598 |
| 20% | 30,639 |
| 10% | 15,320 |
| 5% | 7,660 |

N=25 task codes, 5 draws, 30d occurrence window, same seeds as the rest of
`mimic_iv_sweep`. `query_projector=sequence_mean_1024` ("learned-linear")
only, `retriever.k=4`.

```bash
cd mimic_iv_sweep
python scripts/subsample_train_labels.py \
    data/tasks_zach_uniform_event_n25_30d/draw${DRAW}/tasks \
    data/tasks_zach_uniform_event_n25_30d_train${FRAC}pct/draw${DRAW}/tasks \
    --fraction 0.${FRAC} --seed <seed>

sbatch --array=0-4 scripts/sweep_patient_only_data_scarcity_n25_30d_train50pct.sh
sbatch --array=0-4 scripts/sweep_marginalized_binary_learned_linear_data_scarcity_n25_30d_train50pct.sh
# ... and the same pair for train20pct / train10pct / train5pct
```

## Results, val split (`val/auroc/mean`)

### 50% train

| Draw | patient_only | marginalized | Δ |
| --- | --- | --- | --- |
| 1 | 0.9040 | 0.8881 | -0.0159 |
| 2 | 0.8888 | 0.8811 | -0.0077 |
| 3 | 0.8909 | 0.8888 | -0.0021 |
| 4 | 0.9240 | 0.9273 | +0.0033 |
| 5 | 0.8648 | 0.8604 | -0.0044 |
| **mean** | **0.8945** | **0.8891** | **-0.0053 ± 0.0071, 1/5 won** |

### 20% train

| Draw | patient_only | marginalized | Δ |
| --- | --- | --- | --- |
| 1 | 0.8543 | 0.8910 | +0.0367 |
| 2 | 0.8197 | 0.8428 | +0.0231 |
| 3 | 0.8648 | 0.8676 | +0.0028 |
| 4 | 0.8911 | 0.9022 | +0.0112 |
| 5 | 0.8314 | 0.8316 | +0.0002 |
| **mean** | **0.8522** | **0.8670** | **+0.0148 ± 0.0152, 5/5 won** |

### 10% train

| Draw | patient_only | marginalized | Δ |
| --- | --- | --- | --- |
| 1 | 0.8547 | 0.8677 | +0.0130 |
| 2 | 0.8285 | 0.8326 | +0.0041 |
| 3 | 0.8366 | 0.8478 | +0.0112 |
| 4 | 0.8364 | 0.8651 | +0.0287 |
| 5 | 0.7945 | 0.7942 | -0.0003 |
| **mean** | **0.8302** | **0.8415** | **+0.0113 ± 0.0111, 4/5 won** |

### 5% train

| Draw | patient_only | marginalized | Δ |
| --- | --- | --- | --- |
| 1 | 0.8392 | 0.8640 | +0.0248 |
| 2 | 0.7843 | 0.8057 | +0.0215 |
| 3 | 0.8143 | 0.8321 | +0.0178 |
| 4 | 0.8254 | 0.8318 | +0.0064 |
| 5 | 0.7593 | 0.7692 | +0.0099 |
| **mean** | **0.8045** | **0.8206** | **+0.0161 ± 0.0078, 5/5 won** |

### Summary

| Train fraction | patient_only | marginalized | Δ (mean ± std) | Draws won |
| --- | --- | --- | --- | --- |
| 50% | 0.8945 | 0.8891 | -0.0053 ± 0.0071 | 1/5 |
| 20% | 0.8522 | 0.8670 | +0.0148 ± 0.0152 | 5/5 |
| 10% | 0.8302 | 0.8415 | +0.0113 ± 0.0111 | 4/5 |
| 5% | 0.8045 | 0.8206 | +0.0161 ± 0.0078 | 5/5 |

## Results, test split (`test/auroc/mean`)

No published test table existed prior to the ablation round; these numbers
are from [`mimic_iv_data_scarcity_ablations`](../../mimic_iv_data_scarcity_ablations/README.md)'s
reproduction gate.

| Train fraction | patient_only | marginalized | Δ |
| --- | --- | --- | --- |
| 50% | 0.8824 ± 0.0203 | 0.8801 ± 0.0256 | -0.0023 |
| 20% | 0.8479 ± 0.0295 | 0.8541 ± 0.0281 | +0.0062 |
| 10% | 0.8225 ± 0.0276 | 0.8271 ± 0.0288 | +0.0046 |
| 5% | 0.7883 ± 0.0391 | 0.8070 ± 0.0288 | +0.0187 |

## Results, top1-only inference (test split, `retriever.k=1`, `ablation_mode=none`)

| Train fraction | marginalized (k=4) | marginalized (k=1) | Δ |
| --- | --- | --- | --- |
| 50% | 0.8801 | 0.8800 | -0.0001 |
| 20% | 0.8541 | 0.8539 | -0.0002 |
| 10% | 0.8271 | 0.8273 | +0.0002 |
| 5% | 0.8070 | 0.8067 | -0.0003 |

## Results, random-doc / null-doc ablations (test split, k=4)

Full tables: [`results/random_doc_null_doc_ablations_data_scarcity/README.md`](../random_doc_null_doc_ablations_data_scarcity/README.md).

| Train fraction | patient_only | real docs | random docs | null docs |
| --- | --- | --- | --- | --- |
| 50% | 0.8824 ± 0.0203 | 0.8801 ± 0.0256 | 0.8794 ± 0.0248 | 0.8783 ± 0.0253 |
| 20% | 0.8479 ± 0.0295 | 0.8541 ± 0.0281 | 0.8526 ± 0.0283 | 0.8531 ± 0.0273 |
| 10% | 0.8225 ± 0.0276 | 0.8271 ± 0.0288 | 0.8255 ± 0.0277 | 0.8185 ± 0.0307 |
| 5% | 0.7883 ± 0.0391 | 0.8070 ± 0.0288 | 0.8080 ± 0.0288 | 0.8049 ± 0.0294 |

## Caveat: training-positive scarcity at 10%/5%

At the most aggressive subsampling, some (draw, task) combinations have
zero training-set positives for some of the 25 task codes:

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

`patient_only` and `marginalized` train on the identical subsampled label
file for a given (draw, fraction), so both architectures face the same
positive-count handicap per task.
