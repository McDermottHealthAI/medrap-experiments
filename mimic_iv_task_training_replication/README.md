# MIMIC-IV retrieval replication across task and training seeds

Completed: 27 runs, three task seeds x three training seeds x three independently trained arms.
Real retrieval had lower mean held-out AUROC than patient-only and a mean advantage of 0.00028 over random.

## Motivation

We compare patient-only, real retrieval and random retrieval across three task draws and three training
seeds on MIMIC-IV.

## Setup

| Setting | Value |
| --- | --- |
| Source | MedRAP PR113, commit `8171aeca44b92b0c778ba96fe045181b5a36acba` |
| Preprocessing | MedRAP defaults: minimum 100 subjects per code, 10 events per subject |
| Tasks | 25 random codes per panel; task seeds 101, 202, 303; fixed 30-day horizon |
| Anchors | Uniform-event sampling; minimum history one day |
| Patients per panel | 153220 fitting / 19180 tuning / 19158 held-out; disjoint splits |
| Training seeds | 42, 43, 44, paired across arms |
| Arms | Patient-only, real marginal retrieval, random marginal retrieval |
| Retrieval | Unfiltered textbooks, 125847 documents, four documents per query, GPU FAISS |
| Patient encoder | RoPE, width 128, two layers, four heads; latest 256 events |
| Query / document encoder | Learned sequence_mean_1024 / token_feature width 64 |
| Fusion | Per-document cross-attention for real/random; passthrough for patient-only |
| Optimization | FP32, batch size 32, AdamW learning rate 0.001, weight decay 0.01, gradient clipping at 1 |
| Schedule | Five epochs, cosine decay, 200 warmup steps |
| Exposure per fit | 766100 patient examples; 23945 optimizer steps |
| Checkpoint | Explicit final checkpoint after five epochs; no best-metric selection |
| Monitoring | Full tuning split after each epoch; no early stopping |

Random documents are sampled uniformly with replacement from the same corpus using package random_docs.
Real/random have matched architectures; patient-only has a smaller prediction architecture.
Shared component initialization and patient shuffling are paired by training seed.

## Results

Each cell below is held-out macro AUROC, mean +/- sample SD across the three training seeds.

| Task seed | Patient-only | Real retrieval | Random retrieval |
| --- | ---: | ---: | ---: |
| 101 | 0.89146 +/- 0.00463 | 0.89006 +/- 0.00463 | 0.89640 +/- 0.00680 |
| 202 | 0.86627 +/- 0.00844 | 0.86098 +/- 0.01646 | 0.85474 +/- 0.02041 |
| 303 | 0.87532 +/- 0.00753 | 0.87605 +/- 0.00934 | 0.87511 +/- 0.00407 |

Mean +/- sample SD over the nine task/training seed combinations per arm:

| Arm | Macro AUROC | Predictive BCE |
| --- | ---: | ---: |
| Patient-only | 0.877682 +/- 0.012626 | 0.024349 +/- 0.009462 |
| Real retrieval | 0.875696 +/- 0.015921 | 0.024642 +/- 0.009461 |
| Random retrieval | 0.875416 +/- 0.021101 | 0.024704 +/- 0.009472 |

Real-minus-patient-only AUROC averaged -0.001986 (-0.199 percentage points), with real higher in 6/9
paired runs. Real-minus-random averaged +0.000280 (+0.028 percentage points), with real higher in 5/9.
Patient-only had lower predictive BCE than real in all nine comparisons.

SDs in the per-panel table measure training-seed variation. Overall SDs combine task-panel and
training-seed variation; they are not standard errors or confidence intervals. Panels share patients,
and BCE also varies with the target prevalences in each panel. No patient-level confidence intervals
were computed. Evaluation uses previously inspected cohorts.

## Task codes

Task IDs below match the label columns task_0 through task_24. Each task-seed panel is shared by all
training seeds and arms. Codes are copied from the saved code_index.json files without renaming.

Task generation is implemented in MedRAP's
[task_generation.py](https://github.com/McDermottHealthAI/MedRAP/blob/8171aeca44b92b0c778ba96fe045181b5a36acba/src/medrap/preprocess/task_generation.py),
exposed through medrap-preprocess. This experiment reused the existing task panels.
Regeneration with the same seeds selected different codes: the candidate-code list is unordered before
seeded sampling. Exact reproduction requires the saved panels, not the seeds alone.

### Task seed 101

| Task ID | Code |
| --- | --- |
| 0 | `LAB//227466//sec//value_[39.4,48.5)` |
| 1 | `LAB//51249//g/dL//value_[34.1,inf)` |
| 2 | `MEDICATION//Insulin// in Other Location` |
| 3 | `DIAGNOSIS//ICD//10//Z9282` |
| 4 | `LAB//50861//IU/L//value_[20.0,24.0)` |
| 5 | `LAB//224897//cm//value_[0.0,0.2)` |
| 6 | `LAB//224943//UNK` |
| 7 | `MEDICATION//START//Sodium Chloride` |
| 8 | `LAB//51301//K/uL//value_[-inf,3.9)` |
| 9 | `DIAGNOSIS//ICD//9//7948` |
| 10 | `DIAGNOSIS//ICD//10//B948` |
| 11 | `MEDICATION//PHENYLEPHrine//Stopped - Unscheduled` |
| 12 | `MEDICATION//STOP//Nabumetone` |
| 13 | `INFUSION_END//226364//value_[3200.0,4000.0)` |
| 14 | `MEDICATION//Ibuprofen//Administered` |
| 15 | `LAB//51626//UNK//value_[40.74,inf)` |
| 16 | `DIAGNOSIS//ICD//9//32723` |
| 17 | `DIAGNOSIS//ICD//10//K869` |
| 18 | `LAB//51244//%//value_[41.0,inf)` |
| 19 | `MEDICATION//Glycopyrrolate (CVICU Reversal Protocol)//Administered` |
| 20 | `SUBJECT_FLUID_OUTPUT//226582//mL//value_[100.0,130.0)` |
| 21 | `MEDICATION//START//Amphetamine-Dextroamphetamine XR` |
| 22 | `LAB//227580//cmH2O//value_[14.0,15.0)` |
| 23 | `LAB//224979//UNK` |
| 24 | `LAB//225341//UNK` |

### Task seed 202

| Task ID | Code |
| --- | --- |
| 0 | `LAB//51181//Ratio//value_[-inf,0.2)` |
| 1 | `DIAGNOSIS//ICD//9//78906` |
| 2 | `DRG//HCFA//356//OTHER DIGESTIVE SYSTEM O.R. PROCEDURES W MCC` |
| 3 | `LAB//50867//IU/L//value_[76.0,90.0)` |
| 4 | `LAB//52500//mEq/L//value_[14.0,inf)` |
| 5 | `MEDICATION//STOP//ranolazine` |
| 6 | `LAB//50821//mm Hg//value_[113.0,134.0)` |
| 7 | `LAB//52075//K/uL//value_[10.83,inf)` |
| 8 | `MEDICATION//STOP//Dronabinol` |
| 9 | `MEDICATION//Losartan Potassium//Delayed Administered` |
| 10 | `DRG//HCFA//280//ACUTE MYOCARDIAL INFARCTION, DISCHARGED ALIVE WITH MCC` |
| 11 | `LAB//50826//UNK//value_[450.0,480.0)` |
| 12 | `MEDICATION//MetroNIDAZOLE// in Other Location` |
| 13 | `LAB//229476//UNK//value_[1.0,inf)` |
| 14 | `DIAGNOSIS//ICD//9//78321` |
| 15 | `LAB//224746//cmH2O//value_[0.0,0.8)` |
| 16 | `MEDICATION//Varicella Virus Vaccine//Administered` |
| 17 | `MEDICATION//START//Divalproex (DELayed Release)` |
| 18 | `DRG//HCFA//66//INTRACRANIAL HEMORRHAGE OR CEREBRAL INFARCTION W/O CC/MCC` |
| 19 | `MEDICATION//Latanoprost 0.005% Ophth. Soln.//Not Given` |
| 20 | `MEDICATION//STOP//Arimidex` |
| 21 | `LAB//229633//UNK//value_[19.39,25.8)` |
| 22 | `LAB//223811//UNK//value_[0.0,1.0)` |
| 23 | `PROCEDURE//ICD//10//0QS606Z` |
| 24 | `PROCEDURE//START//224272` |

### Task seed 303

| Task ID | Code |
| --- | --- |
| 0 | `MEDICATION//Gentamicin 0.3% Ophth. Soln//Administered` |
| 1 | `DIAGNOSIS//ICD//9//431` |
| 2 | `LAB//52024//mg/dL//value_[1.6,2.1)` |
| 3 | `MEDICATION//HYDROmorphone (Dilaudid)//Not Given` |
| 4 | `LAB//224752//mV//value_[2.0,inf)` |
| 5 | `MEDICATION//START//Imipramine` |
| 6 | `LAB//224359//sec//value_[-inf,0.38)` |
| 7 | `INFUSION_START//228340//value_[9.999999,10.0)` |
| 8 | `INFUSION_START//229630//value_[-inf,0.29984006)` |
| 9 | `LAB//224332//UNK//value_[2.0,3.0)` |
| 10 | `DIAGNOSIS//ICD//10//E8881` |
| 11 | `LAB//223763//mmHg//value_[18.0,19.0)` |
| 12 | `DIAGNOSIS//ICD//10//E118` |
| 13 | `MEDICATION//Amphetamine-Dextroamphetamine//Administered` |
| 14 | `DIAGNOSIS//ICD//10//A4152` |
| 15 | `LAB//51376//%//value_[-inf,2.0)` |
| 16 | `PROCEDURE//ICD//9//527` |
| 17 | `LAB//227341//UNK//value_[0.0,25.0)` |
| 18 | `LAB//50980//IU/mL//value_[8.0,9.0)` |
| 19 | `LAB//50934//UNK//value_[5.0,6.0)` |
| 20 | `DIAGNOSIS//ICD//10//C772` |
| 21 | `DIAGNOSIS//ICD//9//496` |
| 22 | `INFUSION_START//221986//value_[0.2499333,0.25014886)` |
| 23 | `INFUSION_END//221555//value_[10.0,16.408169)` |
| 24 | `LAB//51257//%` |
