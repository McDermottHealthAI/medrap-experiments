# mimic_iv_sweep_frequent

Most-frequent-code variant of [`mimic_iv_sweep`](../mimic_iv_sweep/)'s Phase 4
(`sweep_marginalized_n1248.sh`). Everything is identical to that run
(architecture, hyperparameters, N ∈ {1, 2, 4, 8, 16, 32, 64, 128}, random prediction times)
**except how task codes are chosen**: instead of sampling uniformly at random
from the train-split vocabulary (which, per
[`mimic_iv_sweep`'s README](../mimic_iv_sweep/README.md#phase-1--task-count-sweep-sweep_task_countsh),
overwhelmingly lands on rare/degenerate codes), this variant picks the `N`
codes with the **highest distinct-subject count** in the train split (not raw
event-row count, which over-weights codes measured repeatedly on a small
subject subset).

Requires `medrap` built from
[`McDermottHealthAI/MedRAP@2c5fc5a`](https://github.com/McDermottHealthAI/MedRAP/commit/2c5fc5a668bf17394a58c49a4949ab6764ef0bc4)
(`experiment/marginalized-binary-plus-task-gen`, not yet merged to `main`),
pinned in `pyproject.toml`. That commit merges two not-yet-merged MedRAP
branches so both features are available together in one env:

- `feat/task-gen-most-frequent-codes` — adds `code_selection=random|most_frequent`
    to `medrap-preprocess`. `most_frequent` ranks codes by distinct-subject
    count in the train split, not raw event-row count -- a code measured
    repeatedly on a small subject subset (e.g. hourly ICU labs) can dominate
    row count while still being near-zero prevalence in the per-subject
    labels this pipeline produces.
- [`fix/marginalized-binary-output-mode`](https://github.com/McDermottHealthAI/MedRAP/pull/93)
    (MedRAP PR #93) — adds `marginalized_output_mode=categorical|binary` to
    `RetrievalAugmentedModel`. See `sweep_marginalized_binary_n1248.sh` below.

`experiment/marginalized-binary-plus-task-gen` is a merge-only integration
branch for this experiment's pin -- it isn't itself a PR under review; once
both `feat/task-gen-most-frequent-codes` and PR #93 land on `main`, drop
back to a main-tracking pin.

## Setup

```bash
cd mimic_iv_sweep_frequent
uv sync
```

## How to run

Reuses [`mimic_iv_sweep`](../mimic_iv_sweep/)'s retrieval index
(`../mimic_iv_sweep/data/retrieval_db`) rather than rebuilding it — the
retrieval corpus (`MedRAG/textbooks`) doesn't depend on task-code selection.
Run `mimic_iv_sweep`'s `prepare_retrieval.sh` first if that index doesn't
exist yet.

```bash
sbatch ../mimic_iv_sweep/scripts/prepare_retrieval.sh   # once, if data/retrieval_db doesn't exist yet
cd mimic_iv_sweep_frequent
sbatch scripts/generate_labels_n1248_frequent.sh        # N=1,2,4,8,16,32,64,128 task labels, most-frequent codes
sbatch scripts/sweep_patient_only_n1248.sh               # no retrieval, N=1,2,4,8,16,32,64,128
sbatch scripts/sweep_retrieval_n1248.sh                  # non-marginalized retrieval, N=1,2,4,8,16,32,64,128
sbatch scripts/sweep_marginalized_n1248.sh               # marginalized retrieval, categorical (buggy), N=1,2,4,8,16,32,64,128
sbatch scripts/sweep_marginalized_binary_n1248.sh        # marginalized retrieval, binary (MedRAP#93 fix), N=1,2,4,8,16,32,64,128
```

All four run the same task labels and hyperparameters (3 epochs, lr=1e-3,
batch size 32, `max_seq_len=256`) — only the architecture/output mode
differs, mirroring `../mimic_iv_sweep/scripts/sweep_architecture.sh`'s three
arms plus one MedRAP#93 variant:

- `sweep_patient_only_n1248.sh` — `patient_only`: RoPE encoder → masked-mean
    pooling → `LinearHead(B,N)` directly, no retriever/fusion at all
    (`fusion=passthrough`). Doesn't touch `data/retrieval_db` — no
    `prepare_retrieval.sh` prerequisite. Isolates how much the patient's own
    EHR sequence alone can do on these tasks.
- `sweep_retrieval_n1248.sh` — `retrieval`: single pooled cross-attention
    fusion over `k=4` retrieved docs, `training/loss=multitask_binary_bce`.
    Runs the same, non-marginalized architecture as the older
    `mt25-rope-cross-attn` runs, just on frequent-code labels instead of
    `mt25`'s pre-`MedRAP#92` positive-rate/count-filtered ones.
- `sweep_marginalized_n1248.sh` — `marginalized`, `marginalized_output_mode=categorical`
    (the current MedRAP default): per-document fusion +
    `marginalized_retrieval=true` + `multitask_binary_bce_marginalized` loss.
    Routes `logits` through `RetrievalAugmentedModel`'s per-document
    softmax-over-tasks path (`_marginal_class_probabilities`) -- correct for
    a single mutually-exclusive `C`-way task, but wrong for `N` independent
    binary tasks, which is suspected to produce the invalid AUROC seen in
    this PR's `marginalized-frequent-n{N}` results (bimodal near-0/near-1
    per-task AUROC).
- `sweep_marginalized_binary_n1248.sh` — identical to
    `sweep_marginalized_n1248.sh` plus one override,
    `marginalized_output_mode=binary`
    ([MedRAP PR #93](https://github.com/McDermottHealthAI/MedRAP/pull/93)):
    marginalizes each task's sigmoid probability independently over
    documents instead of forcing all tasks to compete via softmax. This is
    the same fix a colleague's unmerged branch used to get a genuinely good,
    uniform marginalized AUROC (0.916 mean, no bimodal collapse) on the
    older `mt25-rope-cross-attn-marginalized` W&B run -- this script tests
    it on frequent-code labels before PR #93 merges to `main`.

`sweep_patient_only_n1248.sh` and `sweep_retrieval_n1248.sh` never touch the
marginalized softmax-over-tasks path at all, so together with
`sweep_marginalized_binary_n1248.sh` they give three same-labels,
same-retrieval-index reference points unconfounded by that question — worth
running before drawing conclusions from `sweep_marginalized_n1248.sh`'s
(categorical) AUROC.

Each script accepts extra `medrap-train`/`medrap-preprocess` Hydra
overrides, same as `mimic_iv_sweep`'s scripts.

### Checking task balance

```bash
python scripts/check_task_balance.py data/tasks/n8/tasks
```

Prints per-task, per-split positive rates and flags degenerate (single-class)
or out-of-band tasks. Worth running before `sweep_marginalized_n1248.sh` to
confirm frequent-code selection actually produced usable positive rates,
unlike the random-task variant's N=1/2/4/8 labels.

## Output layout

```
mimic_iv_sweep_frequent/
├── data/
│   └── tasks/
│       ├── n1/tasks/          # {train,tuning,held_out}.parquet + metadata
│       ├── n2/tasks/
│       └── ...
├── logs/                      # SLURM stdout/stderr per array job
└── outputs/
    ├── patient_only_n1248/
    │   ├── n1/                # checkpoints + W&B run (no retrieval)
    │   ├── n2/
    │   └── ...
    ├── retrieval_n1248/
    │   ├── n1/                # checkpoints + W&B run (non-marginalized)
    │   ├── n2/
    │   └── ...
    ├── marginalized_n1248/
    │   ├── n1/                # checkpoints + W&B run (marginalized, categorical)
    │   ├── n2/
    │   └── ...
    └── marginalized_binary_n1248/
        ├── n1/                # checkpoints + W&B run (marginalized, binary -- MedRAP#93)
        ├── n2/
        └── ...
```

(No `data/retrieval_db/` here — shared from `../mimic_iv_sweep/data/retrieval_db`.)

All runs are logged to W&B under the `medrap` project, run names prefixed
`patient-only-frequent-n{N}-*`, `retrieval-frequent-n{N}-*`,
`marginalized-frequent-n{N}-*`, and `marginalized-binary-frequent-n{N}-*`
respectively, alongside the random-task variant's `marginalized-n{N}-*` for
direct comparison of `val/auroc/mean`, `val/auroc/n_valid_tasks` vs
`n_tasks`, and `train/loss`.

## Results: patient_only vs. marginalized (binary mode)

Frequent-code labels, N ∈ {1, 2, 4, 8, 16, 32, 64, 128}, comparing
`sweep_patient_only_n1248.sh` against `sweep_marginalized_binary_n1248.sh`
(`marginalized_output_mode=binary`, [MedRAP PR #93](https://github.com/McDermottHealthAI/MedRAP/pull/93)).
Categorical marginalized mode (the current MedRAP default, pre-#93) is
omitted from this comparison going forward -- see the caveat section above
for why it's not a fair comparison (softmax-over-tasks metric bug).

### Average AUROC

| N | patient_only | marginalized (binary) | Δ (binary − patient_only) | valid tasks |
| --- | --- | --- | --- | --- |
| 1 | [0.9866](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/b252lcbs) | [0.9823](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/b084s53o) | -0.0043 | 1/1 |
| 2 | [0.8894](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/lsgx9cwn) | [0.8709](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/r73ezhcr) | -0.0185 | 2/2 |
| 4 | [0.9808](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/uglkrb02) | [0.9457](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/pf0g1pop) | -0.0351 | 4/4 |
| 8 | [0.8873](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/kesr4bob) | [0.8630](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/fruenovy) | -0.0243 | 8/8 |
| 16 | [0.9107](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/rcg52tcf) | [0.7198](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/nppi07vm) | -0.1909 | 16/16 |
| 32 | [0.9855](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/or2dgh80) | [0.9867](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/0gwrqwfn) | +0.0012 | 28/32 |
| 64 | [0.8987](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/7am95jcz) | [0.9233](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/3xhox0y2) | +0.0246 | 63/64 |
| 128 | [0.9454](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/k0socaoo) | [0.9310](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/2rrgfx15) | -0.0144 | 124/128 |

Values link directly to the W&B run page. `val/auroc/mean` from W&B. "valid tasks" is the number of the N sampled
codes that had both classes present in the held-out validation split for
that split's labels -- identical between the two runs at a given N, since
both train on the exact same `data/tasks/n<N>/tasks/` label file (task
validity depends only on the labels, not the model).

### Per-task win rate

Of the valid (scoreable) tasks at each N, the fraction where
`marginalized (binary)`'s per-task AUROC beat `patient_only`'s on that same
task:

| N | marginalized (binary) wins | valid tasks | win rate |
| --- | --- | --- | --- |
| 1 | 0 | 1 | 0.0% |
| 2 | 0 | 2 | 0.0% |
| 4 | 2 | 4 | 50.0% |
| 8 | 3 | 8 | 37.5% |
| 16 | 5 | 16 | 31.2% |
| 32 | 12 | 28 | 42.9% |
| 64 | 38 | 63 | 60.3% |
| 128 | 16 | 124 | 12.9% |

No clear trend with N -- win rate ranges from 0% (N=1, N=2, both single-
or two-task runs) up to 60.3% (N=64), back down to 12.9% (N=128). At the
task level, `marginalized (binary)` is not uniformly better or worse than
`patient_only`; the two architectures pick up signal on different subsets
of tasks, and which one wins in aggregate depends on which tasks happen to
be sampled at a given N. N=16 is the one run so far where `patient_only`
wins by a wide margin (0.9107 vs. 0.7198, an 11/16 task deficit) -- worth
a closer look before treating the fix as fully validated across the whole
N range.

### Task codes used

The exact `num_tasks` codes selected by `code_selection=most_frequent` at
each N (from `data/tasks/n<N>/tasks/code_index.json`; same codes for both
`patient_only` and `marginalized (binary)` at a given N, since both train
on the same label file):

<details><summary><code>N=1</code> (1 code)</summary>

```
0: LAB//220650//g/dL//value_[6.2,6.5)
```

</details>

<details><summary><code>N=2</code> (2 codes)</summary>

```
0: LAB//224731//UNK
1: DIAGNOSIS//ICD//10//Z993
```

</details>

<details><summary><code>N=4</code> (4 codes)</summary>

```
0: MEDICATION//START//OLANZapine (Disintegrating Tablet)
1: LAB//50981//mg/dL//value_[4.0,5.0)
2: LAB//224952//cm//value_[3.0,5.0)
3: LAB//50958//mIU/mL//value_[5.7,7.0)
```

</details>

<details><summary><code>N=8</code> (8 codes)</summary>

```
0: INFUSION_END//221456//value_[2.0,inf)
1: MEDICATION//Lidocaine 5% Patch//Assessed
2: LAB//225674//%//value_[78.0,inf)
3: LAB//227583//UNK
4: DIAGNOSIS//ICD//10//Z87440
5: LAB//52020//UNK
6: DIAGNOSIS//ICD//9//E8498
7: LAB//50822//mEq/L
```

</details>

<details><summary><code>N=16</code> (16 codes)</summary>

```
0: LAB//228898//UNK
1: TRANSFER_TO//admit//Hematology/Oncology Intermediate
2: LAB//50963//pg/mL//value_[-inf,85.0)
3: LAB//51275//sec//value_[27.6,29.3)
4: LAB//224639//kg//value_[107.9,120.9)
5: MEDICATION//STOP//Ciprofloxacin HCl
6: LAB//223982//UNK
7: MEDICATION//Magnesium Oxide//Not Given
8: LAB//225192//UNK//value_[1.0,inf)
9: DIAGNOSIS//ICD//9//V4589
10: DIAGNOSIS//ICD//10//J9811
11: LAB//51218//#/uL
12: LAB//223835//UNK//value_[60.0,70.0)
13: LAB//51133//K/uL
14: LAB//51200//%//value_[0.0,0.3)
15: DIAGNOSIS//ICD//9//5849
```

</details>

<details><summary><code>N=32</code> (32 codes)</summary>

```
0: MEDICATION//Bisacodyl//Not Given
1: INFUSION_END//229297//value_[799.3,inf)
2: LAB//224688//insp/min//value_[16.0,18.0)
3: MEDICATION//START//OxyCODONE Liquid
4: LAB//51108//mL//value_[1950.0,2250.0)
5: MEDICATION//STOP//PrednisoLONE Acetate 1% Ophth. Susp.
6: INFUSION_END//221668//value_[16.4,49.750004)
7: SUBJECT_FLUID_OUTPUT//227510//mL//value_[5.0,10.0)
8: LAB//224960//UNK
9: LAB//50949//mg/dL//value_[311.0,458.0)
10: LAB//50916//ug/dL//value_[86.0,118.0)
11: DIAGNOSIS//ICD//9//25050
12: DIAGNOSIS//ICD//9//71941
13: LAB//220546//K/uL//value_[5.6,7.2)
14: LAB//51248//pg//value_[26.4,27.9)
15: DRG//APR//244//DIVERTICULITIS AND DIVERTICULOSIS
16: LAB//220339//cmH2O//value_[5.0,8.0)
17: LAB//229663//cmH2O//value_[11.0,13.0)
18: MEDICATION//Acetylcysteine 20%//Administered
19: DRG//APR//751//MAJOR DEPRESSIVE DISORDERS AND OTHER OR UNSPECIFIED PSYCHOSES
20: MEDICATION//PNEUMOcoccal 23-valent polysaccharide vaccine//Not Given
21: INFUSION_END//225944//value_[99.7,100.0)
22: LAB//225628//%//value_[3.2,4.6)
23: LAB//51802//mg/dL//value_[30.0,34.0)
24: MEDICATION//STOP//OLANZapine (Disintegrating Tablet)
25: MEDICATION//START//Dermoplast Spray
26: LAB//52391//#/uL//value_[-inf,80.0)
27: LAB//51795//IU/L//value_[13.0,16.0)
28: LAB//51688//UNK
29: DIAGNOSIS//ICD//10//K660
30: MEDICATION//Mouth Care Oral Rinse//Administered
31: LAB//223958//mV//value_[2.0,2.5)
```

</details>

<details><summary><code>N=64</code> (64 codes)</summary>

```
0: LAB//226512//kg//value_[97.1,109.4)
1: INFUSION_END//225942//value_[0.12986197,0.214)
2: MEDICATION//START//Sodium Bicarbonate
3: INFUSION_START//225170//value_[276.0,293.0)
4: LAB//229660//sec//value_[-inf,0.36)
5: LAB//51300//K/uL//value_[5.8,6.3)
6: INFUSION_END//225152//value_[2586.2815,4333.333)
7: BMI
8: MEDICATION//STOP//Dextrose 50%
9: LAB//224754//mA//value_[5.0,6.0)
10: LAB//50909//ug/dL//value_[20.3,24.7)
11: INFUSION_START//226364
12: LAB//224833//UNK
13: INFUSION_START//225828//value_[70.01167,76.43312)
14: INFUSION_START//225910
15: LAB//51689//UNK//value_[0.5,0.66)
16: INFUSION_START//225170//value_[202.0,237.0)
17: MEDICATION//START//Topiramate (Topamax)
18: LAB//50842//mg/dL//value_[125.0,135.0)
19: MEDICATION//STOP//Nicotine Patch
20: LAB//50958//mIU/mL
21: INFUSION_END//220995//value_[16.3125,33.75)
22: MEDICATION//START//DICYCLOMine
23: LAB//225092//UNK//value_[0.0,1.0)
24: LAB//223960//mA//value_[15.0,inf)
25: LAB//51753//UNK
26: MEDICATION//STOP//Ursodiol
27: LAB//51275//sec//value_[25.7,27.6)
28: LAB//51196//ng/mL FEU//value_[3155.0,5442.0)
29: LAB//223938//UNK
30: LAB//225671//mg/dL//value_[89.0,101.0)
31: MEDICATION//START//Tacrolimus
32: INFUSION_START//225153//value_[2.9896908,4.0)
33: LAB//223830//units//value_[7.42,7.45)
34: LAB//229864//UNK//value_[-inf,8.0)
35: LAB//50970//mg/dL//value_[4.8,inf)
36: LAB//224332//UNK//value_[19.0,inf)
37: MEDICATION//START//OxyCODONE Liquid
38: DIAGNOSIS//ICD//10//I495
39: LAB//228723//cm//value_[1.5,2.0)
40: INFUSION_END//223259//value_[20.0,28.0)
41: MEDICATION//Ipratropium-Albuterol Neb//Delayed Administered
42: SUBJECT_FLUID_OUTPUT//226588//mL//value_[40.0,50.0)
43: MEDICATION//Pregabalin//Not Given
44: LAB//226537//UNK//value_[116.0,123.0)
45: INFUSION_END//225171//value_[117.0,128.0)
46: DIAGNOSIS//ICD//9//40390
47: INFUSION_END//225797//value_[300.0,inf)
48: LAB//229371//mL//value_[344.0,405.0)
49: MEDICATION//STOP//Milk of Magnesia
50: LAB//51254//%//value_[8.5,9.9)
51: LAB//50884//mg/dL//value_[-inf,0.2)
52: LAB//51450//%//value_[3.0,5.0)
53: LAB//229538//UNK//value_[0.0,inf)
54: LAB//225628//%//value_[9.8,12.9)
55: LAB//50903//Ratio//value_[4.0,4.5)
56: LAB//53171//UNK//value_[25.25,25.43)
57: LAB//50973//ng/mL//value_[29.0,49.0)
58: LAB//227582//L/min//value_[6.0,8.0)
59: MEDICATION//STOP//Bumetanide
60: LAB//51689//UNK
61: LAB//224769//UNK
62: LAB//224916//cm//value_[1.0,1.5)
63: SUBJECT_FLUID_OUTPUT//226626//mL//value_[10.0,40.0)
```

</details>

<details><summary><code>N=128</code> (128 codes)</summary>

```
0: LAB//51579//U/mL//value_[8.0,18.0)
1: INFUSION_END//221468//value_[12.499999,20.0)
2: MEDICATION//Magnesium Sulfate Replacement (Critical Care and Oncology)//Administered
3: LAB//50980//IU/mL//value_[17.0,34.0)
4: LAB//50824//mEq/L//value_[140.0,143.0)
5: LAB//226531//UNK//value_[172.5,184.6)
6: LAB//50964//mOsm/kg
7: LAB//51176//%//value_[58.0,65.0)
8: INFUSION_START//222042//value_[1.2464218,1.5013012)
9: INFUSION_END//221744//value_[25.0,50.0)
10: LAB//220545//%//value_[34.1,37.5)
11: LAB//220052//mmHg//value_[80.0,84.0)
12: LAB//50994//ug/dL//value_[9.6,11.4)
13: MEDICATION//Octreotide Acetate//Started
14: LAB//50964//mOsm/kg//value_[-inf,267.0)
15: MEDICATION//Acetaminophen//Administered
16: LAB//223772//%//value_[53.0,58.0)
17: LAB//224328//UNK//value_[0.24,0.25)
18: LAB//225640//%//value_[0.0,0.1)
19: LAB//225641//%//value_[12.0,14.5)
20: LAB//50861//IU/L//value_[52.0,93.0)
21: DIAGNOSIS//ICD//10//B9689
22: LAB//51078//mEq/L//value_[-inf,15.0)
23: LAB//51066//mg/24hr//value_[346.0,inf)
24: MEDICATION//START//Calcium Replacement (Oncology)
25: INFUSION_END//227526//value_[110.07751,111.43229)
26: DIAGNOSIS//ICD//9//07054
27: INFUSION_END//225161//value_[35.0,67.08137)
28: LAB//51066//mg/24hr//value_[65.0,91.0)
29: LAB//50982//nmol/L//value_[46.0,55.0)
30: LAB//50902//mEq/L//value_[101.0,102.0)
31: LAB//51398//UNK//value_[12.5,13.8)
32: MEDICATION//STOP//CARVedilol
33: LAB//51689//UNK//value_[0.5,0.66)
34: LAB//50802//mEq/L//value_[2.0,3.0)
35: LAB//227073//mEq/L//value_[16.0,17.0)
36: LAB//226873//UNK//value_[1.0,inf)
37: LAB//50986//ng/mL//value_[6.3,7.1)
38: INFUSION_END//229297//value_[128.0,184.83334)
39: LAB//50937//N/A
40: LAB//227464//mEq/L//value_[5.2,inf)
41: INFUSION_END//225154//value_[2.0,2.5)
42: LAB//52369//#/uL//value_[225.0,590.0)
43: LAB//50964//mOsm/kg//value_[324.0,inf)
44: LAB//224409//UNK//value_[1.0,2.0)
45: DIAGNOSIS//ICD//10//R001
46: LAB//50956//IU/L//value_[33.0,39.0)
47: LAB//50986//ng/mL//value_[-inf,3.8)
48: LAB//51006//mg/dL//value_[13.0,16.0)
49: LAB//50867//IU/L//value_[64.0,76.0)
50: LAB//52264//%//value_[82.0,88.0)
51: PROCEDURE//START//224270
52: LAB//223923//UNK
53: DIAGNOSIS//ICD//10//Z7984
54: INFUSION_END//225171//value_[100.0,103.0)
55: LAB//220045//bpm//value_[75.0,80.0)
56: LAB//50815//L/min//value_[4.0,5.0)
57: LAB//226063//mmHg//value_[48.0,53.0)
58: SUBJECT_FLUID_OUTPUT//226613//mL//value_[200.0,300.0)
59: LAB//224747//cmH2O//value_[4.0,5.6)
60: INFUSION_END//229297//value_[240.0,325.0)
61: LAB//224663//UNK//value_[1.0,3.0)
62: LAB//227465//sec//value_[13.6,14.3)
63: LAB//225732//UNK//value_[0.0,inf)
64: LAB//51093//mOsm/kg//value_[479.0,549.0)
65: LAB//226169//UNK//value_[-inf,1.0)
66: INFUSION_START//225154
67: LAB//51463//UNK
68: DRG//APR//282//DISORDERS OF PANCREAS EXCEPT MALIGNANCY
69: MEDICATION//STOP//MetFORMIN XR (Glucophage XR)
70: LAB//228724//cm//value_[6.0,inf)
71: MEDICATION//START//Iohexol 240 (Omnipaque 240)
72: LAB//223810//UNK//value_[4.0,inf)
73: LAB//52008//UNK//value_[-119.0,inf)
74: INFUSION_END//226372//value_[450.0,500.0)
75: LAB//226062//mmHg//value_[42.0,45.0)
76: LAB//51754//UNK//value_[2.53,3.01)
77: TRANSFER_TO//transfer//Medical/Surgical (Gynecology)
78: LAB//51085//N/A
79: LAB//50910//IU/L//value_[324.0,880.0)
80: LAB//226117//UNK//value_[0.0,1.0)
81: LAB//226062//mmHg//value_[45.0,48.0)
82: LAB//51001//ng/dL//value_[135.0,170.0)
83: LAB//50890//mg/dL//value_[96.0,106.0)
84: LAB//52281//%//value_[0.0,2.0)
85: LAB//227442//mEq/L//value_[3.9,4.1)
86: LAB//51005//Ratio//value_[0.96,0.99)
87: LAB//228610//cm//value_[-inf,0.0)
88: DIAGNOSIS//ICD//9//3659
89: LAB//229154//UNK
90: INFUSION_START//221385//value_[4.0064263,5.989189)
91: LAB//224161//insp/min//value_[35.0,inf)
92: MEDICATION//STOP//Dermoplast Spray
93: INFUSION_END//227525//value_[17.707994,19.917013)
94: LAB//224057//UNK//value_[2.0,3.0)
95: LAB//220235//mmHg//value_[55.0,inf)
96: LAB//51228//IU/mL//value_[0.54,0.7)
97: INFUSION_END//221662//value_[45.662098,73.44159)
98: DIAGNOSIS//ICD//10//I120
99: DIAGNOSIS//ICD//9//5939
100: INFUSION_START//220864//value_[249.99998,499.99997)
101: LAB//227580//cmH2O//value_[17.0,20.0)
102: INFUSION_START//225825//value_[101.05931,150.0)
103: LAB//51200//%//value_[0.7,1.0)
104: LAB//227667//UNK
105: DIAGNOSIS//ICD//10//Z7289
106: LAB//226534//mEq/L//value_[133.0,134.0)
107: LAB//220045//bpm//value_[89.0,95.0)
108: DIAGNOSIS//ICD//10//B961
109: LAB//50849//g/dL//value_[1.2,1.5)
110: LAB//52172//fL
111: LAB//50824//mEq/L//value_[136.0,137.0)
112: INFUSION_END//222168//value_[45.0,93.98496)
113: INFUSION_START//221347//value_[0.5,0.5000608)
114: LAB//227130//UNK//value_[-inf,1.0)
115: LAB//224754//mA//value_[7.0,8.0)
116: LAB//51438//#/uL//value_[3.0,20.0)
117: LAB//228006//ml/hr//value_[400.0,inf)
118: LAB//51265//K/uL//value_[135.0,167.0)
119: LAB//224738//sec//value_[1.0,inf)
120: LAB//50992//IU/mL//value_[217.0,452.0)
121: LAB//224696//cmH2O//value_[17.0,18.0)
122: MEDICATION//STOP//Phosphorus
123: LAB//224059//UNK//value_[3.0,inf)
124: LAB//50938//N/A
125: INFUSION_END//227526//value_[104.997,110.07751)
126: SUBJECT_FLUID_OUTPUT//227701//mL//value_[160.0,220.0)
127: LAB//51008//ug/mL//value_[45.0,53.0)
```

</details>
