# mimic_iv_sweep

Systematic sweep over task scale, architecture variants, and hyperparameters
on MIMIC-IV, building on results from [`mimic_iv/`](../mimic_iv/) and
[`mimic_iv_smoke/`](../mimic_iv_smoke/).

All scripts use the post-refactor flat CLI (`medrap-train`, `medrap-preprocess`,
`medrap-prepare-retrieval-dataset`) and reuse the lab-shared tensorized cohort
at `/groups/mm6677_gp/data/MIMIC_MEDS/MEDS_cohort/processed`.

## Setup

```bash
cd mimic_iv_sweep
uv sync
```

This creates `.venv/` and installs `medrap` and all dependencies (including
`torch` and `nvidia-*-cu12` CUDA packages) from PyPI.

## Sweep design

### Phase 1 — Task-count sweep (`sweep_task_count.sh`)

Fixed architecture (RoPE encoder + cross-attention medium fusion, k=4, lr=1e-3,
3 epochs). Varies N ∈ {1, 2, 4, 8, 16, 32} to measure how jointly
training more prediction targets affects per-task AUROC.

N is kept small: task codes are sampled from a long-tailed clinical
vocabulary, and only a handful of codes (e.g. Blood Pressure, Weight, BMI)
have enough in-window positive volume for a stable per-task AUROC given the
current anchor-sampling scheme (see `scripts/check_task_balance.py`) --
requesting a large N mostly adds low-count, noise-dominated tasks rather
than signal.

### Phase 2 — Architecture ablation (`sweep_architecture.sh`)

Fixed N ∈ {1, 8, 32}. Compares three architectures:

| Variant | Description |
| --- | --- |
| `patient_only` | RoPE encoder → masked-mean pooling → head; no retrieval |
| `retrieval` | RoPE + cross-attention medium fusion, k=4 retrieved docs |
| `marginalized` | Same as `retrieval` but with `multitask_binary_bce_marginalized` loss |

### Phase 3 — Hyperparameter sweep (`sweep_hparams.sh`)

Fixed N=8, RoPE + cross-attention, 3 epochs. Full grid:
k ∈ {4, 8, 16, 32} × lr ∈ {1e-4, 1e-3, 3e-3} → 12 jobs.

### Phase 4 — Marginalized retrieval, random tasks (`sweep_marginalized_n1248.sh`)

See [below](#marginalized-retrieval-random-tasks-phase-4) for the full
architecture breakdown, exact run commands, and known caveats.

## Running the sweep

### Step 1 — Build the retrieval index (once)

```bash
cd mimic_iv_sweep
sbatch scripts/prepare_retrieval.sh
```

### Step 2 — Generate task labels for each N (once per N)

```bash
sbatch scripts/generate_labels.sh          # all 6 N values in parallel
# or a subset, e.g. only N=8 and N=32:
sbatch --array=3,5 scripts/generate_labels.sh
```

Label N values and their array indices:

| Index | N |
| --- | --- |
| 0 | 1 |
| 1 | 2 |
| 2 | 4 |
| 3 | 8 |
| 4 | 16 |
| 5 | 32 |

### Step 3 — Run the sweeps

```bash
sbatch scripts/sweep_task_count.sh
sbatch scripts/sweep_architecture.sh
sbatch scripts/sweep_hparams.sh
```

Phase 4 (`sweep_marginalized_n1248.sh`) has its own task-generation script and run
instructions — see [below](#marginalized-retrieval-random-tasks-phase-4).

Each script accepts extra arguments forwarded to `medrap-train` as Hydra
overrides:

```bash
sbatch scripts/sweep_task_count.sh training.trainer.max_epochs=5
```

## Output layout

```
mimic_iv_sweep/
├── data/
│   ├── retrieval_db/          # FAISS index + HF dataset (prepare_retrieval.sh)
│   └── tasks/
│       ├── n1/tasks/          # {train,tuning,held_out}.parquet + metadata
│       ├── n2/tasks/
│       └── ...
├── logs/                      # SLURM stdout/stderr per array job
└── outputs/
    ├── task_count/
    │   ├── n1/                # checkpoints + W&B run
    │   ├── n2/
    │   └── ...
    ├── architecture/
    │   ├── patient_only_n8/
    │   ├── retrieval_n8/
    │   ├── marginalized_n8/
    │   └── ...
    └── hparams/
        ├── k4_lr1e-4/
        ├── k8_lr1e-3/
        └── ...
```

All runs are logged to W&B under the `medrap` project with run names prefixed
`sweep-task-count-*`, `sweep-arch-*`, and `sweep-hparams-*`.

## Marginalized retrieval, random tasks (Phase 4)

Compares two architectures on task labels sampled uniformly at random
from the train-split vocabulary
([MedRAP#92](https://github.com/McDermottHealthAI/MedRAP/pull/92) --
no positive-rate/count filtering), across N ∈ {1, 2, 4, 8, 16, 32, 64,
128} simultaneous binary tasks:

- **`patient_only`** -- no retrieval, predicts directly from the
  patient's own EHR sequence.
- **`marginalized` (binary mode)** -- retrieval-augmented, predicts per
  retrieved document and marginalizes independently per task
  (`marginalized_output_mode=binary`,
  [MedRAP PR #93](https://github.com/McDermottHealthAI/MedRAP/pull/93)).

A sampled code can turn out rare or degenerate (single-class) on a given
split, which is why AUROC is undefined for several N here -- see
Results below. See
[`../mimic_iv_sweep_frequent/`](../mimic_iv_sweep_frequent/) for the
same comparison with task codes selected by frequency instead, which
has usable labels at every N.

### Architecture & hyperparameters

Both arms share: RoPE patient encoder, 3 training epochs, `lr=1e-3`,
`warmup_steps=200`, batch size 32, `max_seq_len=256`,
`seq_sampling_strategy=to_end`, `gradient_clip_val=1.0`.

| Stage | `patient_only` | `marginalized` (binary) |
| --- | --- | --- |
| Encoder | `TimeDeltaRoPEPatientEncoder` -- vocab 65536, embed dim 128, 4 heads, 2 layers, ff dim 256 | same |
| Query projector | unused (`fusion=passthrough` discards it) | `SequenceMeanQueryProjector` -- in 128, out 1024 |
| Retriever | none | `hf_dataset` (FAISS) -- `k=4`, corpus = `MedRAG/textbooks` |
| Retrieval encoder | none | `TokenFeatureRetrievalEncoder` -- vocab 151936, embed dim 64 |
| Fusion | `PassthroughFusion` (no retrieval) | `PerDocCrossAttentionFusion` -- d_model 256, 8 heads, ff dim 512, 2 layers, dropout 0.1 |
| Head | `LinearHead`, in 128, out N | `LinearHead`, in 256, out N |
| Loss | `MultiTaskBCELoss` | `MultiTaskBCEMarginalizedLoss` (marginalizes per-task sigmoid over the 4 retrieved docs) |

## Results: patient_only vs. marginalized (binary mode) -- random-task labels

Same comparison as
[`mimic_iv_sweep_frequent`](../mimic_iv_sweep_frequent/README.md#results),
but on this directory's random-task labels (uniform sampling from the
train-split vocabulary, no frequency bias) instead of most-frequent-code
selection.

### Run instructions

```bash
cd mimic_iv_sweep
uv sync
sbatch scripts/prepare_retrieval.sh          # once, if data/retrieval_db doesn't exist yet
sbatch scripts/generate_labels_n1248.sh       # task labels, N=1..128
sbatch scripts/sweep_patient_only_n1248.sh    # patient_only, N=1..128
sbatch scripts/sweep_marginalized_binary_n1248.sh  # marginalized (binary), N=1..128
```

Results land in W&B (`medrap` project) as `patient-only-n{N}-*` and
`marginalized-binary-n{N}-*`. Architecture and hyperparameters are
identical to
[`mimic_iv_sweep_frequent`'s](../mimic_iv_sweep_frequent/README.md#architecture--hyperparameters)
-- only the task-code selection differs (`code_selection=random`, the
default, vs. `most_frequent`).

### Average AUROC

**What "valid task" means:** a sampled task counts as valid only if its
held-out validation split contains at least one positive *and* one
negative example -- AUROC is undefined otherwise, so those tasks are
excluded from the mean rather than scored as 0.

**Update:** the `0/1`, `0/2`, `0/4`, `0/8` valid-task counts previously
here were a real bug, not an inherent property of random sampling --
`code_selection=random` had no filtering at all
([MedRAP#89](https://github.com/McDermottHealthAI/MedRAP/pull/89) had
removed positive-rate/count filtering by design), so a sampled code could
come up degenerate (single-class) on a split with no recourse. Two fixes
landed in [MedRAP#98](https://github.com/McDermottHealthAI/MedRAP/pull/98):

1. For `duration_distribution="fixed"` (this sweep), task-code selection
   now rejects any code that comes up degenerate in any generated split
   and redraws a replacement, repeating until `num_tasks` valid codes are
   found.
2. That rejection logic surfaced a second, pre-existing bug: prediction-time
   anchor sampling wasn't actually deterministic given a fixed seed --
   `_sample_prediction_anchors`'s internal `group_by("subject_id")` output
   row order isn't guaranteed stable across calls (polars can use a
   multi-threaded hash aggregation), so the same RNG draw could land on a
   different subject each call, silently reshuffling everyone's
   `prediction_time` between calls even with an identical seed. Fixed by
   sorting by `subject_id` before consuming the RNG draw.

All labels below were regenerated with both fixes; every N from 1 to 128 is
now **100% valid** (was 0/1, 0/2, 0/4, 0/8 at low N, and only partially
valid at N=16-128, before the fixes).

| N | patient_only | marginalized (binary) | Δ (binary − patient_only) | valid tasks |
| --- | --- | --- | --- | --- |
| 1 | [0.9998](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/6pxuykxx) | [0.9998](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/wf2jvams) | -0.0001 | 1/1 |
| 2 | [0.9898](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/uy6ok7np) | [0.9910](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/16f16n21) | +0.0012 | 2/2 |
| 4 | [0.9899](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/v7i92udt) | [0.9966](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/8ebg51m3) | +0.0067 | 4/4 |
| 8 | [0.9905](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/jdvghvkd) | [0.9868](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/dkzecbnp) | -0.0036 | 8/8 |
| 16 | [0.9951](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/w1gsrgs9) | [0.9960](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/xlv8h390) | +0.0009 | 16/16 |
| 32 | [0.9908](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/3ed0ulyz) | [0.9908](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/3yv3gu3t) | +0.0000 | 32/32 |
| 64 | [0.9837](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/6cvr6ayc) | [0.9882](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/osiopk1q) | +0.0045 | 64/64 |
| 128 | [0.9913](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/ormprpv8) | [0.9899](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/wv3runb8) | -0.0014 | 128/128 |

With every task now usable, the gap between architectures is small and
inconsistent in sign (-0.0036 to +0.0067) -- no clear win for retrieval at
this scale/epoch budget on random-task labels. See the [duration x variance
study](#duration-x-variance-study-n25-patient_only-vs-marginalized-binary-at-7-day-and-30-day-durations-5-random-draws-each)
below for whether this holds up across repeated random draws.

### Per-task win rate

Of the (now all-valid) tasks at each N, the fraction where `marginalized
(binary)` beat `patient_only` on that same task -- pending a re-run of the
per-task win-rate script against the regenerated labels above; the prior
table here was computed against the old, mostly-degenerate label set and
has been removed rather than left stale.

### Task codes used

The exact `num_tasks` codes selected by `code_selection=random` at each N
(from `data/tasks/n<N>/tasks/code_index.json`; same codes for both
architectures at a given N). Regenerated with
[MedRAP#98](https://github.com/McDermottHealthAI/MedRAP/pull/98)'s fixes,
so these differ from the codes previously listed here. Unlike the
frequent-code selection, each N's codes are an independent random draw --
no nesting across N:

<details><summary><code>N=1</code> (1 code)</summary>

```
0: LAB//51274//sec//value_[28.1,inf)
```

</details>

<details><summary><code>N=2</code> (2 codes)</summary>

```
0: MEDICATION//START//Ampicillin-Sulbactam
1: LAB//228305//UNK//value_[1.0,inf)
```

</details>

<details><summary><code>N=4</code> (4 codes)</summary>

```
0: LAB//225091//UNK//value_[1.0,inf)
1: LAB//50804//mEq/L//value_[21.0,23.0)
2: LAB//225978//UNK
3: LAB//220181//mmHg//value_[61.0,66.0)
```

</details>

<details><summary><code>N=8</code> (8 codes)</summary>

```
0: MEDICATION//START//OxycoDONE (Immediate Release) 
1: MEDICATION//START//Dextrose 50%
2: LAB//229109//UNK
3: LAB//224016//UNK
4: LAB//224373//UNK
5: LAB//51274//sec//value_[21.6,28.1)
6: LAB//228099//UNK//value_[0.0,inf)
7: LAB//51279//m/uL//value_[3.68,3.92)
```

</details>

<details><summary><code>N=16</code> (16 codes)</summary>

```
0: LAB//51092//UNK
1: MEDICATION//STOP//Lidocaine 1% (For PICC/Midline Insertions)
2: LAB//224848//UNK
3: LAB//50862//g/dL//value_[4.4,4.6)
4: LAB//227368//UNK//value_[1.0,inf)
5: LAB//52073//K/uL//value_[0.0,0.01)
6: LAB//52281//%//value_[0.0,2.0)
7: LAB//51279//m/uL//value_[3.44,3.68)
8: LAB//50993//uIU/mL//value_[0.6,0.95)
9: LAB//229088//UNK
10: LAB//52172//fL//value_[57.1,63.2)
11: LAB//220545//%//value_[30.0,31.8)
12: LAB//223770//%//value_[92.0,inf)
13: LAB//50975//UNK
14: LAB//50934//UNK//value_[40.0,inf)
15: LAB//50885//mg/dL//value_[0.3,0.4)
```

</details>

<details><summary><code>N=32</code> (32 codes)</summary>

```
0: LAB//223782//UNK
1: MEDICATION//STOP//MetRONIDAZOLE (FLagyl)
2: LAB//224862//UNK
3: LAB//50861//IU/L//value_[24.0,29.0)
4: LAB//51214//mg/dL//value_[479.0,603.0)
5: DIAGNOSIS//ICD//10//N179
6: LAB//51233//UNK
7: LAB//225625//mg/dL//value_[8.4,8.6)
8: LAB//223947//UNK
9: LAB//223835//UNK//value_[50.0,60.0)
10: LAB//51243//N/A
11: LAB//50903//Ratio//value_[3.7,4.0)
12: LAB//50911//ng/mL//value_[3.0,4.0)
13: LAB//51006//mg/dL//value_[13.0,16.0)
14: LAB//50919//UNK
15: LAB//220545//%//value_[30.0,31.8)
16: LAB//50983//mEq/L//value_[133.0,135.0)
17: LAB//50954//IU/L//value_[232.0,261.0)
18: LAB//51508//UNK
19: MEDICATION//HYDROmorphone (Dilaudid)//Administered
20: LAB//51246//UNK
21: LAB//50983//mEq/L//value_[141.0,142.0)
22: LAB//51146//%//value_[0.2,0.3)
23: LAB//51133//K/uL//value_[1.09,1.31)
24: LAB//224007//UNK
25: LAB//220210//insp/min//value_[-inf,13.0)
26: LAB//50878//IU/L//value_[31.0,39.0)
27: LAB//224076//UNK
28: LAB//51254//%//value_[12.2,inf)
29: LAB//50863//IU/L//value_[-inf,54.0)
30: LAB//223761//°F//value_[98.9,99.2)
31: LAB//223770//%//value_[90.0,92.0)
```

</details>

<details><summary><code>N=64</code> (64 codes)</summary>

```
0: LAB//51279//m/uL//value_[3.68,3.92)
1: LAB//220292//L/min//value_[6.0,7.5)
2: LAB//50931//mg/dL//value_[97.0,103.0)
3: LAB//51487//UNK
4: MEDICATION//START//Nicotine Patch
5: LAB//51244//%//value_[17.4,21.0)
6: LAB//52135//%//value_[0.5,0.6)
7: LAB//51519//UNK
8: LAB//228516//UNK
9: LAB//52111//UNK
10: TRANSFER_TO//ED//Emergency Department
11: LAB//229685//UNK//value_[1.0,inf)
12: LAB//50970//mg/dL//value_[3.2,3.4)
13: LAB//50820//units//value_[7.44,7.47)
14: LAB//224059//UNK//value_[2.0,3.0)
15: LAB//51279//m/uL//value_[4.16,4.41)
16: DIAGNOSIS//ICD//10//J189
17: LAB//51484//mg/dL//value_[10.0,15.0)
18: LAB//224862//UNK
19: LAB//50931//mg/dL//value_[91.0,97.0)
20: MEDICATION//STOP//Milk of Magnesia
21: LAB//52073//K/uL//value_[0.06,0.09)
22: LAB//50902//mEq/L//value_[98.0,100.0)
23: LAB//226732//UNK
24: LAB//51075//N/A
25: LAB//224641//UNK//value_[1.0,inf)
26: LAB//224773//UNK
27: LAB//51248//pg//value_[32.9,inf)
28: MEDICATION//Senna//Not Given
29: LAB//50993//uIU/mL//value_[1.2,1.5)
30: LAB//50893//mg/dL//value_[9.4,9.7)
31: LAB//50861//IU/L//value_[14.0,17.0)
32: MEDICATION//CeFAZolin//Administered
33: LAB//51498// //value_[1.018,1.02)
34: ED_REGISTRATION
35: DIAGNOSIS//ICD//10//Y92230
36: LAB//50813//mmol/L//value_[1.7,2.0)
37: LAB//227345//UNK//value_[0.0,10.0)
38: LAB//50868//mEq/L//value_[15.0,16.0)
39: LAB//50912//mg/dL//value_[1.1,1.2)
40: LAB//229381//UNK//value_[1.0,inf)
41: MEDICATION//Docusate Sodium//Not Given
42: LAB//50889//mg/L//value_[4.3,7.1)
43: LAB//52073//K/uL//value_[0.33,inf)
44: LAB//224054//UNK//value_[3.0,4.0)
45: MEDICATION//STOP//Polyethylene Glycol
46: MEDICATION//Aspirin//Administered
47: HCPCS//Hospital observation services
48: MEDICATION//START//CefePIME
49: LAB//51221//%//value_[-inf,24.4)
50: LAB//51248//pg//value_[30.0,30.5)
51: MEDICATION//START//Ketorolac
52: LAB//51222//g/dL//value_[14.0,inf)
53: LAB//50971//mEq/L//value_[4.9,inf)
54: LAB//52069//K/uL//value_[0.03,0.04)
55: LAB//228400//UNK
56: MEDICATION//Acetaminophen//Not Given
57: LAB//227240//UNK
58: LAB//223943//UNK
59: LAB//50983//mEq/L//value_[143.0,inf)
60: LAB//50934//UNK//value_[-inf,1.0)
61: MEDICATION//STOP//Bisacodyl
62: LAB//51463//UNK
63: MEDICATION//STOP//Enoxaparin (Prophylaxis)
```

</details>

<details><summary><code>N=128</code> (128 codes)</summary>

```
0: LAB//50868//mEq/L//value_[12.0,13.0)
1: LAB//228307//UNK
2: LAB//50943//N/A
3: MEDICATION//Acetaminophen//Not Given
4: LAB//224415//UNK
5: LAB//224879//UNK
6: LAB//227341//UNK//value_[25.0,inf)
7: LAB//52075//K/uL//value_[3.81,4.48)
8: LAB//224058//UNK//value_[2.0,3.0)
9: MEDICATION//START//Senna
10: LAB//50910//IU/L//value_[34.0,49.0)
11: LAB//220181//mmHg//value_[74.0,78.0)
12: MEDICATION//Pantoprazole//Administered
13: LAB//51301//K/uL//value_[14.5,inf)
14: LAB//220277//%//value_[-inf,93.0)
15: LAB//51501//#/hpf//value_[1.0,2.0)
16: LAB//51301//K/uL//value_[6.0,6.8)
17: LAB//51249//g/dL//value_[33.0,33.5)
18: LAB//50993//uIU/mL//value_[-inf,0.6)
19: MEDICATION//Multivitamins//Administered
20: LAB//51478//mg/dL
21: LAB//51244//%//value_[6.6,10.3)
22: LAB//50934//UNK//value_[1.0,2.0)
23: LAB//50882//mEq/L//value_[-inf,20.0)
24: LAB//51279//m/uL//value_[2.92,3.18)
25: MEDICATION//STOP//UNK
26: LAB//51989//N/A
27: LAB//51678//UNK//value_[7.0,8.0)
28: MEDICATION//STOP//Levothyroxine Sodium
29: LAB//50905//mg/dL//value_[132.0,151.0)
30: LAB//52172//fL//value_[45.1,46.9)
31: LAB//50863//IU/L//value_[-inf,54.0)
32: LAB//51279//m/uL//value_[3.18,3.44)
33: LAB//225090//UNK
34: LAB//224689//insp/min//value_[0.0,10.0)
35: LAB//51274//sec//value_[14.1,15.3)
36: MEDICATION//STOP//LORazepam
37: LAB//220001//UNK
38: LAB//51069//mg/dL//value_[59.8,inf)
39: LAB//51274//sec//value_[13.2,14.1)
40: PROCEDURE//ICD//9//8744
41: LAB//50883//mg/dL//value_[0.1,0.2)
42: LAB//220179//mmHg//value_[130.0,138.0)
43: LAB//51250//fL//value_[100.0,inf)
44: LAB//229323//UNK//value_[0.0,2.0)
45: DIAGNOSIS//ICD//10//N179
46: LAB//50934//UNK//value_[5.0,6.0)
47: LAB//51221//%//value_[42.1,inf)
48: LAB//50903//Ratio//value_[2.6,2.8)
49: MEDICATION//START//Famotidine
50: LAB//228559//UNK
51: LAB//50920//UNK
52: LAB//51498// //value_[1.015,1.018)
53: LAB//51265//K/uL//value_[243.0,272.0)
54: LAB//50993//uIU/mL//value_[2.7,3.4)
55: LAB//52073//K/uL//value_[0.12,0.16)
56: LAB//224162//insp/min//value_[8.0,inf)
57: LAB//51214//mg/dL//value_[479.0,603.0)
58: MEDICATION//Lisinopril//Administered
59: LAB//229089//UNK
60: LAB//50900//ng/mL//value_[22.2,79.1)
61: MEDICATION//STOP//Morphine Sulfate
62: LAB//227467//UNK//value_[1.2,1.3)
63: LAB//220180//mmHg//value_[47.0,53.0)
64: LAB//50893//mg/dL//value_[8.6,8.8)
65: LAB//50863//IU/L//value_[54.0,64.0)
66: LAB//224785//UNK
67: LAB//51301//K/uL//value_[11.4,14.5)
68: LAB//227809//UNK//value_[1.0,inf)
69: LAB//223834//L/min//value_[2.0,3.0)
70: INFUSION_START//225943//value_[22.47191,28.92)
71: LAB//50821//mm Hg//value_[75.0,87.0)
72: LAB//50912//mg/dL//value_[2.3,inf)
73: LAB//51260//UNK
74: LAB//223769//%//value_[100.0,inf)
75: LAB//51277//%//value_[18.7,inf)
76: LAB//220277//%//value_[97.0,98.0)
77: LAB//220179//mmHg//value_[93.0,100.0)
78: LAB//50993//uIU/mL//value_[1.2,1.5)
79: MEDICATION//STOP//Calcium Carbonate
80: LAB//50868//mEq/L//value_[14.0,15.0)
81: MEDICATION//Dextrose 50%//Administered
82: MEDICATION//UNK//Stopped
83: LAB//228594//UNK
84: MEDICATION//START//Dextrose 50%
85: MEDICATION//STOP//Venlafaxine
86: MEDICATION//START//Multivitamins
87: LAB//50868//mEq/L//value_[18.0,inf)
88: LAB//51200//%//value_[1.0,1.4)
89: LAB//224024//UNK
90: LAB//51221//%//value_[-inf,24.4)
91: LAB//51476//#/hpf//value_[3.0,6.0)
92: LAB//50934//UNK//value_[6.0,8.0)
93: LAB//51254//%//value_[4.2,5.1)
94: LAB//51493//#/hpf//value_[1.0,2.0)
95: LAB//51006//mg/dL//value_[11.0,13.0)
96: MEDICATION//STOP//Bisacodyl
97: LAB//50933//UNK
98: MEDICATION//START//Oxytocin
99: LAB//50976//g/dL//value_[6.8,7.0)
100: LAB//224692//UNK
101: LAB//229347//UNK
102: LAB//227961//UNK
103: LAB//50878//IU/L//value_[18.0,20.0)
104: MEDICATION//START//Heparin Flush (10 units/ml)
105: LAB//51265//K/uL//value_[272.0,310.0)
106: LAB//220179//mmHg//value_[150.0,inf)
107: MEDICATION//STOP//Vitamin D
108: MEDICATION//STOP//Famotidine
109: LAB//50912//mg/dL//value_[0.7,0.8)
110: LAB//227957//UNK
111: LAB//223784//UNK
112: LAB//51006//mg/dL//value_[9.0,11.0)
113: LAB//224073//UNK
114: LAB//229108//UNK
115: LAB//220046//bpm//value_[120.0,130.0)
116: LAB//50903//Ratio//value_[3.1,3.4)
117: LAB//224072//UNK
118: LAB//225092//UNK//value_[1.0,inf)
119: LAB//220739//UNK//value_[4.0,inf)
120: MEDICATION//Gabapentin//Administered
121: LAB//51254//%//value_[5.1,5.9)
122: LAB//50983//mEq/L//value_[140.0,141.0)
123: LAB//50970//mg/dL//value_[3.2,3.4)
124: MEDICATION//Lidocaine 5% Patch//Removed
125: MEDICATION//CeFAZolin//Administered
126: LAB//220277//%//value_[95.0,96.0)
127: LAB//51082//mg/dL//value_[102.0,122.0)
```

</details>


## Random task codes + random per-task durations (`generate_labels_duration_n1248.sh` + `sweep_marginalized_binary_duration_n1248.sh`)

Combines this directory's random task-code selection
(`code_selection=random`, the `medrap-preprocess` default) with
[`mimic_iv_sweep_frequent`'s](../mimic_iv_sweep_frequent/README.md#random-per-task-occurrence-window-durations-generate_labels_duration_n1248_frequentsh--sweep_marginalized_binary_duration_n1248sh)
random-duration idea (`duration_distribution=log-uniform`,
[MedRAP PR #96](https://github.com/McDermottHealthAI/MedRAP/pull/96)) --
the "random tasks, random durations" combination neither directory ran on
its own before this. Same architecture/hyperparameters as
`sweep_marginalized_binary_n1248.sh` above; only the label file differs.

```bash
sbatch scripts/generate_labels_duration_n1248.sh       # task labels, N=1..128
sbatch scripts/sweep_marginalized_binary_duration_n1248.sh  # training, N=1..128
```

Results land in W&B as `marginalized-binary-duration-n{N}-*`; checkpoints
at `outputs/marginalized_binary_duration_n1248/n{N}/`; labels at
`data/tasks_duration/n{N}/tasks/` (separate from `data/tasks/n{N}/`, so
this never overwrites the fixed-duration random-task labels every other
script in this directory reads).

**Caveat (written before [MedRAP#98](https://github.com/McDermottHealthAI/MedRAP/pull/98)):**
random-code-selection labels used to be extremely sparse at low N on this
directory's *fixed*-duration labels (`n_valid_tasks=0` for N=1/2). That was
a real bug in task-code selection, now fixed for `duration_distribution=
"fixed"` -- see the [base sweep results](#results-patient_only-vs-marginalized-binary-mode----random-task-labels)
above. This section's `duration_distribution="log-uniform"` combination is
**not** covered by that fix (per-task random durations mean prediction
anchors would shift between rejection-sampling candidate draws, so validity
can't be checked consistently -- see MedRAP#98's PR description), so it may
still produce degenerate labels at low N. Run `check_task_balance.py`
against `data/tasks_duration/n<N>/tasks` after label generation to check
before spending GPU time training on a given N. The [duration x variance
study](#duration-x-variance-study-n25-patient_only-vs-marginalized-binary-at-7-day-and-30-day-durations-5-random-draws-each)
below sidesteps this by using `duration_distribution="fixed"` at two fixed
horizons (7d, 30d) instead of per-task random durations.

*(This combination (random codes + random per-task durations) has not been
run to completion -- superseded in practice by the duration x variance
study below, which answers a similar question with a supported, bug-free
configuration.)*

## Duration x variance study (N=25): patient_only vs. marginalized(binary) at 7-day and 30-day durations, 5 random draws each (`generate_labels_duration_variance_n25.sh` + `sweep_patient_only_duration_variance_n25.sh` + `sweep_marginalized_binary_duration_variance_n25.sh`)

Advisor's ask: check whether the `patient_only` vs. `marginalized(binary)`
comparison holds up at both a short and a long fixed occurrence window, and
across different random samples of tasks rather than a single draw. Fixed
N=25 (the original sweep's default task count), 5 independent random draws
of 25 task codes each (`code_selection=random`, seeds `101, 202, 303, 404,
505` -- same draw seeds as the (unrun) variance study below, so a given
draw number picks the same codes there too), at two fixed occurrence
windows: 7 days and 30 days. `patient_only` and `marginalized(binary)`
always train on the *identical* codes/labels within a (duration, draw) pair
(a paired comparison). Uses `duration_distribution="fixed"`, so
[MedRAP#98](https://github.com/McDermottHealthAI/MedRAP/pull/98)'s
degenerate-code rejection applies -- all 10 label sets came out 25/25 valid,
confirmed via `check_task_balance.py` before training.

```bash
sbatch scripts/generate_labels_duration_variance_n25.sh          # labels: 2 durations x 5 draws = 10 jobs, CPU
sbatch scripts/sweep_patient_only_duration_variance_n25.sh        # training: 10 jobs, GPU
sbatch scripts/sweep_marginalized_binary_duration_variance_n25.sh # training: 10 jobs, GPU
```

Labels land at `data/tasks_duration_variance/d{7,30}/draw{1..5}/tasks/`.
Checkpoints at `outputs/patient_only_duration_variance_n25/d{7,30}/draw{d}/`
and `outputs/marginalized_binary_duration_variance_n25/d{7,30}/draw{d}/`.
W&B run names: `patient-only-duration-variance-d{7,30}-draw{d}-*` and
`marginalized-binary-duration-variance-d{7,30}-draw{d}-*`.

### Results

All 20 training runs finished; all 10 label sets were 25/25 valid (no
degenerate tasks).

| Duration | Draw | patient_only | marginalized (binary) | Δ (binary − patient_only) |
| --- | --- | --- | --- | --- |
| 7d | 1 | [0.9716](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/i7st70mr) | [0.9511](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/wvh4bed6) | -0.0204 |
| 7d | 2 | [0.9923](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/yhhmda1x) | [0.9899](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/6fsm6q44) | -0.0024 |
| 7d | 3 | [0.9849](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/td78ypsm) | [0.9850](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/q2yt4g2b) | +0.0001 |
| 7d | 4 | [0.9861](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/v2mt0zo0) | [0.9863](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/patoedk6) | +0.0002 |
| 7d | 5 | [0.7867](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/xizo5b26) | [0.7816](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/7jfqv5ij) | -0.0051 |
| 30d | 1 | [0.9317](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/ldizdna4) | [0.9057](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/0p1mpdah) | -0.0260 |
| 30d | 2 | [0.9666](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/36zhroe5) | [0.9528](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/w2i3zzlj) | -0.0138 |
| 30d | 3 | [0.9054](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/6wjvzje8) | [0.9007](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/jsox2u8t) | -0.0048 |
| 30d | 4 | [0.9831](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/7dvnefxi) | [0.9880](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/pc668dq8) | +0.0049 |
| 30d | 5 | [0.9277](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/demfunrj) | [0.9363](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/v3tmxob6) | +0.0087 |

**Δ mean ± std across the 5 draws:**

| Duration | mean Δ | std Δ |
| --- | --- | --- |
| 7d | -0.0055 | 0.0086 |
| 30d | -0.0062 | 0.0141 |

**Takeaway:** across both durations, `marginalized(binary)` does not show a
consistent AUROC improvement over `patient_only` -- the mean Δ is slightly
negative at both 7d and 30d, and in both cases the std across draws (0.009,
0.014) is larger than the mean effect, so the sign of the win/loss is
essentially noise from draw to draw (e.g. draw 1 loses by ~0.02-0.026 at
both durations, while draws 3-5 are close to a wash or a small win). No
systematic retrieval benefit emerges at N=25 with this architecture/epoch
budget, at either horizon length.

## Anchor-sampling fix rerun (`generate_labels_duration_variance_n25.sh` + `sweep_patient_only_duration_variance_n25.sh` + `sweep_marginalized_binary_duration_variance_n25.sh`, MedRAP#99)

Advisor feedback: the original anchor-sampling logic (`_sample_prediction_anchors`)
drew the prediction-time anchor **uniformly from continuous calendar time**
within `[first_event+min_history_days, last_event-horizon_days]`. Clinical
data is bursty -- dense during an encounter, then long silent gaps between
visits -- so a continuous draw frequently landed in one of those gaps: a
timestamp where nothing was recorded, not a real clinical moment to predict
from. [MedRAP#99](https://github.com/McDermottHealthAI/MedRAP/pull/99) fixes
this by porting the core idea from
[EveryQuery](https://github.com/payalchandak/EveryQuery)'s
`build_prediction_times`: the anchor is now drawn as a **uniform random
index over the subject's own real event timestamps** within that same
window, instead of a continuous offset -- guaranteeing every anchor
coincides with an actual observation. Same `min_history_days`/`horizon_days`
config surface, same function signature -- only what's sampled from within
the window changed.

This reruns the exact same duration x variance study above (N=25, k=4,
3 epochs, 5 draws, 7d/30d) with the fixed anchor sampling, on the *same*
seeds/draws, so old vs. new is an apples-to-apples comparison of the anchor
mechanism alone.

```bash
sbatch scripts/generate_labels_duration_variance_n25.sh   # relabel with the fixed anchor sampling
sbatch scripts/sweep_patient_only_duration_variance_n25.sh
sbatch scripts/sweep_marginalized_binary_duration_variance_n25.sh
```

All 10 relabeled sets confirmed 100% valid (no degenerate tasks) via
`check_task_balance.py`; all 20 training runs finished cleanly.

### Results: old (continuous-time) vs. new (real-event) anchors

| Duration | Draw | patient_only old | patient_only new | marginalized old | marginalized new |
| --- | --- | --- | --- | --- | --- |
| 7d | 1 | 0.9716 | [0.9023](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/834gfe16) | 0.9511 | [0.9033](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/mav9ruay) |
| 7d | 2 | 0.9923 | [0.9176](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/3rv35c08) | 0.9899 | [0.9131](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/e2nsb4oa) |
| 7d | 3 | 0.9849 | [0.8958](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/to3plbt9) | 0.9850 | [0.8891](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/cft7ey7q) |
| 7d | 4 | 0.9861 | [0.8145](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/dwaxl3mo) | 0.9863 | [0.8049](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/7j8mlxqr) |
| 7d | 5 | 0.7867 | [0.8563](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/vw6i91ss) | 0.7816 | [0.8637](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/0ea83wxa) |
| 30d | 1 | 0.9317 | [0.8741](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/2c1r7wqv) | 0.9057 | [0.8739](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/fs9c7ge7) |
| 30d | 2 | 0.9666 | [0.9091](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/w6sa33zf) | 0.9528 | [0.9026](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/v61q9hbg) |
| 30d | 3 | 0.9054 | [0.8678](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/era18qcy) | 0.9007 | [0.8629](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/d8h23u95) |
| 30d | 4 | 0.9831 | [0.9117](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/aizud74k) | 0.9880 | [0.9094](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/kyismvfr) |
| 30d | 5 | 0.9277 | [0.9118](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/cdjd4prh) | 0.9363 | [0.9079](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/vpjgpdwx) |

**Summary statistics, old vs. new:**

| Duration | Metric | Old (continuous) | New (real-event) |
| --- | --- | --- | --- |
| 7d | patient_only mean AUROC | 0.9443 | 0.8773 |
| 7d | patient_only std across draws | 0.0884 | 0.0418 |
| 7d | marginalized mean AUROC | 0.9388 | 0.8748 |
| 7d | marginalized std across draws | 0.0893 | 0.0433 |
| 7d | Δ (marginalized − patient_only) mean | -0.0055 | -0.0025 |
| 7d | Δ std | 0.0086 | 0.0068 |
| 30d | patient_only mean AUROC | 0.9429 | 0.8949 |
| 30d | patient_only std across draws | 0.0314 | 0.0220 |
| 30d | marginalized mean AUROC | 0.9367 | 0.8913 |
| 30d | marginalized std across draws | 0.0359 | 0.0214 |
| 30d | Δ (marginalized − patient_only) mean | -0.0062 | -0.0036 |
| 30d | Δ std | 0.0141 | 0.0024 |

**Takeaway:** two clear effects from the anchor fix, both in the direction
expected from a genuine bug fix rather than a regression:

1. **Absolute AUROC dropped for both architectures roughly equally**
   (patient_only: -0.067 at 7d, -0.048 at 30d; marginalized: -0.064 at 7d,
   -0.045 at 30d). This is expected, not concerning: continuous-time anchors
   could land in a "dead" gap where the task was structurally easier (little
   real signal to confuse the model, or accidental leakage-like artifacts
   from an anchor's relationship to nearby recorded events); real-event
   anchors put every prediction at a genuine clinical decision point, which
   is intrinsically harder -- a more honest difficulty level, not a broken
   model.
2. **Draw-to-draw variance shrank substantially** -- patient_only's std
   across the 5 draws roughly halved at 7d (0.088 → 0.042) and dropped by
   ~30% at 30d (0.031 → 0.022); marginalized shows the same pattern. Fixing
   the "sometimes the anchor is meaningless" problem makes results far more
   reproducible draw-to-draw, which directly addresses the [earlier finding](#summary-across-k--4-32-64-128)
   that draw-to-draw variance was the dominant source of noise in this whole
   study.

**The core finding is unchanged**: `marginalized(binary)` still shows no
consistent AUROC benefit over `patient_only` -- the architecture Δ stays
small and slightly negative at both durations (7d: -0.0055 → -0.0025; 30d:
-0.0062 → -0.0036), well within noise either way. The anchor fix makes the
underlying task harder and the measurements more stable, but does not
change the substantive conclusion about retrieval.

## Zach's anchor refinement (MedRAP#100): excluding TIMELINE tokens from anchor candidates (`generate_labels_zach_uniform_event_n25_{7,30}d.sh` + `sweep_patient_only_zach_uniform_event_n25_{7,30}d.sh` + `sweep_marginalized_binary_zach_uniform_event_n25_{7,30}d.sh`)

Teammate Zach opened [MedRAP#100](https://github.com/McDermottHealthAI/MedRAP/pull/100)
with a design that overlaps with #99 above (also draws anchors from real
event timestamps rather than continuous calendar time) but adds one thing
#99 was missing: a `_clinical_events()` filter that excludes `TIMELINE//`
boundary-marker tokens (and `meds.birth_code`) from the **anchor candidate
pool itself**, not just from task-code eligibility. #99's anchor sampling
could still land exactly on a synthetic `TIMELINE//` timestamp -- a
structural marker, not something a clinician actually observed -- which
partially undermines the "anchor at a real clinical moment" guarantee. #100
was opened against a stale `main` (pre-#99) and doesn't carry
`code_selection`/`duration_distribution`/degenerate-code rejection (#98) or
the eval config (#95/#97), so this experiment integrates Zach's
`_clinical_events()` idea into the current stack as an `anchor_strategy`
switch (`"uniform_event"`, now the new default, vs. `"uniform_lifetime"` to
recover the pre-#99 behavior) rather than running #100 as-is --
see `MedRAP@experiment/zach-uniform-event-plus-stack`.

This reruns the same N=25/5-draw/7d+30d study as the "Anchor-sampling fix
rerun" above, same seeds (101/202/303/404/505), with only the anchor
candidate pool changed (#99's real-event anchors → Zach's TIMELINE-excluded
real-event anchors) -- so "new" in the table below is #99 and "newer" is
#100/Zach.

```bash
sbatch scripts/generate_labels_zach_uniform_event_n25_7d.sh
sbatch scripts/generate_labels_zach_uniform_event_n25_30d.sh
sbatch scripts/sweep_patient_only_zach_uniform_event_n25_7d.sh
sbatch scripts/sweep_marginalized_binary_zach_uniform_event_n25_7d.sh
sbatch scripts/sweep_patient_only_zach_uniform_event_n25_30d.sh
sbatch scripts/sweep_marginalized_binary_zach_uniform_event_n25_30d.sh
```

All 10 relabeled sets confirmed 100% valid (no degenerate tasks) via
`check_task_balance.py`; all 20 training runs finished cleanly.

### Results: #99 (plain real-event) vs. #100/Zach (TIMELINE-excluded real-event) anchors

| Duration | Draw | patient_only #99 | patient_only Zach | marginalized #99 | marginalized Zach |
| --- | --- | --- | --- | --- | --- |
| 7d | 1 | 0.9023 | [0.8609](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/8b0lbgtn) | 0.9033 | [0.8504](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/dhznne0g) |
| 7d | 2 | 0.9176 | [0.8893](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/yvj45z05) | 0.9131 | [0.8875](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/f5qf9z7s) |
| 7d | 3 | 0.8958 | [0.9219](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/uuzhmnqu) | 0.8891 | [0.8932](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/8zc3rt76) |
| 7d | 4 | 0.8145 | [0.9371](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/95819r6h) | 0.8049 | [0.9304](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/wvrpg223) |
| 7d | 5 | 0.8563 | [0.8894](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/3qp3172h) | 0.8637 | [0.8726](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/wly3t44v) |
| 30d | 1 | 0.8741 | [0.8923](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/paagyvs5) | 0.8739 | [0.9050](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/zv6znbgy) |
| 30d | 2 | 0.9091 | [0.9095](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/llsm3nrh) | 0.9026 | [0.8807](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/z0mkqk7x) |
| 30d | 3 | 0.8678 | [0.9030](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/67lvzdwm) | 0.8629 | [0.8937](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/yf58w6k6) |
| 30d | 4 | 0.9117 | [0.9446](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/s99xd227) | 0.9094 | [0.9434](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/s7wcvz6r) |
| 30d | 5 | 0.9118 | [0.8691](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/zcmvtx1m) | 0.9079 | [0.8543](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/zlfwb75k) |

**Summary statistics, #99 vs. #100/Zach:**

| Duration | Metric | #99 (plain real-event) | #100/Zach (TIMELINE-excluded) |
| --- | --- | --- | --- |
| 7d | patient_only mean AUROC | 0.8773 | 0.8997 |
| 7d | patient_only std across draws | 0.0418 | 0.0301 |
| 7d | marginalized mean AUROC | 0.8748 | 0.8868 |
| 7d | marginalized std across draws | 0.0433 | 0.0295 |
| 7d | Δ (marginalized − patient_only) mean | -0.0025 | -0.0129 |
| 7d | Δ std | 0.0068 | 0.0104 |
| 30d | patient_only mean AUROC | 0.8949 | 0.9037 |
| 30d | patient_only std across draws | 0.0220 | 0.0275 |
| 30d | marginalized mean AUROC | 0.8913 | 0.8954 |
| 30d | marginalized std across draws | 0.0214 | 0.0328 |
| 30d | Δ (marginalized − patient_only) mean | -0.0036 | -0.0083 |
| 30d | Δ std | 0.0024 | 0.0154 |

**Takeaway:**

1. **Zach's refinement modestly increases mean AUROC for both
   architectures at both durations** (patient_only: +0.022 at 7d, +0.009 at
   30d; marginalized: +0.012 at 7d, +0.004 at 30d). This is the expected
   direction: excluding `TIMELINE//` boundary markers from the anchor pool
   removes a small number of anchors that weren't real clinical moments,
   nudging every prediction toward a genuine observation -- consistent with
   #99's fix but slightly more complete.
2. **Draw-to-draw variance shrinks at 7d but does not shrink at 30d** --
   patient_only's std drops from 0.042 to 0.030 at 7d, but *increases*
   slightly at 30d (0.022 → 0.028, and marginalized's 30d std nearly
   doubles: 0.021 → 0.033, driven mostly by draw 5 dropping ~0.04 AUROC).
   With only 5 draws this is plausibly noise rather than a real effect --
   `TIMELINE//` tokens are a much smaller fraction of a 30-day window's
   anchor candidates than of a 7-day window's, so the mechanical effect of
   excluding them should be smaller at 30d, not larger. Worth another look
   with more draws if this matters for a paper claim.
3. **The core finding from #99 still holds**: `marginalized(binary)` shows
   no consistent AUROC benefit over `patient_only` under Zach's refinement
   either -- the architecture Δ stays small and negative at both durations
   (7d: -0.0129, 30d: -0.0083), if anything slightly more negative than
   under #99's anchors, but still well within noise. Excluding `TIMELINE//`
   tokens from the anchor pool is a real, if modest, improvement to anchor
   quality; it does not change the substantive conclusion about retrieval.

## Aligning the query space to the doc space: Qwen3TextQueryProjector (MedRAP#101) (`sweep_marginalized_binary_qwen3_text_query_n25_{7,30}d.sh`)

Every retrieval result so far -- across k ∈ {4, 32, 64, 128}, 5-epoch
training, and both anchor-sampling fixes above -- shows the same thing:
`marginalized(binary)` never beats `patient_only` by more than noise. The
[top1-vs-random1 inference-style eval](#inference-style-evaluation-top-1-doc-vs-random-1-doc-eval_inference_style_n1248sh--eval_inference_style_duration_variance_n25sh--eval_inference_style_duration_variance_n25_large_ksh--eval_inference_style_duration_variance_n25_epoch5sh)
explains why at the mechanism level: across every N=25 multi-draw family,
the model's *actual best-retrieved document* performs statistically
indistinguishably from a *uniform-random corpus document* (Δ = +0.010,
std = 0.064, 59% win rate -- a coin flip). The model cannot tell its best
retrieved doc from a random one.

Root cause: the retrieval corpus (`MedRAG/textbooks`, generic medical
textbook chunks) is embedded with a frozen `Qwen3-Embedding-0.6B` model,
but the query side (`SequenceMeanQueryProjector`, a randomly-initialized
128→1024 linear layer) has no mechanism to align to that space. FAISS
nearest-neighbor search is non-differentiable, so the downstream task loss
can only reweight *among already-retrieved* documents -- it can never teach
the projector to find better ones in the first place. The query space and
the (fixed, pretrained) document space have no reason to share any
geometric structure, so nearest-neighbor search returns near-arbitrary
chunks throughout training.

`Qwen3TextQueryProjector` ([MedRAP#101](https://github.com/McDermottHealthAI/MedRAP/pull/101))
fixes this directly: instead of a learned projection, it renders each
patient's recent event codes as text (via the tensorized cohort's
`metadata/codes.parquet` code→description lookup) and embeds that text with
the *same* frozen Qwen3-Embedding-0.6B model used to build the doc corpus.
Query and document embeddings land in the same space by construction --
no learned alignment step, and nothing left for the query encoder to
"discover."

This reruns the same N=25/5-draw/7d+30d `marginalized(binary)` sweep as the
Zach anchor-refinement study above (same labels, same seeds), swapping only
`query_projector=qwen3_text` in place of `query_projector=sequence_mean_1024`.
`patient_only` doesn't use a query projector at all (`fusion=passthrough`),
so its results are unaffected and reused as-is from that study.

```bash
sbatch scripts/sweep_marginalized_binary_qwen3_text_query_n25_7d.sh
sbatch scripts/sweep_marginalized_binary_qwen3_text_query_n25_30d.sh
```

All 10 training runs (5 draws x 2 durations) finished cleanly.

### Results: patient_only vs. old (learned linear) vs. new (Qwen3-aligned) query projector

| Duration | Draw | patient_only | marginalized, old projector | marginalized, Qwen3TextQueryProjector |
| --- | --- | --- | --- | --- |
| 7d | 1 | 0.8609 | 0.8504 | [0.8528](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/9d2ggd13) |
| 7d | 2 | 0.8893 | 0.8875 | [0.8869](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/3ajzg6va) |
| 7d | 3 | 0.9219 | 0.8932 | [0.9097](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/oh694ktz) |
| 7d | 4 | 0.9371 | 0.9304 | [0.9275](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/ews5a1ij) |
| 7d | 5 | 0.8894 | 0.8726 | [0.8911](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/ckbhwxcv) |
| 30d | 1 | 0.8923 | 0.9050 | [0.9030](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/4697oi9a) |
| 30d | 2 | 0.9095 | 0.8807 | [0.8936](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/vgj0n56v) |
| 30d | 3 | 0.9030 | 0.8937 | [0.8989](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/ic0sgdx4) |
| 30d | 4 | 0.9446 | 0.9434 | [0.9466](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/6v0ytkwh) |
| 30d | 5 | 0.8691 | 0.8543 | [0.8698](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/vjle52il) |

**Summary statistics:**

| Duration | Metric | patient_only | old projector | Qwen3TextQueryProjector |
| --- | --- | --- | --- | --- |
| 7d | mean AUROC | 0.8997 | 0.8868 | 0.8936 |
| 7d | std across draws | 0.0300 | 0.0294 | 0.0279 |
| 30d | mean AUROC | 0.9037 | 0.8954 | 0.9024 |
| 30d | std across draws | 0.0275 | 0.0328 | 0.0278 |

**Architecture Δ (marginalized − patient_only), old vs. new projector:**

| Duration | Δ, old projector | Δ, Qwen3TextQueryProjector | draws won, old | draws won, new |
| --- | --- | --- | --- | --- |
| 7d | -0.0129 ± 0.0104 | **-0.0061 ± 0.0057** | 0/5 | 1/5 |
| 30d | -0.0083 ± 0.0155 | **-0.0013 ± 0.0097** | 1/5 | 3/5 |

**Takeaway:** aligning the query space to the doc space is a real, if
partial, improvement:

1. **The gap to `patient_only` roughly halves at both durations** (7d Δ:
   -0.0129 → -0.0061; 30d Δ: -0.0083 → -0.0013) and shrinks further in
   relative terms once you account for draw-to-draw std -- at 30d the new
   Δ (-0.0013) is now smaller than its own std (0.0097), i.e.
   indistinguishable from *zero*, not just closer to zero. At 30d,
   `marginalized` now wins outright on 3 of 5 draws, up from 1 of 5 under
   the old projector.
2. **7d still lags 30d.** The new projector's 7d Δ (-0.0061) is still
   negative, and `marginalized` only wins 1/5 draws there. This is
   consistent with the [top1-vs-random1 finding](#inference-style-evaluation-top-1-doc-vs-random-1-doc-eval_inference_style_n1248sh--eval_inference_style_duration_variance_n25sh--eval_inference_style_duration_variance_n25_large_ksh--eval_inference_style_duration_variance_n25_epoch5sh)
   that TIMELINE-token density (and thus how much of the anchor pool is
   "real" clinical signal vs. structural noise) differs by window length --
   a shorter window gives the model less to work with regardless of
   retrieval quality.
3. **Retrieval still doesn't consistently beat `patient_only`.** Fixing
   query-space alignment was necessary -- the model can now, at minimum,
   ask a geometrically meaningful nearest-neighbor question of the corpus --
   but it isn't sufficient to flip the sign of the effect. The most likely
   remaining bottleneck, per the root-cause analysis above, is the corpus
   itself: `MedRAG/textbooks` is generic medical domain knowledge, not
   patient-specific or even task-specific content. Even a perfectly
   calibrated query can only retrieve the closest textbook passage to "this
   patient's recent history" -- there's no guarantee that passage says
   anything about *this patient's specific future*. The next highest-value
   experiment is probably swapping the corpus itself (e.g. an
   episodic/similar-patient retrieval index) rather than further tuning the
   query encoder.

## 5-epoch variant (`sweep_patient_only_duration_variance_n25_epoch5.sh` + `sweep_marginalized_binary_duration_variance_n25_epoch5.sh`)

Same N=25, 5-draw, 7d/30d labels/architecture as above (k=4), but
`training.trainer.max_epochs=5` instead of 3, for both `patient_only` and
`marginalized(binary)` -- checks whether the (inconclusive, noise-dominated)
3-epoch comparison looks different with more training.

```bash
sbatch scripts/sweep_patient_only_duration_variance_n25_epoch5.sh
sbatch scripts/sweep_marginalized_binary_duration_variance_n25_epoch5.sh
```

### Results: 3-epoch vs. 5-epoch

| Duration | Draw | patient_only 3ep | patient_only 5ep | marginalized 3ep | marginalized 5ep | Δ 5ep (binary − patient_only) |
| --- | --- | --- | --- | --- | --- | --- |
| 7d | 1 | 0.9716 | [0.9436](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/zh4ibjnk) | 0.9511 | [0.9498](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/3uazx97p) | +0.0062 |
| 7d | 2 | 0.9923 | [0.9925](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/m8q67122) | 0.9899 | [0.9910](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/vxvo3uim) | -0.0015 |
| 7d | 3 | 0.9849 | [0.9844](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/vugazvja) | 0.9850 | [0.9875](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/1utd6jno) | +0.0031 |
| 7d | 4 | 0.9861 | [0.9729](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/pz5mj0j2) | 0.9863 | [0.9855](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/p3xtiyfz) | +0.0126 |
| 7d | 5 | 0.7867 | [0.8538](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/uqqhld7g) | 0.7816 | [0.7264](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/r44khzev) | -0.1274 |
| 30d | 1 | 0.9317 | [0.9465](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/mfvfuv6h) | 0.9057 | [0.9437](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/275hb7uo) | -0.0028 |
| 30d | 2 | 0.9666 | [0.9663](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/ij1wq87y) | 0.9528 | [0.9588](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/l9axo4h3) | -0.0075 |
| 30d | 3 | 0.9054 | [0.9043](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/goe2layj) | 0.9007 | [0.9308](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/30lraub1) | +0.0265 |
| 30d | 4 | 0.9831 | [0.8361](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/yx7ziupc) | 0.9880 | [0.9806](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/15tfv19g) | +0.1445 |
| 30d | 5 | 0.9277 | [0.9554](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/kd638bx8) | 0.9363 | [0.9583](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/u6b8hlyv) | +0.0029 |

**Δ mean ± std across the 5 draws, and the effect of going 3ep → 5ep per architecture:**

| Duration | 3ep Δ (mean ± std) | 5ep Δ (mean ± std) | patient_only 5ep − 3ep (mean) | marginalized 5ep − 3ep (mean) |
| --- | --- | --- | --- | --- |
| 7d | -0.0055 ± 0.0086 | -0.0214 ± 0.0595 | +0.0051 | -0.0107 |
| 30d | -0.0062 ± 0.0141 | +0.0327 ± 0.0638 | -0.0212 | +0.0177 |

**Takeaway:** training longer does **not** cleanly help either
architecture, and doesn't resolve the 3-epoch ambiguity -- if anything it
makes both the per-draw AUROC and the architecture-gap std noticeably
*more* volatile (7d Δ std goes from 0.009 to 0.060; 30d from 0.014 to
0.064). Two individual draws swing enormously with more training: 7d
draw 5's `patient_only` AUROC jumps 0.79 → 0.85 while `marginalized` drops
0.78 → 0.73 (a -0.127 flip in Δ), and 30d draw 4's `patient_only` AUROC
*collapses* 0.98 → 0.84 while `marginalized` barely moves (a +0.145 flip
in Δ) -- both look like instability/overfitting on a 25-task, small-N
setup rather than a genuine architecture effect. On average, `patient_only`
improves slightly at 7d but gets noticeably worse at 30d with more
training (mean -0.021, driven by that draw-4 collapse); `marginalized`
moves in the opposite direction each time. Net: no evidence that 5 epochs
is better than 3 for either architecture here, and the added variance
makes the `patient_only` vs. `marginalized` comparison harder to read, not
easier.

## Larger retriever k (`sweep_marginalized_binary_duration_variance_n25_large_k.sh`)

Same N=25, 5-draw, 7d/30d duration x variance study above, but sweeping
`retriever.k` up from the usual 4 to 32, 64, 128 (and 256, which reliably
OOMs -- see caveat below) to check whether attending to more retrieved
documents changes the `marginalized(binary)` vs. `patient_only` picture.
Reuses the exact same labels (`data/tasks_duration_variance/d{7,30}/
draw{1-5}/tasks`) and architecture -- only `retriever.k` differs.
`batch_size=32` is left unchanged from the k=4 sweeps (not scaled down for
larger k) per explicit direction: let it OOM if it OOMs, since that's a
real data point about the largest k this architecture/GPU actually
supports, not something to avoid by guessing a smaller batch size upfront.

```bash
sbatch scripts/sweep_marginalized_binary_duration_variance_n25_large_k.sh
```

**k=256 caveat:** at `batch_size=32`, k=256 reliably OOMs immediately on
the first forward pass (`PerDocCrossAttentionFusion` expands the batch to
`B*K` internally, so memory scales ~linearly with k) -- confirmed on an
L40S (44.4GiB total, ~44.25GiB in use at the crash, all 10/10 k=256 jobs
failed in under a minute). Dropped in favor of k in {32, 64, 128}, which
stays under that ceiling at the same batch size. k=256 can be revisited
later with a reduced batch size if needed.

**Status:** all 30 runs (k=32, k=64, k=128) finished. k=128 took ~13.5-14h
per run (vs. k=64's ~7-7.5h) -- each epoch's runtime scales roughly with k,
since the fusion module's effective batch size is `B*K`.

### Results: k=32 vs. patient_only and k=4

| Duration | Draw | patient_only | marginalized k=4 | marginalized k=32 | Δ k=32 vs. patient_only | Δ k=32 vs. k=4 |
| --- | --- | --- | --- | --- | --- | --- |
| 7d | 1 | 0.9716 | 0.9511 | [0.9563](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/xwi389rg) | -0.0153 | +0.0052 |
| 7d | 2 | 0.9923 | 0.9899 | [0.9822](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/4od4ifzs) | -0.0101 | -0.0077 |
| 7d | 3 | 0.9849 | 0.9850 | [0.9838](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/f74nouh1) | -0.0011 | -0.0012 |
| 7d | 4 | 0.9861 | 0.9863 | [0.9864](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/as804mub) | +0.0003 | +0.0001 |
| 7d | 5 | 0.7867 | 0.7816 | [0.7251](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/grzz8vfl) | -0.0616 | -0.0565 |
| 30d | 1 | 0.9317 | 0.9057 | [0.8883](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/ak0513mr) | -0.0434 | -0.0174 |
| 30d | 2 | 0.9666 | 0.9528 | [0.9595](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/8angtw06) | -0.0071 | +0.0067 |
| 30d | 3 | 0.9054 | 0.9007 | [0.8693](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/obm7eegn) | -0.0361 | -0.0314 |
| 30d | 4 | 0.9831 | 0.9880 | [0.9822](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/a2gg8zx0) | -0.0009 | -0.0058 |
| 30d | 5 | 0.9277 | 0.9363 | [0.9572](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/8xlyu1z8) | +0.0295 | +0.0209 |

**Δ mean ± std across the 5 draws (k=32 vs. patient_only):**

| Duration | mean Δ | std Δ |
| --- | --- | --- |
| 7d | -0.0176 | 0.0254 |
| 30d | -0.0116 | 0.0293 |

**So far:** k=32 doesn't reverse the pattern from k=4 -- still no
consistent benefit from retrieval over `patient_only` (mean Δ negative at
both durations, std larger than the mean effect). If anything, k=32 looks
slightly *worse* on average than k=4 at 7d (draw 5 in particular drops
sharply, 0.7816 -> 0.7251), though the small sample (5 draws) makes this
weak evidence rather than a clear trend -- k=64/128 results below should
clarify whether this is noise or a real degradation from attending to more
(likely less relevant) retrieved documents.

### Results: k=64 vs. patient_only and k=4

| Duration | Draw | patient_only | marginalized k=4 | marginalized k=64 | Δ k=64 vs. patient_only |
| --- | --- | --- | --- | --- | --- |
| 7d | 1 | 0.9716 | 0.9511 | [0.9496](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/frujicz8) | -0.0220 |
| 7d | 2 | 0.9923 | 0.9899 | [0.9882](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/65bbbbux) | -0.0041 |
| 7d | 3 | 0.9849 | 0.9850 | [0.9886](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/9p3khpl8) | +0.0037 |
| 7d | 4 | 0.9861 | 0.9863 | [0.9720](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/3pmeukdm) | -0.0141 |
| 7d | 5 | 0.7867 | 0.7816 | [0.8028](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/6aex40o1) | +0.0161 |
| 30d | 1 | 0.9317 | 0.9057 | [0.9085](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/5lw3wzxi) | -0.0232 |
| 30d | 2 | 0.9666 | 0.9528 | [0.9227](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/zou34lwq) | -0.0439 |
| 30d | 3 | 0.9054 | 0.9007 | [0.8785](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/9aw8z8ds) | -0.0269 |
| 30d | 4 | 0.9831 | 0.9880 | [0.9774](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/iryanard) | -0.0057 |
| 30d | 5 | 0.9277 | 0.9363 | [0.9470](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/cydbf1ju) | +0.0193 |

**Δ mean ± std across the 5 draws (k=64 vs. patient_only):**

| Duration | mean Δ | std Δ |
| --- | --- | --- |
| 7d | -0.0041 | 0.0149 |
| 30d | -0.0161 | 0.0240 |

**So far:** same story as k=4 and k=32 -- no consistent AUROC benefit from
retrieval, mean Δ is negative (or ~0) at both durations with std
comfortably larger than the mean effect. k=64's 7d mean Δ (-0.004) is
closer to zero than k=32's (-0.018) -- see the final k=128 results and
overall summary below for whether a trend across k actually holds up.

### Results: k=128 vs. patient_only and k=4

| Duration | Draw | patient_only | marginalized k=4 | marginalized k=128 | Δ k=128 vs. patient_only |
| --- | --- | --- | --- | --- | --- |
| 7d | 1 | 0.9716 | 0.9511 | [0.9486](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/cjlymdh1) | -0.0230 |
| 7d | 2 | 0.9923 | 0.9899 | [0.9899](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/430fx3b7) | -0.0024 |
| 7d | 3 | 0.9849 | 0.9850 | [0.9837](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/49fol7aq) | -0.0012 |
| 7d | 4 | 0.9861 | 0.9863 | [0.9722](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/znnf1s4i) | -0.0139 |
| 7d | 5 | 0.7867 | 0.7816 | [0.8002](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/p9dit375) | +0.0135 |
| 30d | 1 | 0.9317 | 0.9057 | [0.9047](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/1ftna39c) | -0.0270 |
| 30d | 2 | 0.9666 | 0.9528 | [0.9223](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/8hczalnc) | -0.0443 |
| 30d | 3 | 0.9054 | 0.9007 | [0.8471](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/r45vhdj8) | -0.0583 |
| 30d | 4 | 0.9831 | 0.9880 | [0.9825](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/zx6nmdpq) | -0.0006 |
| 30d | 5 | 0.9277 | 0.9363 | [0.9584](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/vzm04lrf) | +0.0307 |

**Δ mean ± std across the 5 draws (k=128 vs. patient_only):**

| Duration | mean Δ | std Δ |
| --- | --- | --- |
| 7d | -0.0054 | 0.0138 |
| 30d | -0.0199 | 0.0355 |

## Summary across k ∈ {4, 32, 64, 128}

**Δ mean (marginalized − patient_only) across the 5 draws, by k and duration:**

| k | 7d mean Δ | 30d mean Δ |
| --- | --- | --- |
| 4 | -0.0055 | -0.0062 |
| 32 | -0.0176 | -0.0116 |
| 64 | -0.0041 | -0.0161 |
| 128 | -0.0054 | -0.0199 |

**Raw AUROC mean ± std across the 5 random task draws** (not the Δ -- the
absolute per-architecture AUROC variance driven purely by which 25 tasks
got sampled into a given draw):

| Architecture | 7d mean ± std (min-max) | 30d mean ± std (min-max) |
| --- | --- | --- |
| `patient_only` | 0.9443 ± 0.0884 (0.7867-0.9923) | 0.9429 ± 0.0314 (0.9054-0.9831) |
| `marginalized` k=4 | 0.9388 ± 0.0893 (0.7816-0.9899) | 0.9367 ± 0.0359 (0.9007-0.9880) |
| `marginalized` k=32 | 0.9268 ± 0.1134 (0.7251-0.9864) | 0.9313 ± 0.0494 (0.8693-0.9822) |
| `marginalized` k=64 | 0.9402 ± 0.0785 (0.8028-0.9886) | 0.9268 ± 0.0376 (0.8785-0.9774) |
| `marginalized` k=128 | 0.9389 ± 0.0791 (0.8002-0.9899) | 0.9230 ± 0.0522 (0.8471-0.9825) |

This is the more important variance number for planning future N=25 runs:
**which 25 random tasks you happen to draw swings absolute AUROC far more
than the choice of architecture or k does.** At 7d, std across draws
(0.079-0.113) is roughly 15-20x the architecture-vs-patient_only Δ std
seen per k (0.009-0.025 in the sections above); at 30d it's smaller in
absolute terms (0.031-0.052) but still several times the Δ std. Draw 5 at
7d (patient_only=0.7867) and draw 4 at 30d in the epoch5 study are the
extreme low points pulling these numbers around -- with only 5 draws, a
single hard draw dominates the spread. This is the main argument for why
any single-draw comparison (the very first "Results" table earlier in this
README, before the variance study existed) should not be over-interpreted,
and why more draws (not just more k or more epochs) would be the highest-
value next experiment if tighter confidence intervals are wanted.

**Takeaway:** no k in {4, 32, 64, 128} produces a consistent AUROC
improvement from retrieval over `patient_only` -- every mean Δ is negative
(or ~0), and every std (0.014-0.036, not shown per-k above but see each
section) is comparable to or larger than the mean effect, so within a
given k the win/loss is noise-dominated. Across k, the picture differs by
duration: at **7d**, Δ bounces around a small negative band (-0.0041 to
-0.0176) with no clear direction as k grows -- k=32 is the worst point,
k=64 the closest to zero, k=128 back down near k=4's level. At **30d**,
Δ gets **monotonically more negative as k increases** (-0.0062 → -0.0116 →
-0.0161 → -0.0199 from k=4 to k=128) -- a real, if modest, trend suggesting
that attending to more retrieved documents doesn't help and may mildly
hurt on the longer occurrence window, though the std at each k (0.014-0.036)
is still large enough that this shouldn't be overclaimed from 5 draws.
Combined with the [5-epoch results](#5-epoch-variant-sweep_patient_only_duration_variance_n25_epoch5sh--sweep_marginalized_binary_duration_variance_n25_epoch5sh)
(more training also doesn't help), the overall picture across this whole
round of experiments is that neither training longer nor retrieving more
documents rescues `marginalized(binary)` retrieval's lack of benefit over
`patient_only` at N=25 on random MIMIC-IV task labels.

## Inference-style evaluation: top-1 doc vs. random-1 doc (`eval_inference_style_n1248.sh` + `eval_inference_style_duration_variance_n25.sh` + `eval_inference_style_duration_variance_n25_large_k.sh` + `eval_inference_style_duration_variance_n25_epoch5.sh`)

Two questions about every `marginalized(binary)` checkpoint trained on
random-task-code labels so far (base N-sweep, k=4/32/64/128 duration
x variance, 3-epoch and 5-epoch): (1) does "inference style" prediction
using only the single **top-retrieved** document (`retriever.k=1`,
matching [`mimic_iv_sweep_frequent`'s top-1-doc
eval](../mimic_iv_sweep_frequent/README.md#held-out-test-auroc-top-retrieved-document-only))
track the training-time marginalized-over-k AUROC, and (2) is that
top-retrieved document actually doing anything useful, or would a
**uniform-random** corpus document (breaking patient-document alignment
entirely) perform just as well?

Both modes use `retriever.k=1` at eval time (valid for any checkpoint
regardless of its trained k -- `PerDocCrossAttentionFusion` and
`marginalized_output_mode=binary` marginalization are both K-agnostic, no
weight or positional embedding is sized by K):

- **top1**: `retriever.ablation_mode=none` -- the single nearest-neighbor
  (highest-scoring) retrieved document, MedRAP's normal retrieval
  behavior.
- **random1**: `retriever.ablation_mode=random_docs` -- MedRAP's built-in
  retrieval ablation (`medrap/model/retrievers.py`'s
  `_apply_retrieval_ablation`), a uniform-random corpus row sampled with
  replacement, discarding retrieval quality entirely while keeping the
  same payload/model path.

Both evaluate held-out **test** split AUROC (`eval_mode=test`), not the
`tuning`/`val` split every other result in this README reports -- see
[`mimic_iv_sweep_frequent`'s top-1-doc
section](../mimic_iv_sweep_frequent/README.md#held-out-test-auroc-top-retrieved-document-only)
for why (`eval_mode=test` targets a split nothing else here has touched).

```bash
sbatch scripts/eval_inference_style_n1248.sh                            # base N-sweep, 16 jobs
sbatch scripts/eval_inference_style_duration_variance_n25.sh            # k=4 duration-variance, 20 jobs
sbatch scripts/eval_inference_style_duration_variance_n25_large_k.sh    # k=32/64/128 duration-variance, 60 jobs
sbatch scripts/eval_inference_style_duration_variance_n25_epoch5.sh     # 5-epoch duration-variance, 20 jobs
```

All 116 jobs (58 checkpoints x 2 modes) finished cleanly, no failures.


### Summary: does top1 beat random1?

**Δ (top1 − random1) mean ± std, per checkpoint family:**

| Family | n pairs | mean Δ | std Δ | top1 wins |
| --- | --- | --- | --- | --- |
| Base N-sweep (N=1..128, single draw, k=4 trained) | 8 | +0.0656 | 0.1156 | 7/8 |
| Duration-variance, k=4 trained | 10 | +0.0077 | 0.0244 | 4/10 |
| Duration-variance, k=32 trained | 10 | +0.0105 | 0.0301 | 7/10 |
| Duration-variance, k=64 trained | 10 | -0.0055 | 0.0737 | 6/10 |
| Duration-variance, k=128 trained | 10 | -0.0038 | 0.0602 | 5/10 |
| Duration-variance, 5-epoch (k=4) | 10 | -0.0020 | 0.0393 | 5/10 |
| **All 58 pairs combined** | 58 | **+0.0102** | **0.0639** | **34/58 (59%)** |

**Takeaway:** overall, top1 is not reliably better than random1 -- the
combined mean Δ (+0.010) is an order of magnitude smaller than the
combined std (0.064), and the win rate (59%) is barely above a coin flip.
**The one clear exception is the base N-sweep**, where top1 beats random1
on 7/8 checkpoints with a much larger mean gap (+0.066) -- most strikingly
at N=1 (top1=0.987 vs. random1=0.686, Δ=+0.301). That family differs from
the other three in two ways that could explain the gap rather than a
genuine "retrieval quality matters more at N=1" effect: it's a single
draw per N (no averaging over multiple random task-code samples like the
duration-variance families), and small N means AUROC is dominated by
very few tasks (N=1 is literally one task), so a single held-out-split
quirk can swing the metric enormously -- exactly the kind of small-sample
instability the [duration x variance
study](#duration-x-variance-study-n25-patient_only-vs-marginalized-binary-at-7-day-and-30-day-durations-5-random-draws-each)
was built to average out. Across the four N=25, multi-draw families (which
control for that), top1 vs. random1 is indistinguishable from noise --
the trained marginalization does not appear to be leaning on retrieval
quality in a way that survives collapsing to a single document, for
better or worse.


### Full results

### Base N-sweep (N=1..128, k=4 trained)

| N | top1 test AUROC | random1 test AUROC | Δ (top1 − random1) |
| --- | --- | --- | --- |
| 1 | [0.9871](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/2cm4xtmi) | [0.6861](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/zs4gicgm) | +0.3011 |
| 2 | [0.8888](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/7uc2e07d) | [0.8288](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/nl9yhyb7) | +0.0600 |
| 4 | [0.7864](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/u3p7q0xs) | [0.6137](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/cwwlofm9) | +0.1727 |
| 8 | [0.7962](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/92ikv84c) | [0.8507](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/3e98wu8l) | -0.0544 |
| 16 | [0.8686](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/udqplito) | [0.8654](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/364zvjmh) | +0.0032 |
| 32 | [0.9314](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/amfcwe3f) | [0.9311](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/5ddbcbwl) | +0.0003 |
| 64 | [0.8819](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/9jofl3d5) | [0.8568](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/3yqn6oqm) | +0.0252 |
| 128 | [0.8927](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/2v6n9whl) | [0.8754](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/jjqe0zjp) | +0.0173 |

### Duration x variance (N=25, k=4 trained, 3 epochs)

| Duration | Draw | top1 | random1 | Δ |
| --- | --- | --- | --- | --- |
| 7d | 1 | [0.9525](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/9m3fk7v6) | [0.9550](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/ov3ylhut) | -0.0025 |
| 7d | 2 | [0.9600](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/x3ztt5sg) | [0.9633](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/3j5ja70l) | -0.0033 |
| 7d | 3 | [0.7365](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/w8wtr83l) | [0.6644](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/mzrdxntm) | +0.0721 |
| 7d | 4 | [0.9883](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/t5ewf5f5) | [0.9886](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/rd0s6ek2) | -0.0003 |
| 7d | 5 | [0.9905](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/8hg4s4l4) | [0.9898](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/8av7sm2b) | +0.0007 |
| 30d | 1 | [0.8161](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/438682n2) | [0.7965](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/6jlci5as) | +0.0196 |
| 30d | 2 | [0.8195](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/uyk9s6rk) | [0.8198](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/8bzlno33) | -0.0003 |
| 30d | 3 | [0.8366](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/7gulz1g9) | [0.8479](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/bq4pexqn) | -0.0113 |
| 30d | 4 | [0.9250](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/7xc0dhxm) | [0.9339](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/78jdc4o3) | -0.0089 |
| 30d | 5 | [0.7826](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/hpu212v4) | [0.7709](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/y78kkrl4) | +0.0117 |

### Duration x variance (N=25, k=32 trained, 3 epochs)

| Duration | Draw | top1 | random1 | Δ |
| --- | --- | --- | --- | --- |
| 7d | 1 | [0.9450](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/0a0lrd37) | [0.9534](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/koyft47j) | -0.0083 |
| 7d | 2 | [0.9183](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/2ceq49tv) | [0.9183](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/433q0ta7) | +0.0000 |
| 7d | 3 | [0.6636](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/0vfb4gns) | [0.7047](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/palru30e) | -0.0412 |
| 7d | 4 | [0.9856](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/rhzcgrt3) | [0.9853](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/4m0py3fs) | +0.0003 |
| 7d | 5 | [0.9910](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/31kr6cez) | [0.9898](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/cay4nh20) | +0.0012 |
| 30d | 1 | [0.8381](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/u3srrydo) | [0.7763](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/u9bwtj2m) | +0.0618 |
| 30d | 2 | [0.8312](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/8grhfn44) | [0.8031](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/5svdnm5r) | +0.0281 |
| 30d | 3 | [0.8225](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/1syagvsm) | [0.8288](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/as8ncu0t) | -0.0063 |
| 30d | 4 | [0.9357](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/3ns18bp1) | [0.8860](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/3ige9uwu) | +0.0497 |
| 30d | 5 | [0.7148](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/uhvair19) | [0.6951](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/8hggd7ng) | +0.0197 |

### Duration x variance (N=25, k=64 trained, 3 epochs)

| Duration | Draw | top1 | random1 | Δ |
| --- | --- | --- | --- | --- |
| 7d | 1 | [0.9352](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/cjsvh9ja) | [0.9094](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/xo6o2qza) | +0.0258 |
| 7d | 2 | [0.8684](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/pn3a8ffk) | [0.7687](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/cx2sm2ut) | +0.0997 |
| 7d | 3 | [0.6739](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/t9nrto5m) | [0.8631](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/0o8ia3rx) | -0.1893 |
| 7d | 4 | [0.9876](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/rzndb1iy) | [0.9858](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/qjami2q5) | +0.0018 |
| 7d | 5 | [0.9897](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/9d6akz52) | [0.9900](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/i6iawe1r) | -0.0003 |
| 30d | 1 | [0.8167](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/vgal51do) | [0.8277](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/ztbg00q4) | -0.0110 |
| 30d | 2 | [0.8241](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/3d6dzucl) | [0.8120](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/4639b3vf) | +0.0121 |
| 30d | 3 | [0.8297](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/py0cf2g0) | [0.8032](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/jff2rlxg) | +0.0265 |
| 30d | 4 | [0.8994](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/jtlqci9h) | [0.9382](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/nu327cno) | -0.0388 |
| 30d | 5 | [0.7811](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/3l50m4o1) | [0.7627](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/1j3awjwx) | +0.0184 |

### Duration x variance (N=25, k=128 trained, 3 epochs)

| Duration | Draw | top1 | random1 | Δ |
| --- | --- | --- | --- | --- |
| 7d | 1 | [0.9347](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/m590y12h) | [0.8833](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/xf4qxlcg) | +0.0514 |
| 7d | 2 | [0.9584](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/6q32nivs) | [0.9578](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/zgdiu6am) | +0.0007 |
| 7d | 3 | [0.6593](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/lw6i0vir) | [0.7543](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/m8ww3zn8) | -0.0949 |
| 7d | 4 | [0.9820](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/q9ue62qv) | [0.9803](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/q2i29jyw) | +0.0017 |
| 7d | 5 | [0.9876](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/jp40hvyb) | [0.9880](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/17ve6zvy) | -0.0004 |
| 30d | 1 | [0.8259](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/yvssjmp4) | [0.8087](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/amy1zhc4) | +0.0171 |
| 30d | 2 | [0.8518](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/te3b75zw) | [0.7414](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/qyiin51c) | +0.1104 |
| 30d | 3 | [0.8290](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/8vma56ju) | [0.8301](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/c8p39q1w) | -0.0011 |
| 30d | 4 | [0.8719](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/wccnwq2c) | [0.9100](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/cliuykjy) | -0.0381 |
| 30d | 5 | [0.7145](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/bto3dxsn) | [0.7993](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/t4u8jcz7) | -0.0848 |

### Duration x variance (N=25, k=4 trained, 5 epochs)

| Duration | Draw | top1 | random1 | Δ |
| --- | --- | --- | --- | --- |
| 7d | 1 | [0.9209](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/7644gcux) | [0.8770](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/p9ricznx) | +0.0439 |
| 7d | 2 | [0.9596](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/4m3vtmz4) | [0.9634](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/fpd92ysu) | -0.0038 |
| 7d | 3 | [0.6658](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/x5y4mv9y) | [0.7198](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/a2h77wzt) | -0.0539 |
| 7d | 4 | [0.9840](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/qbohppa0) | [0.9870](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/m43bpsn2) | -0.0030 |
| 7d | 5 | [0.9902](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/yd3jcg5u) | [0.9892](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/srs6vdos) | +0.0010 |
| 30d | 1 | [0.8029](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/6haf6g5t) | [0.7398](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/bumwmjyr) | +0.0631 |
| 30d | 2 | [0.8359](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/lwo5kbcp) | [0.8284](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/kfmisuop) | +0.0075 |
| 30d | 3 | [0.8266](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/20xmv55p) | [0.8966](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/xt3qha3t) | -0.0700 |
| 30d | 4 | [0.9034](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/gz1ylzyb) | [0.8980](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/xctqtk7t) | +0.0055 |
| 30d | 5 | [0.7556](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/ck6g7jts) | [0.7663](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/4mdzp52z) | -0.0107 |

## Variance study: patient_only vs. marginalized(binary) across repeated random task draws (`generate_labels_variance_n1248.sh` + `sweep_patient_only_variance_n1248.sh` + `sweep_marginalized_binary_variance_n1248.sh`)

**Note:** the smaller-scope [duration x variance study](#duration-x-variance-study-n25-patient_only-vs-marginalized-binary-at-7-day-and-30-day-durations-5-random-draws-each)
above (fixed N=25, 5 draws, 7d and 30d) already answers a similar question
and has results. This section is the original, larger 8-N x 5-draw design
(120 SLURM tasks) -- run it if per-N (not just N=25) variance is needed;
otherwise the section above is the faster path to "is the win robust to
random draws."

Advisor's ask: for each N, sample N random task codes, train both
`patient_only` and `marginalized(binary)` on that exact draw, compute the
AUROC difference, then repeat with a fresh random draw of N codes and see
whether the difference holds up or is just noise from one particular
sample. Answers "is the retrieval benefit robust to which N tasks you
happened to pick?" as opposed to the single-draw results in the sections
above.

For every N in `{1, 2, 4, 8, 16, 32, 64, 128}`, this generates 5
independent random draws of N task codes (`code_selection=random`, a
distinct seed per draw: `101, 202, 303, 404, 505` -- deliberately different
from `seed=42` used everywhere else in this repo, so no draw accidentally
coincides with an existing label set), then trains both architectures on
each of the resulting 40 label sets, so `patient_only` and
`marginalized(binary)` see the *identical* task codes/labels within a draw
(a paired comparison). Fixed 7-day horizon, same as the base random-task
sweep above -- only the extra draw dimension is new.

```bash
sbatch scripts/generate_labels_variance_n1248.sh          # labels: 8 N x 5 draws = 40 jobs, CPU
sbatch scripts/sweep_patient_only_variance_n1248.sh        # training: 40 jobs, GPU
sbatch scripts/sweep_marginalized_binary_variance_n1248.sh # training: 40 jobs, GPU
```

Run the label-generation array first and wait for it to finish before
submitting either training sweep (both training scripts check for their
draw's label directory and exit with an error if it's missing). The two
training sweeps don't depend on each other and can run concurrently.

Labels land at `data/tasks_variance/draw{1..5}/n{N}/tasks/`. Checkpoints at
`outputs/patient_only_variance_n1248/draw{d}/n{N}/` and
`outputs/marginalized_binary_variance_n1248/draw{d}/n{N}/`. W&B run names:
`patient-only-variance-draw{d}-n{N}-*` and
`marginalized-binary-variance-draw{d}-n{N}-*`.

**Scale note:** this is 40 CPU label-gen jobs + 80 GPU training jobs (120
SLURM array tasks total) -- substantially more compute than any prior
sweep in this repo. Consider trimming `--array` to a subset of N values
first (e.g. `sbatch --array=0-4,35-39 ...` for just N=1 and N=128, the
extremes) if GPU allocation is tight.

Once all 5 draws finish for a given N, compute
`AUROC(marginalized) - AUROC(patient_only)` per draw and report
mean ± std across the 5 draws (a small `pandas`/W&B-API script pulling
`val/auroc/mean` from both run families, grouped by N and draw, is the
simplest way to do this -- no such aggregation script exists yet).

*(Results pending -- all three sweeps above still need to be run.)*

## Capacity-starved patient_only vs. marginalized retrieval (N=25, 30d)

Moved to its own directory for easier review -- see
[`results/capacity_starved_retrieval/README.md`](../results/capacity_starved_retrieval/README.md).

**Headline result**: this is the first experiment in the repo where
`marginalized` retrieval *consistently* beats `patient_only`. Under an
~8x capacity cut to the patient encoder, both query-projector variants
flip from a small inconsistent loss at full capacity to a small but
unanimous (5/5 draws) win (learned-linear: -0.0083 → +0.0125;
`qwen3_text`: -0.0013 → +0.0248).
