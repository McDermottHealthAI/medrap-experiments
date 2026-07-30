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
