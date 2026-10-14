# mimic_iv_scaling_laws: design

This doc explains what the experiment runs and why. The README covers how to run it.

## Question

Does retrieval's advantage over a patient-only model shrink, hold, or grow as the model
and the training set get bigger?

Earlier experiments looked at one axis at a time:

- `mimic_iv_sweep/*capacity_sweep*` varied encoder width at full data.
- `mimic_iv_sweep/*data_scarcity*` and `*extreme_starved*trainN*` varied training-set size
    at a fixed width.

Both found that retrieval helps most when one resource is scarce. Neither can say how the
two axes interact, or extrapolate past the largest point run. This experiment runs the
full cross product and fits a scaling law to each arm, so the two arms' curves can be
compared anywhere on the grid and a short distance past it.

## Law and fitting

For each arm separately:

```
AUROC ~ E + A * N^-alpha + B * D^-beta      (fit on the logit scale)
```

| Symbol | Meaning | Column in `scaling_laws_results.parquet` |
|---|---|---|
| `N` | patient-encoder parameter count | `model_size__n_params` |
| `D` | number of training patients | `dataset_size__n_train_patients` |
| AUROC | test (`held_out`) AUROC, mean over the 25 tasks | `test_metric__auroc` |

Fitting is done by [`simple-scaling-laws`](https://github.com/mmcdermott/simple-scaling-laws),
pinned by commit in `pyproject.toml`. It fits the separable power law, bootstraps over
configurations (wild cluster bootstrap) for uncertainty, and fits unit-interval metrics
like AUROC on the logit scale so predictions stay inside (0, 1). `compare` then evaluates
both arms' bootstrap draws at the same points and reports the difference and
`p_better_than_reference`. This experiment adds no statistics of its own.

**Why AUROC and not loss:** the two arms train with different losses (BCE vs. a
marginalized BCE over retrieved documents), so their loss values aren't on the same scale.
AUROC is.

**Why one fit per arm** rather than one joint fit with an arm term: each arm gets its own
exponents, so "retrieval improves faster or slower with data" shows up directly as a
different `beta`. Comparisons happen afterwards, on predictions.

**How the IDs map:**

| simple-scaling-laws column | Value | Why |
|---|---|---|
| `training_run_id` | `<arm>_dim<D>_trainN<N>_draw<d>` | one per trained model |
| `train_set_id` | `trainN<N>_draw<d>` | both arms train on the identical label file |
| `test_set_id` | `held_out_draw<d>` | each draw has its own 25 tasks, so its own test labels |
| `optimizer_seed` | draw | `seed=${DRAW}` in `train.sh` |

The 5 draws at a given (dim, N) are repeats of one configuration, which is what lets the
package estimate run-to-run noise. This is why `N` is an exact count shared by every draw.
With percentages, draws differed by a few rows and each one would count as its own
configuration.

## Grid

All values live in `scripts/grid.sh`:

| Axis | Values | Count |
|---|---|---|
| arm | `patient_only`, `retrieval` | 2 |
| `encoder.embedding_dim` | 4, 8, 16, 32, 64, 128 | 6 |
| training patients | 1,000, 3,000, 10,000, 30,000, 100,000 | 5 |
| draw | 1–5 | 5 |

That's 300 training runs, submitted as 60 SLURM arrays of 5 tasks each.

- **Both axes are log-spaced** (roughly ×2 for width, ×3 for data), because a power law
  is linear in log-log space and evenly spaced log points pin the exponents down best.
- **Width tops out at 128**, the width all previous full-capacity runs used.
- **Data tops out at 100k, not the full ~153k.** The exact full size differs per draw and
  can't be checked while the cluster is down. Any `N` above a draw's size would make
  `subsample_train_labels.py` fail for that draw. A full-data point can be added once the
  sizes are known.
- **Data starts at 1k.** Experiment 3 of `results/extreme_capacity_starved_retrieval`
  showed that at N=100 both arms sit at chance (AUROC ≈ 0.52) even with tuned epochs.
  Points like that carry no signal about the exponents.

## Models

Both arms share the patient side. They differ only in what sits between the encoder and
the head.

```
patient_only:  codes ─▶ RoPE encoder ─▶ passthrough ─▶ masked-mean pool ─▶ linear head (25 logits)

retrieval:     codes ─▶ RoPE encoder ─┬─▶ query projector ─▶ FAISS top-4 docs ─▶ token-embedding encoder ─┐
                                      └──────────────────────────────────────────────────────────────────┴─▶ per-doc cross-attention fusion
                                                                                     ─▶ masked-mean pool ─▶ linear head ─▶ marginalize over docs
```

### Shared: patient encoder (the capacity axis)

`encoder=rope` → `medrap.model.encoders.TimeDeltaRoPEPatientEncoder`, a transformer over
the patient's MEDS code sequence with rotary position embeddings driven by the time between
events.

| Override | Value | Why |
|---|---|---|
| `encoder.embedding_dim` | `DIM` | **the only thing varied on the capacity axis** |
| `encoder.num_heads` | 1 | fixed, so a width of 4 is valid and the model shape doesn't change across the grid |
| `encoder.num_layers` | 1 | fixed, for the same reason |
| `encoder.ff_dim` | `2 * DIM` | scales with width, as in `capacity_sweep` |
| `encoder.vocab_size` | 65536 (config default) | |
| `training.datamodule.config.max_seq_len` | 256 | fixed. Older `capacity_starved` lines also cut context, which mixed two effects |

Changing only width matches `mimic_iv_sweep`'s `capacity_sweep` cut and gives one clean
knob. The cost is that the dim=128 cell here (1 head, 1 layer) is **not** the same model as
the earlier full-capacity baseline (`encoder=rope` default: 4 heads, 2 layers).

Patient-encoder parameter counts (`N`). The code-embedding table (65,536 × DIM) dominates:

| DIM | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|
| `N` | 262K | 525K | 1.05M | 2.11M | 4.23M | 8.52M |

`aggregate_results.py` computes these by building the encoder with the same shape as
`train.sh`, so the two must be kept in sync.

Shared settings: `pooling=masked_mean`, `head=linear` with 25 outputs,
`training/task=multitask_binary`, `training/datamodule=meds_multitask`, batch size 32,
`seq_sampling_strategy=to_end` (the most recent 256 events), `lr=1e-3`, gradient clipping 1.0.

### patient_only

| Override | Value |
|---|---|
| `fusion` | `passthrough` (encoder output goes straight to pooling) |
| `head.in_dim` | `DIM` |
| `training/loss` | `multitask_binary_bce` (`MultiTaskBCELoss`) |

### retrieval

The same configuration as `mimic_iv_sweep`'s `marginalized_binary` + learned-linear line,
which is the configuration behind the existing capacity and data-scarcity results.

| Component | Override | What it is |
|---|---|---|
| Query projector | `query_projector=sequence_mean_1024`, `in_dim=DIM` | mean-pools the encoder states over the sequence, then a learned linear map to 1024-D, the key-embedding size of the corpus |
| Retriever | `retriever=hf_dataset`, `k=4` | FAISS search over the prebuilt corpus at `data/retrieval_db` |
| Corpus | (built once by `mimic_iv_sweep/scripts/prepare_retrieval.sh`) | `MedRAG/textbooks` (title + content), tokenized and embedded with `Qwen/Qwen3-Embedding-0.6B`, max 256 tokens per document |
| Document encoder | `retrieval_encoder=token_feature`, vocab 151,936, dim 64 | a learned embedding over the retrieved documents' Qwen token ids |
| Fusion | `fusion=cross_attention_perdoc_medium` | `PerDocCrossAttentionFusion`: patient states attend to each document separately (d_model 256, 8 heads, 2 layers) |
| Output | `marginalized_retrieval=true`, `marginalized_output_mode=binary` | per-document predictions, weighted by the softmax over retrieval scores (dot product scaled by 1/√D, MedRAP #113) |
| Head | `head.in_dim=256` | fusion output width |
| Loss | `multitask_binary_bce_marginalized` | BCE on the marginalized per-task probability |

The document encoder and fusion **don't change** with `DIM`. Only the query projector and
fusion's patient-side input width follow the encoder.

### Parameter accounting

`N` counts only the patient encoder. The retrieval arm also carries about **10.6M fixed
parameters**: the 9.72M document token embedding (151,936 × 64), about 0.86–0.89M of fusion,
a 5K–132K query projector, and a 6.4K head. That's ~40× the patient encoder at dim=4 and
~1.2× at dim=128.

This is deliberate. The capacity question is "how big does the *patient model* have to be",
and the retrieval machinery is a fixed cost that doesn't scale with it. But it means the two
arms do not have equal parameter counts at the same `N`. Most of the extra parameters are a
token-embedding table, which looks up rather than computes, so it isn't comparable to
encoder capacity. If a reviewer asks for a parameter-matched comparison, the fix is to refit
using total parameters as `N`. That needs only a change in `aggregate_results.py`, not new
runs.

## Data

- **Cohort:** MIMIC-IV MEDS, the tensorized restore at
    `/groups/mm6677_gp/zzw2102/data/MIMIC_MEDS_restored/processed`. This is the same copy
    recent `mimic_iv_sweep` scripts and the ablation lines use, since the original was
    deleted.
- **Tasks:** the existing `mimic_iv_sweep/data/tasks_zach_uniform_event_n25_30d/draw{1..5}`
    labels. Each draw has 25 binary tasks (does code X occur within 30 days of the
    prediction time), one `uniform_event` prediction time per patient, and its own task
    sample. No new labels are generated. Reusing them keeps the results comparable with
    every earlier N=25 30d result.
- **Unit of data:** a training patient. MedRAP's task generation writes one row per patient
    (wide `task_0..task_24` columns, one shared prediction time), so rows and patients are
    the same thing.
- **Subsampling (`scripts/make_labels.sh`):** for each N and draw,
    `subsample_train_labels.py --n-rows N --seed 1000000*draw + N` draws N patients from that
    draw's `train.parquet`, uniformly at random. `tuning.parquet` and `held_out.parquet` are
    copied unchanged, so within a draw every grid point is validated and tested on the same
    patients. Both arms read the same subsampled file.
- **Subsets are independent, not nested:** the 1k set is not a subset of the 3k set. This
    matches the existing script. Nesting would lower variance along the data axis slightly,
    but would mean changing a script shared with earlier experiments.

## Training schedule

A fixed epoch count gives small datasets almost no optimizer steps: 1k patients ÷ 32 per
batch × 3 epochs ≈ 96 steps, which never gets past a 200-step warmup. Epochs and warmup
therefore scale with N:

| N | `max_epochs` | `warmup_steps` | ≈ optimizer steps |
|---|---|---|---|
| 1,000 | 10 | 60 | 320 |
| 3,000 | 7 | 100 | 660 |
| 10,000 | 5 | 200 | 1,560 |
| 30,000 | 4 | 200 | 3,750 |
| 100,000 | 3 | 200 | 9,380 |

1k, 10k and 100k are copied from Experiment 3 of `results/extreme_capacity_starved_retrieval`.
3k and 30k are interpolated. The schedule is **not compute-matched**: small-N runs take far
fewer steps. That's the usual data-scaling setup (more data means more steps at a fixed
epoch budget), but it means `beta` measures data plus the extra steps that come with it,
not data alone.

Width doesn't change the schedule. Every DIM at a given N gets the same epochs and warmup.

## Evaluation

Each `train.sh` task trains, then runs `medrap-eval eval_mode=test` on
`checkpoints/last.ckpt` in the same job. It evaluates the last checkpoint rather than the
best one on validation, which matches the existing `eval_test_*` scripts and avoids picking
checkpoints on validation data at small N.

`aggregate_results.py` reads `test/auroc/mean` from each eval's CSV logger
(`outputs/<arm>_dim<D>_trainN<N>_test/draw<d>/loggers/csv/*/metrics.csv`). Runs are also
logged to W&B, named `sl-<arm>_dim<D>_trainN<N>[-test]-draw<d>-<jobid>_<idx>`.

There's one test evaluation per run, so the package can't separate evaluation noise from
run-to-run noise. It warns about this (`no_repeated_evaluations`) and the run-level variance
absorbs both. Bootstrapping the test set inside each eval would separate them, but MedRAP's
eval doesn't output that today.

## How it runs

Everything happens on the C2B2 cluster from inside `mimic_iv_scaling_laws/`. Nothing in
MedRAP changes. The experiment pins a MedRAP commit and passes Hydra overrides.

```
uv sync                       # .venv with pinned medrap + simple-scaling-laws
ln -s .../mimic_iv_sweep/data/retrieval_db data/retrieval_db
scripts/make_labels.sh        # 25 label dirs: data/tasks_n25_30d_trainN<N>/draw<d>/tasks
scripts/submit_sweep.sh       # 60 × sbatch scripts/train.sh <arm> <dim> <N>   (array 0-4 = draws)
.venv/bin/python scripts/aggregate_results.py
```

| File | Role |
|---|---|
| `scripts/grid.sh` | the grid. Sourced by `make_labels.sh` and `submit_sweep.sh` |
| `scripts/make_labels.sh` | builds the subsampled label dirs |
| `scripts/subsample_train_labels.py` | copied unchanged from `mimic_iv_sweep` so this directory is self-contained |
| `scripts/train.sh` | one SLURM array per cell: `ARM DIM N_TRAIN`, with shared overrides, the arm-specific block, the per-N schedule, train, then test eval |
| `scripts/submit_sweep.sh` | loops over the grid and submits each cell |
| `scripts/aggregate_results.py` | collects AUROC, fits each arm, compares them, writes two parquets and prints the summary |

`train.sh` builds one override array shared by `medrap-train` and `medrap-eval`, so the
eval model is guaranteed to have the same architecture as the trained one. Training-only
settings (epochs, warmup, lr, seed) are passed to `medrap-train` only.

**Resources per task:** 1× L40S, 8 CPUs, 32 GB, 6 h limit. This matches the capacity-sweep
scripts, with extra time to cover the 10-epoch small-N cells and the eval step.

## Outputs

- `outputs/scaling_laws_results.parquet`: the fit input, one row per test eval.
- `outputs/scaling_laws_comparison.parquet`: for every (N, D) point on the grid plus one
    step of 10× in each direction, each arm's fitted AUROC (median and 95% interval), the
    retrieval − patient_only difference, `p_better_than_reference`, and whether the point is
    interpolated or extrapolated.
- Printed: each arm's `model.summary()` (parameters, confidence intervals, R², and warnings
    such as `weakly_identified_exponent` and `lack_of_fit`) and the difference table.

Extrapolated rows (`domain == "extrapolation"`) are predictions outside the fitted range.
Read them alongside the confidence interval, not as point estimates.

## What was verified before launch

- `train.sh` was run under bash for both arms, with `medrap-train`/`medrap-eval` replaced by
    wrappers that add Hydra's `--cfg job`. Both compose against the pinned MedRAP commit.
    Dims, seed, labels path, checkpoint path, per-N epochs/warmup and `eval_mode=test`
    resolve as intended.
- `aggregate_results.py` was run on a synthetic `outputs/` tree generated from a known law.
    It finds 30 configurations × 5 runs per arm and recovers the planted retrieval advantage.
- Not yet run on the cluster: `make_labels.sh` and `submit_sweep.sh`. The cluster paths are
    unverified.

## Known limitations

1. The two arms aren't parameter-matched (see [Parameter accounting](#parameter-accounting)).
2. The schedule isn't compute-matched across N (see [Training schedule](#training-schedule)).
3. Hyperparameters (lr, batch size, dropout) are tuned for full capacity and full data, and
    aren't re-tuned per cell. Small or large cells may be slightly mistuned, which would
    bend the curve at the edges.
4. Five points per axis is the minimum for a 5-parameter law. Expect wide exponent
    intervals, and add points before drawing conclusions from extrapolations.
5. The dim=128 cell differs from the earlier full-capacity baseline (1 head and 1 layer,
    vs. 4 heads and 2 layers).
