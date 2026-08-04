# mimic_iv_ablation_ladder

A **4-column ablation ladder** on MIMIC-IV: one table that separates "does
retrieval help?" from "does *selecting* documents help?" from "do document
*contents* help?".

The published tables in [`../mimic_iv_sweep/`](../mimic_iv_sweep/) and
[`../mimic_iv_sweep_frequent/`](../mimic_iv_sweep_frequent/) could not answer
any of those questions.
[`NULL_RESULT_DIAGNOSIS.md`](../mimic_iv_sweep_frequent/NULL_RESULT_DIAGNOSIS.md)
found three reasons: the metric was too noisy to resolve the effect (single
last-value readout, 6,400-row validation prefix, ~2 positives per task), the
retriever had collapsed in all 32 marginalized runs (`effective_k_mean` ~ 1.0,
which makes the marginalization arithmetically a no-op), and no arm was ever
replicated, so no significance could be computed. Six fixes have since landed.
This directory reruns the comparison with all six and produces the table the
original was trying to be.

All four columns are scored on the MEDS **held_out** split by
`medrap-eval eval_mode=test` -- one split, one protocol, no train/eval
discrepancy in how documents are combined:

| Column | Arm | What it is |
| --- | --- | --- |
| `patient_only` | separately trained | RoPE encoder -> masked-mean pooling -> head. No retriever at all (`fusion=passthrough`). |
| `marginalized (binary)` | separately trained | Retrieval-augmented, `k=4` real retrieved documents, per-document prediction marginalized independently per task (`marginalized_output_mode=binary`). |
| `frozen_random` | **same checkpoint** as `marginalized` | Weights frozen; `retriever.ablation_mode=random_docs` swaps the retrieved documents for random ones **at inference only**. 3 reps (the draw is unseeded). |
| `frozen_null` | **same checkpoint** as `marginalized` | Weights frozen; the corpus is a hand-built 16-row content-free index (`build_null_retrieval_db.py`) -- all-zero key embeddings, zero attention mask, placeholder tokens. Deterministic, so one rep. |

### The three comparisons

| # | comparison | structure | question it answers |
| --- | --- | --- | --- |
| C1 | `marginalized` − `patient_only` | **between-arm** (separately trained) | does retrieval help at all? |
| C2 | `frozen_random` − `marginalized` | **within-checkpoint** (weights frozen, docs swapped) | is retrieval *selecting* useful documents, or just adding machinery? |
| C3 | `frozen_null` − `marginalized` | **within-checkpoint**, deterministic | do document *contents* contribute anything? |

C2 and C3 are the tight ones: they hold every weight fixed and change only what
the retriever hands the fusion layer, so the difference is attributable to the
documents and nothing else. C1 compares two separately trained architectures and
carries the confounds listed under [Caveats](#caveats). C2 and C3 reuse the C1
checkpoints, so they cost ~9% of the total compute.

**Grid:** 2 code selections x 5 model seeds x 8 N values = 40 paired cells per
selection, per column. `MODEL_SEEDS=(1001 2002 3003 4004 5005)` (distinct from
the `seed=42` used for label draws elsewhere in this repo, and from the
`101/202/...` draw seeds of the variance studies), `NS=(1 2 4 8 16 32 64 128)`,
`SELECTIONS=(random frequent)`.

## The six fixes this experiment depends on

| # | fix | where it appears |
| --- | --- | --- |
| 1 | `marginalized_score_similarity=cosine` -- bounds document scores to [-1, 1] so the softmax over the k retrieved documents cannot saturate and `effective_k` cannot collapse to 1 | marginalized **training** script and **all three** marginalized eval scripts; never in `patient_only` |
| 2 | `frozen_random_docs` / `frozen_null_docs` as real ablations | `retriever.ablation_mode=random_docs` and the null corpus, both via `medrap-eval` against a frozen checkpoint |
| 3 | full validation split + 5 model seeds + a fixed readout | `training.trainer.limit_val_batches=1.0` (full tuning split, not a 200-batch prefix) with `training.trainer.val_check_interval=0.5` to offset the cost |
| 4 | content-free null corpus | `scripts/build_null_retrieval_db.py`, 16 rows (clears every `retriever.k` used here), verified at k=4 and k=10 |
| 5 | `anchor_strategy=uniform_event` -- prediction anchors land on real clinical events instead of uniformly over a subject's lifetime | both label-generation scripts |
| 6 | the balance check is a real gate, not a report | `python scripts/check_task_balance.py "${OUTPUT_DIR}/tasks" --min-positives 25 --quiet` after `medrap-preprocess`, under `set -euo pipefail`, so a bad label config aborts the array before any GPU time is spent |

Plus `training.datamodule.num_workers=8` on every training script. The default is
`null` -> 0 while the scripts already request `--cpus-per-task=16` that sit idle.
It cannot change any number: `seq_sampling_strategy=to_end` has no RNG in the
subsequence path, the label join is keyed on `(subject_id, prediction_time)`, and
`seed_everything(workers=True)` seeds worker RNGs regardless.

### The MedRAP pin

Fix 5 is not on MedRAP `main`. `pyproject.toml` pins it by SHA:

```toml
dependencies = [
    "medrap @ git+file:///groups/mm6677_gp/zzw2102/MedRAP@878c03818bb7aaedbd0a673d2586bf91ac1c9f1d",
]
```

| | |
| --- | --- |
| repo | `/groups/mm6677_gp/zzw2102/MedRAP` |
| branch | `feat/anchor-strategy-uniform-event` (based on `0a4a1fc` = `origin/experiment/reject-degenerate-codes-v4`, **not** `main`) |
| SHA | `878c03818bb7aaedbd0a673d2586bf91ac1c9f1d` |

**The branch is local only -- it has no upstream and is not on `origin`, so this
pin is unreproducible for anyone but the owner of that checkout, and it breaks if
the repo moves.** Push the branch, then change `file:///groups/mm6677_gp/zzw2102/MedRAP`
to `https://github.com/McDermottHealthAI/MedRAP.git` with the **same SHA** -- an
identical artifact, a one-token edit. Because the branch builds on `0a4a1fc`
rather than `main`, the eventual PR belongs against that lineage.

A `uv.lock` is committed here (none of the sibling experiment dirs have one), so
the 13 transitive dependencies are pinned too, not just `medrap`.

Verify the pin actually carries the feature after `uv sync`:

```bash
python -c "import medrap.preprocess.task_generation as t, inspect; print('anchor_strategy' in inspect.signature(t.generate_tasks).parameters)"
```

This must print `True`. If it prints `False`, the sweep would silently generate
`uniform_lifetime` labels. (`PreprocessDatasetAppConfig` is a structured
dataclass, so passing `anchor_strategy=` against an older pin is a hard Hydra
struct error rather than a silent ignore -- but that only protects the label
stage, not the case where the pin is right and the flag is never passed.)

## Run instructions

**Scale note:** one selection is **320 GPU array tasks** (80 training + 240
eval) plus 8 CPU label jobs, ~124 GPU-hours at the measured ~1.0 h
(`patient_only`) / ~1.8 h (`marginalized`) per-run baseline. Both selections is
double that. Every array is throttled to `%12` on purpose: measured on this
account, one job alone runs at 32 batch/s, ten jobs at 117 batch/s aggregate,
and thirty-six jobs at 134 batch/s -- aggregate throughput saturates around
~130 batch/s, so going 36-wide buys **+15%** while tripling exposure to the 8 h
`--time` wall (a past array lost 10 of 20 tasks that way). Contention, not
configuration, is the binding constraint. Run one selection at a time: `random`
first gives a complete 40-cell table in half the wall-clock, with no loss of
statistical power.

```bash
cd mimic_iv_ablation_ladder
uv sync && mkdir -p logs        # logs/ must exist BEFORE sbatch -- SLURM opens the log file first

./run_all.sh --selection random     # returns in seconds, prints every job id
# walk away

./run_all.sh --selection frequent   # the second table, once the first is through
```

`run_all.sh` submits every array with `--dependency` links so SLURM sequences the
whole pipeline unattended, and makes aggregation the last job, so `RESULTS.md`
lands on disk without anyone watching. `afterok` on the label job means a failed
`check_task_balance.py` gate stops the sweep instead of producing a table from
bad labels; `afterany` on the evals plus each eval script's
`[ ! -f "${CHECKPOINT_PATH}" ]` guard means one dead training cell is a clean
skip, not a corrupt row.

**The array range is the selection.** `SEL_IDX = IDX / 40` (or `/ 120` for the
random-docs eval), so `0-39` is `random` and `40-79` is `most_frequent` -- same
scripts, run twice, and `--selection` just picks the range.

### Checking on it when you get back

```bash
squeue -u "$USER"                   # empty == done
cat RESULTS.md                      # table + coverage header + effective_k gate
sacct -j <array_job_id> --format=JobID,State,Elapsed | grep -v COMPLETED
```

`RESULTS.md` opens with a coverage header: expected cells (40), cells found, and
every missing `(seed, N)` listed by name. A table built from 12 of 40 cells says
so out loud.

**Watch the first `marginalized` job's `metrics.csv`.** If `effective_k_mean` is
still ~1.0, fix 1 did not take, the retrieval arm is collapsed, and C2/C3 are
meaningless -- cancel and investigate rather than spending the rest of the
window. The aggregator re-checks this and reports an `effective_k_mean >= 2.0`
gate per run.

### What runs

| Script | Array | Purpose |
| --- | --- | --- |
| `scripts/prepare_retrieval.sh` | 1 | FAISS index + HF dataset over 125,847 `MedRAG/textbooks` chunks -> `data/retrieval_db`. Up to 3 h GPU, one-time, independent of labels. **The long pole** -- running it once ahead of everything drops the chain from ~8 h to ~5 h. |
| `scripts/build_null_retrieval_db.py` | inline | 16-row content-free corpus -> `data/retrieval_db_null`. Seconds, CPU, run directly (not via sbatch). |
| `scripts/generate_labels_random.sh` | 0-7 | `anchor_strategy=uniform_event` + `code_selection=random` -> `data/tasks_random/n{N}/tasks` |
| `scripts/generate_labels_frequent.sh` | 0-7 | `anchor_strategy=uniform_event` + `code_selection=most_frequent` -> `data/tasks_frequent/n{N}/tasks` |
| `scripts/sweep_patient_only_ladder.sh` | 0-79 | train arm 1 |
| `scripts/sweep_marginalized_binary_ladder.sh` | 0-79 | train arm 2 |
| `scripts/eval_ladder_patient_only.sh` | 0-79 | **column 1** |
| `scripts/eval_ladder_real_docs.sh` | 0-79 | **column 2** |
| `scripts/eval_ladder_random_docs.sh` | 0-239 | **column 3**, `ablation_mode=random_docs`, 3 reps |
| `scripts/eval_ladder_null_docs.sh` | 0-79 | **column 4** |
| `scripts/aggregate_ladder.sh` | 1 | runs `aggregate_ladder.py`, writes `RESULTS.md` |

Array index arithmetic, documented as an explicit table in each script header:

```bash
# 80-task scripts: 2 selections x 5 seeds x 8 N
SEL_IDX=$((IDX / 40)); REM=$((IDX % 40)); SEED_IDX=$((REM / 8)); N_IDX=$((REM % 8))

# 240-task random-docs eval: + 3 reps
SEL_IDX=$((IDX / 120)); REM=$((IDX % 120)); SEED_IDX=$((REM / 24))
REM2=$((REM % 24)); N_IDX=$((REM2 / 3)); REP_IDX=$((REM2 % 3))
```

Each script accepts extra arguments forwarded as Hydra overrides, e.g.
`sbatch scripts/sweep_patient_only_ladder.sh training.trainer.max_epochs=5`.

### Output layout

```
mimic_iv_ablation_ladder/
├── data/
│   ├── retrieval_db/                  # FAISS index + HF dataset (prepare_retrieval.sh)
│   ├── retrieval_db_null/             # 16-row content-free corpus (build_null_retrieval_db.py)
│   ├── tasks_random/n{N}/tasks/       # {train,tuning,held_out}.parquet + code_index.json
│   └── tasks_frequent/n{N}/tasks/
├── logs/                              # SLURM stdout/stderr per array task
├── outputs/
│   ├── patient_only_ladder/{random,frequent}/seed{S}/n{N}/          # checkpoints
│   ├── marginalized_binary_ladder/{random,frequent}/seed{S}/n{N}/   # checkpoints
│   ├── eval_patient_only/{random,frequent}/seed{S}/n{N}/            # column 1
│   ├── eval_real_docs/{random,frequent}/seed{S}/n{N}/               # column 2
│   ├── eval_random_docs/{random,frequent}/seed{S}/n{N}/rep{1,2,3}/  # column 3
│   └── eval_null_docs/{random,frequent}/seed{S}/n{N}/               # column 4
└── RESULTS.md                         # written by aggregate_ladder.sh
```

W&B run names mirror the output path, all in the `medrap` project:
`patient-only-ladder-{selection}-seed{S}-n{N}-*`,
`marginalized-binary-ladder-{selection}-seed{S}-n{N}-*`,
`eval-patient-only-*`, `eval-real-docs-*`, `eval-random-docs-*-rep{R}-*` and
`eval-null-docs-*`.

Aggregation runs three paired comparisons against the shared `real_docs`
baseline and assembles the 4-column table:

```bash
python scripts/aggregate_ladder.py --split test --rep-component '(seed|rep)\d+'
```

`--split test` is mandatory: `medrap-eval eval_mode=test` runs log `test/auroc/*`
and no `val/*` column at all, so the default `--split val` would find nothing.
The rep regex must collapse **both** dimensions -- it is `re.fullmatch`ed per
path component, and the four arms have different directory depths
(`.../seed*/n*` vs `.../seed*/n*/rep*`), so `'rep\d+'` alone leaves `seed1001` in
the cell key and unmatched cells are **silently ignored**, yielding a quietly
empty table rather than an error. `seed` (training noise) and `rep` (the unseeded
inference-time random-document draw) measure different things; both are kept.

## Architecture & hyperparameters

Both arms share: RoPE patient encoder, 3 training epochs, `lr=1e-3`,
`warmup_steps=200`, batch size 32, `max_seq_len=256`,
`seq_sampling_strategy=to_end`, `gradient_clip_val=1.0`,
`limit_val_batches=1.0` (full tuning split) with `val_check_interval=0.5`,
`num_workers=8`, and 5 model seeds (1001, 2002, 3003, 4004, 5005).

| Stage | `patient_only` | `marginalized` (binary) |
| --- | --- | --- |
| Encoder | `TimeDeltaRoPEPatientEncoder` -- vocab 65536, embed dim 128, 4 heads, 2 layers, ff dim 256 | same |
| Query projector | unused (`fusion=passthrough` discards it; `query_projector.in_dim=128` is still required because `model.forward()` calls it unconditionally) | `SequenceMeanQueryProjector` -- in 128, out 1024 |
| Retriever | none | `hf_dataset` (FAISS) -- `k=4`, corpus = `MedRAG/textbooks` |
| Retrieval encoder | none | `TokenFeatureRetrievalEncoder` -- vocab 151936, embed dim 64 |
| Fusion | `PassthroughFusion` (no retrieval, no dropout) | `PerDocCrossAttentionFusion` -- d_model 256, 8 heads, ff dim 512, 2 layers, dropout 0.1 |
| Document scoring | n/a | `marginalized_score_similarity=cosine` (**not** the dot product used before `NULL_RESULT_DIAGNOSIS.md`) |
| Head | `LinearHead`, in 128, out N | `LinearHead`, in 256, out N |
| Loss | `MultiTaskBCELoss` | `MultiTaskBCEMarginalizedLoss` (marginalizes per-task sigmoid over the 4 retrieved docs) |

`marginalized_score_similarity` is a plain Python attribute, not a `state_dict`
entry, so an eval that omits it loads the checkpoint cleanly and then silently
scores documents under a different rule than training used. It is set in the
marginalized training script and in all three marginalized eval scripts, and is
absent from `patient_only` everywhere.

The `patient_only` eval deliberately does **not** override `query_projector`,
`retriever` or `retrieval_encoder` as Hydra *groups*: `load_state_dict` runs with
`strict=True`, and training leaves those three at their config defaults, so the
checkpoint literally contains `model.query_projector.linear.weight [4, 128]` and
the `InMemoryRetriever`'s persistent buffers. Overriding the group would build a
differently-shaped module against trained weights and fail. Only
`query_projector.in_dim=128`, `head.in_dim=128`, `fusion=passthrough` and
`training/loss=multitask_binary_bce` are passed.

## Results

*(Pending -- no cell in either table has been run yet. `aggregate_ladder.sh`
writes `RESULTS.md`; transcribe it here.)*

Conventions, matching the sibling sweeps: 4-decimal values, each linked to its
W&B run; deltas always explicitly signed to 4 dp (`+0.0012`, `-0.0036`,
`+0.0000`); `valid tasks` as `n_valid/n_total`. Each cell is the mean over the 5
model seeds; `Δ C1` = marginalized − patient_only, `Δ C2` = frozen_random −
marginalized, `Δ C3` = frozen_null − marginalized. A negative `Δ C2`/`Δ C3` means
the real documents were worth something.

### Average AUROC -- random-code labels (`code_selection=random`)

**What "valid task" means:** a sampled task counts as valid only if its
held-out validation split contains at least one positive *and* one
negative example -- AUROC is undefined otherwise, so those tasks are
excluded from the mean rather than scored as 0.

| N | patient_only | marginalized (binary) | frozen_random | frozen_null | Δ C1 | Δ C2 | Δ C3 | valid tasks |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 |  |  |  |  |  |  |  |  |
| 2 |  |  |  |  |  |  |  |  |
| 4 |  |  |  |  |  |  |  |  |
| 8 |  |  |  |  |  |  |  |  |
| 16 |  |  |  |  |  |  |  |  |
| 32 |  |  |  |  |  |  |  |  |
| 64 |  |  |  |  |  |  |  |  |
| 128 |  |  |  |  |  |  |  |  |

**Δ mean ± std across the 8 N values:**

| Comparison | mean Δ | std Δ |
| --- | --- | --- |
| C1 (marginalized − patient_only) |  |  |
| C2 (frozen_random − marginalized) |  |  |
| C3 (frozen_null − marginalized) |  |  |

Also record the **realized cross-seed σ** per arm here: the old σ = 0.011
included estimator noise these labels largely remove, but training variance does
not shrink with better labels and is currently unmeasured. `random` draws are
independent across N (verified: the N=8 and N=16 code sets are disjoint), so the
pooled paired test over N is sound for this table.

**Takeaway:** *(pending -- compare the effect size against the cross-seed σ, not
against zero.)*

### Average AUROC -- most-frequent-code labels (`code_selection=most_frequent`)

**What "valid task" means:** a sampled task counts as valid only if its
held-out validation split contains at least one positive *and* one
negative example -- AUROC is undefined otherwise, so those tasks are
excluded from the mean rather than scored as 0.

| N | patient_only | marginalized (binary) | frozen_random | frozen_null | Δ C1 | Δ C2 | Δ C3 | valid tasks |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 |  |  |  |  |  |  |  |  |
| 2 |  |  |  |  |  |  |  |  |
| 4 |  |  |  |  |  |  |  |  |
| 8 |  |  |  |  |  |  |  |  |
| 16 |  |  |  |  |  |  |  |  |
| 32 |  |  |  |  |  |  |  |  |
| 64 |  |  |  |  |  |  |  |  |
| 128 |  |  |  |  |  |  |  |  |

**Δ mean ± std across the 8 N values:**

| Comparison | mean Δ | std Δ |
| --- | --- | --- |
| C1 (marginalized − patient_only) |  |  |
| C2 (frozen_random − marginalized) |  |  |
| C3 (frozen_null − marginalized) |  |  |

`most_frequent` selection is **nested** across N (the N=8 codes are a subset of
the N=16 codes), so per-N results are correlated and pooling across N overstates
significance for this table -- report per-N means and the cross-seed σ, and treat
any pooled p-value as optimistic. In exchange it carries roughly 8x the positives
per task (median prevalence 0.206 vs 0.026), so each individual cell is far less
noisy.

**Takeaway:** *(pending.)*

## Caveats

**These labels are not comparable to any existing `data/tasks` tree in this
repo.** `anchor_strategy=uniform_event` places each subject's prediction anchor
on a real clinical event rather than uniformly over their lifetime. Measured
end-to-end on the transformed cohort, that raises the positive rate by **91.9x**
(0.0022 -> 0.2057 on `most_frequent`), moves 554/554 sampled anchors onto a real
event timestamp (0 at birth, minimum 18.5 years post-birth), and drops ~17% of
otherwise-eligible subjects for having no clinical event inside the window. Every
AUROC in this directory is measured against a substantially different -- and much
better posed -- prediction problem than the tables in `mimic_iv_sweep` and
`mimic_iv_sweep_frequent`. Do not put them in the same table or the same
sentence. For the same reason, these numbers are not comparable to any run
predating the cosine document scoring or the full-validation protocol, since
those change the trained weights and the readout respectively.

**C1 is not a clean retrieval ablation.** `patient_only` and `marginalized`
differ in more than the presence of a retriever: `head.in_dim` is 128 vs 256, the
fusion stack adds dropout 0.1 where the patient-only path has none, the
marginalized model has ~2.2x the parameters, and the two are trained under
different losses (`MultiTaskBCELoss` vs `MultiTaskBCEMarginalizedLoss`). So C1
answers "does this retrieval architecture beat this patient-only architecture",
not "does retrieval help this architecture". **C2 and C3 are the clean
comparisons** -- they load the *same checkpoint*, freeze every weight, and change
only the documents handed to the fusion layer, so nothing but the documents can
account for the difference. Weight C1 accordingly when reading the table: a large
C1 with a near-zero C2 and C3 means the extra capacity did the work, not
retrieval.

**A collapsed retriever makes C2 and C3 vacuous.** If `effective_k_mean` is ~1.0,
the marginalization is arithmetically a no-op and swapping documents cannot move
the score in an interpretable way. The aggregator reports this gate per run;
treat any run below `effective_k_mean >= 2.0` as uninterpretable rather than as
evidence of a null effect.

**`frozen_random` is 3 reps of an unseeded draw.** `medrap-eval` never calls
`seed_everything` and `_eval.yaml` has no `seed` field, so the random-document
draw differs run to run by construction. Compare the single deterministic
`marginalized` and `frozen_null` numbers against the mean ± sd of those reps, not
against any one of them.

**Every script here is executed for the first time by this run.** No GPU smoke
test was run. Hydra composition errors and bad paths fail in seconds and cost
little; a `load_state_dict` shape mismatch on the `patient_only` eval or an
aggregator cell-key mismatch fail only *after* training completes and cost the
whole window. The CPU-only pre-flight (state_dict compatibility for all four
arms, aggregator cell-key alignment on a stub tree, `medrap-train --cfg job
--resolve` for all six job types, the pin check above, `bash -n` on every script)
takes ~3 minutes and targets exactly that second class. Run it before
`run_all.sh`.
