# mimic_iv_scaling_laws

Scaling laws for retrieval vs. patient-only, fit jointly over model capacity and
number of training patients, at several numbers of 30d MIMIC-IV tasks:

```
AUROC ~ E + A * N^-alpha + B * D^-beta     (fit per arm and task count, on the logit scale)
```

`N` = patient-encoder parameter count, `D` = training patients (`train.parquet` has one
row per patient, ~153k at full size). Fitting and arm comparison
are done by [`simple-scaling-laws`](https://github.com/mmcdermott/simple-scaling-laws)
(wild cluster bootstrap over configurations, paired draws for comparisons). The
question is whether retrieval's AUROC advantage over `patient_only` shrinks, holds,
or grows as `N` and `D` scale, and whether that depends on how many tasks the model
trains on.

The cohort and all task labels are rebuilt from the raw MEDS data by this experiment's
own scripts. Nothing is reused from other experiments' label or cohort directories.

For the reasoning behind each choice (models, configs, grid, data, schedule, and known
limitations), see [`DESIGN.md`](DESIGN.md).

The previous capacity sweep (`mimic_iv_sweep/*capacity_sweep*`), data-scarcity sweep
(`mimic_iv_sweep/*data_scarcity*`) and task-count ablation each varied one axis with the
others fixed. This experiment runs the full cross product.

## Grid

Defined in [`scripts/grid.sh`](scripts/grid.sh):

| Axis | Values |
|---|---|
| arm | `patient_only`, `retrieval` (marginalized binary, learned-linear query, K=4) |
| `encoder.embedding_dim` | 4, 8, 16, 32, 64, 128 (num_heads=1, num_layers=1, ff_dim=2*dim, max_seq_len=256) |
| training patients | 1,000, 3,000, 10,000, 30,000, 100,000 (exact count, same for every draw) |
| tasks | 25 random codes, 30-day horizon (`MAIN_NUM_TASKS`) |
| draw | 1-5 (task seed 101/202/303/404/505; training seed = draw) |

That's the main grid: 2 × 6 × 5 × 5 draws = 300 runs.

**Task-count ablation:** 5, 125, 625 and 2,500 tasks (`ABLATION_TASK_COUNTS`) at dim=32
(`ABLATION_DIM`), for both arms × all 5 training-set sizes × 5 draws, adds 200 runs. The
25-task point comes from the main grid.

The total is 500 runs (100 SLURM arrays of 5 draws). Each job trains, then evaluates
`last.ckpt` on `held_out` in the same job.

Epochs and warmup scale with the number of training patients, so small-data cells still
get enough optimizer steps. 1k/10k/100k follow `mimic_iv_sweep`'s `extreme_starved_*_trainN*`
schedule, and 3k/30k sit between them:

| Training patients | `max_epochs` | `warmup_steps` |
|---|---|---|
| 1,000 | 10 | 60 |
| 3,000 | 7 | 100 |
| 10,000 | 5 | 200 |
| 30,000 | 4 | 200 |
| 100,000 | 3 | 200 |

All other hyperparameters match the existing `mimic_iv_sweep` capacity-sweep scripts.

## Run order

```bash
cd mimic_iv_scaling_laws
uv sync                                                              # once
mkdir -p data
ln -s /groups/mm6677_gp/hs3627/medrap-experiments/mimic_iv_sweep/data/retrieval_db data/retrieval_db

scripts/submit_tasks.sh     # build_cohort.sh, then (afterok) one generate_tasks.sh array per task count
                            # its last line is SBATCH_DEPENDENCY=afterok:<task job ids>
SBATCH_DEPENDENCY=afterok:<ids> scripts/submit_sweep.sh   # queued behind the task jobs

# after every job finishes
.venv/bin/python scripts/aggregate_results.py
```

What the data steps write:

| Script | Output |
|---|---|
| `build_cohort.sh` | `data/cohort/intermediate` (MEDS-transforms filtered, `min_subjects_per_code=100`, `min_events_per_subject=10`) and `data/cohort/tensorized`, from the raw `/groups/mm6677_gp/data/MIMIC_MEDS/MEDS_cohort` |
| `generate_tasks.sh T` | `data/tasks_n<T>_30d/draw<d>/tasks` (full size) and `data/tasks_n<T>_30d_trainN<N>/draw<d>/tasks` for every N |

`scripts/submit_tasks.sh --skip-cohort` regenerates only the tasks, reusing an existing
`data/cohort`.

To run a single cell: `sbatch --job-name=sl-retrieval-dim16-trainN10000-tasks25 scripts/train.sh retrieval 16 10000 25`.

`aggregate_results.py` writes three files:

- `outputs/scaling_laws_results.parquet`: one row per test eval.
- `outputs/scaling_laws_comparison.parquet`: fitted AUROC and `p_better_than_reference`
  per arm at every 25-task grid point, plus points 10x past the largest model and dataset.
- `outputs/task_count_ablation.parquet`: mean and std AUROC over draws per (task count,
  training patients, arm) at dim=32, plus the retrieval − patient_only difference.

It prints both fitted laws and both tables.

## Notes

- `N` counts only the patient encoder. The retrieval arm also has fusion and
    retrieval-encoder parameters, which stay fixed across the grid, so they shift the
    retrieval arm's curve but are not part of the capacity axis.
- Subsets are drawn independently per (size, draw), so the 1k set is not contained in
    the 3k set. tuning/held_out are full-size and identical across sizes within a draw.
- Within a draw, every task count labels the same patients at the same prediction times
    and trains on the same subsampled patients. Only the task codes differ, and they are
    drawn independently per task count, not nested.
- With one test eval per run, evaluation noise can't be separated from run-to-run noise.
    `simple-scaling-laws` warns about this, and the run-level variance absorbs it.
