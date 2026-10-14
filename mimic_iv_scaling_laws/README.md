# mimic_iv_scaling_laws

Scaling laws for retrieval vs. patient-only, fit jointly over model capacity and
number of training patients on the N=25, 30d MIMIC-IV tasks:

```
AUROC ~ E + A * N^-alpha + B * D^-beta     (fit per arm, on the logit scale)
```

`N` = patient-encoder parameter count, `D` = training patients (`train.parquet` has one
row per patient, ~153k at full size). Fitting and arm comparison
are done by [`simple-scaling-laws`](https://github.com/mmcdermott/simple-scaling-laws)
(wild cluster bootstrap over configurations, paired draws for comparisons). The
question is whether retrieval's AUROC advantage over `patient_only` shrinks, holds,
or grows as `N` and `D` scale.

For the reasoning behind each choice (models, configs, grid, schedule, and known
limitations), see [`DESIGN.md`](DESIGN.md).

The previous capacity sweep (`mimic_iv_sweep/*capacity_sweep*`) and data-scarcity sweep
(`mimic_iv_sweep/*data_scarcity*`) each varied one axis with the other fixed. This
experiment runs the full cross product so both exponents can be fit together.

## Grid

Defined in [`scripts/grid.sh`](scripts/grid.sh):

| Axis | Values |
|---|---|
| arm | `patient_only`, `retrieval` (marginalized binary, learned-linear query, K=4) |
| `encoder.embedding_dim` | 4, 8, 16, 32, 64, 128 (num_heads=1, num_layers=1, ff_dim=2*dim, max_seq_len=256) |
| training patients | 1,000, 3,000, 10,000, 30,000, 100,000 (exact count, same for every draw) |
| draw | 1-5 (each draw has its own 25 tasks; seed = draw) |

2 × 6 × 5 = 60 SLURM arrays of 5 tasks = 300 training runs. Each task trains, then
evaluates `last.ckpt` on `held_out` in the same job.

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

scripts/make_labels.sh      # subsampled train label dirs (exact patient counts) from mimic_iv_sweep's full N=25 30d labels
scripts/submit_sweep.sh     # one sbatch per (arm, dim, n_patients)

# after every job finishes
.venv/bin/python scripts/aggregate_results.py
```

To run a single cell: `sbatch --job-name=sl-retrieval-dim16-trainN10000 scripts/train.sh retrieval 16 10000`.

`aggregate_results.py` writes `outputs/scaling_laws_results.parquet` (fit input, one
row per test eval) and `outputs/scaling_laws_comparison.parquet` (fitted AUROC and
`p_better_than_reference` per arm at every grid point, plus points 10x past the
largest model and dataset). It prints both fitted laws and the retrieval − patient_only
AUROC difference table.

## Notes

- `N` counts only the patient encoder. The retrieval arm also has fusion and
    retrieval-encoder parameters, which stay fixed across the grid, so they shift the
    retrieval arm's curve but are not part of the capacity axis.
- Subsets are drawn independently per (size, draw), so the 1k set is not contained in
    the 3k set. tuning/held_out are full-size and identical across sizes within a draw.
- With one test eval per run, evaluation noise can't be separated from run-to-run noise.
    `simple-scaling-laws` warns about this, and the run-level variance absorbs it.
