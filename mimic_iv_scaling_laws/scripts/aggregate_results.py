"""Fit a scaling law per arm from this experiment's outputs/ and compare the arms.

Reads test/auroc/mean from every outputs/<arm>_dim<D>_train<P>pct_test/draw<d> eval,
fits ``AUROC ~ E + A * N^-alpha + B * D^-beta`` per arm with simple-scaling-laws
(N = patient-encoder parameter count, D = training rows), and compares the arms at
every grid point plus 10x beyond the largest model / dataset.

Writes outputs/scaling_laws_results.parquet (one row per eval, the fit input) and
outputs/scaling_laws_comparison.parquet (fitted AUROC + p_better_than_reference per
arm per point), and prints both fitted laws and the comparison.

Usage:  cd mimic_iv_scaling_laws && .venv/bin/python scripts/aggregate_results.py
"""

import csv
import glob
import itertools
import re
from pathlib import Path

import polars as pl
from medrap.model.encoders import TimeDeltaRoPEPatientEncoder
from simple_scaling_laws import compare, fit

RUN_RE = re.compile(r"outputs/(?P<arm>\w+?)_dim(?P<dim>\d+)_train(?P<pct>\d+)pct_test/draw(?P<draw>\d+)$")
REFERENCE_ARM = "patient_only"
N_DRAWS = 1000


def test_auroc(eval_dir: str) -> float:
    files = sorted(glob.glob(f"{eval_dir}/loggers/csv/*/metrics.csv"))
    rows = [r for r in csv.DictReader(open(files[-1])) if r.get("test/auroc/mean")]
    return float(rows[-1]["test/auroc/mean"])


def encoder_n_params(dim: int) -> int:
    """Same encoder shape as train.sh: 1 head, 1 layer, ff_dim = 2 * dim."""
    encoder = TimeDeltaRoPEPatientEncoder(
        vocab_size=65536, embedding_dim=dim, num_heads=1, num_layers=1, ff_dim=2 * dim, dropout=0.0
    )
    return sum(p.numel() for p in encoder.parameters())


def n_train_rows(pct: int, draw: int) -> int:
    train = f"data/tasks_n25_30d_train{pct}pct/draw{draw}/tasks/train.parquet"
    return pl.scan_parquet(train).select(pl.len()).collect().item()


def collect_results() -> pl.DataFrame:
    rows = []
    for eval_dir in sorted(glob.glob("outputs/*_test/draw*")):
        m = RUN_RE.search(eval_dir)
        if m is None:
            continue
        arm, dim, pct, draw = m["arm"], int(m["dim"]), int(m["pct"]), int(m["draw"])
        rows.append(
            {
                "arm": arm,
                "pct": pct,
                "training_run_id": f"{arm}_dim{dim}_train{pct}pct_draw{draw}",
                "train_set_id": f"train{pct}pct_draw{draw}",
                # Each draw samples its own 25 tasks, so held_out labels differ per draw.
                "test_set_id": f"held_out_draw{draw}",
                "optimizer_seed": draw,
                "model_size__n_params": encoder_n_params(dim),
                "n_train_rows": n_train_rows(pct, draw),
                "test_metric__auroc": test_auroc(eval_dir),
            }
        )
    # Draws at one percentage differ by a few rows; give them one shared dataset size
    # so simple-scaling-laws treats them as repeats of a single configuration.
    return (
        pl.DataFrame(rows)
        .with_columns(dataset_size__n_train_rows=pl.col("n_train_rows").mean().over("pct").round())
        .drop("n_train_rows")
    )


def main() -> None:
    results = collect_results()
    results.write_parquet("outputs/scaling_laws_results.parquet")
    print(results.group_by("arm").len().sort("arm"))

    models = {
        arm: fit(results.filter(pl.col("arm") == arm).drop("arm", "pct"), n_draws=N_DRAWS, seed=0)
        for arm in sorted(results["arm"].unique())
    }
    for arm, model in models.items():
        print(f"\n=== {arm} ===")
        print(model.summary())

    n_params = sorted(results["model_size__n_params"].unique())
    n_rows = sorted(results["dataset_size__n_train_rows"].unique())
    n_params.append(10 * n_params[-1])
    n_rows.append(10 * n_rows[-1])
    grid = list(itertools.product(n_params, n_rows))
    points = {
        "point_id": [f"N{n}_D{d}" for n, d in grid],
        "model_size__n_params": [n for n, _ in grid],
        "dataset_size__n_train_rows": [d for _, d in grid],
    }
    comparison = compare(models, points, target="test_metric__auroc", reference=REFERENCE_ARM)
    comparison.write_parquet("outputs/scaling_laws_comparison.parquet")
    retrieval_vs_reference = (
        comparison.filter(pl.col("system") != REFERENCE_ARM)
        .select(
            pl.col("model_size__n_params", "dataset_size__n_train_rows").cast(pl.Int64),
            "domain",
            "difference__q025",
            "difference__median",
            "difference__q975",
            "p_better_than_reference",
        )
        .sort("model_size__n_params", "dataset_size__n_train_rows")
    )
    print(f"\n=== AUROC difference vs {REFERENCE_ARM} ===")
    with pl.Config(tbl_rows=-1, tbl_cols=-1, float_precision=4):
        print(retrieval_vs_reference)


if __name__ == "__main__":
    Path("outputs").mkdir(exist_ok=True)
    main()
