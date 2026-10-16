"""Fit the scaling laws from this experiment's outputs/ and summarize the task-count ablation.

Reads test/auroc/mean from every outputs/<arm>_dim<D>_trainN<P>_tasks<T>_test/draw<d> eval.

- Main grid (T = MAIN_NUM_TASKS): fits ``AUROC ~ E + A * N^-alpha + B * D^-beta`` per arm with
  simple-scaling-laws (N = patient-encoder parameter count, D = training patients) and compares
  the arms at every grid point plus 10x beyond the largest model / dataset.
- Task-count ablation (dim = ABLATION_DIM, every T): mean +- std test AUROC over draws per
  (T, training patients, arm), and the retrieval - patient_only difference.

Writes outputs/scaling_laws_results.parquet (one row per eval), outputs/scaling_laws_comparison.parquet
(main-grid fitted AUROC + p_better_than_reference per arm per point) and
outputs/task_count_ablation.parquet, and prints the fitted laws and both tables.
"""

import csv
import glob
import itertools
import re
from pathlib import Path

import polars as pl
from medrap.model.encoders import TimeDeltaRoPEPatientEncoder
from simple_scaling_laws import compare, fit

RUN_RE = re.compile(r"outputs/(?P<arm>\w+?)_dim(?P<dim>\d+)_trainN(?P<n_train>\d+)_tasks(?P<num_tasks>\d+)_test/draw(?P<draw>\d+)$")
REFERENCE_ARM = "patient_only"
MAIN_NUM_TASKS = 25  # keep in sync with scripts/grid.sh
ABLATION_DIM = 32
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


def collect_results() -> pl.DataFrame:
    rows = []
    for eval_dir in sorted(glob.glob("outputs/*_test/draw*")):
        m = RUN_RE.search(eval_dir)
        if m is None:
            continue
        arm, dim, n_train, draw = m["arm"], int(m["dim"]), int(m["n_train"]), int(m["draw"])
        num_tasks = int(m["num_tasks"])
        rows.append(
            {
                "arm": arm,
                "num_tasks": num_tasks,
                "training_run_id": f"{arm}_dim{dim}_trainN{n_train}_tasks{num_tasks}_draw{draw}",
                "train_set_id": f"trainN{n_train}_tasks{num_tasks}_draw{draw}",
                # Each (task count, draw) samples its own tasks, so held_out labels differ.
                "test_set_id": f"held_out_tasks{num_tasks}_draw{draw}",
                "optimizer_seed": draw,
                "model_size__n_params": encoder_n_params(dim),
                "dataset_size__n_train_patients": n_train,
                "test_metric__auroc": test_auroc(eval_dir),
            }
        )
    return pl.DataFrame(rows)


def fit_and_compare(results: pl.DataFrame) -> pl.DataFrame:
    """Fit each arm on one task count's results and compare it with the reference arm."""
    models = {
        arm: fit(results.filter(pl.col("arm") == arm).drop("arm", "num_tasks"), n_draws=N_DRAWS, seed=0)
        for arm in sorted(results["arm"].unique())
    }
    for arm, model in models.items():
        print(f"\n=== {arm} ===")
        print(model.summary())

    n_params = sorted(results["model_size__n_params"].unique())
    n_patients = sorted(results["dataset_size__n_train_patients"].unique())
    n_params.append(10 * n_params[-1])
    n_patients.append(10 * n_patients[-1])
    grid = list(itertools.product(n_params, n_patients))
    points = {
        "point_id": [f"N{n}_D{d}" for n, d in grid],
        "model_size__n_params": [n for n, _ in grid],
        "dataset_size__n_train_patients": [d for _, d in grid],
    }
    return compare(models, points, target="test_metric__auroc", reference=REFERENCE_ARM)


def task_count_ablation(results: pl.DataFrame) -> pl.DataFrame:
    """Mean/std test AUROC over draws per (task count, training patients, arm) at ABLATION_DIM."""
    per_arm = (
        results.filter(pl.col("model_size__n_params") == encoder_n_params(ABLATION_DIM))
        .group_by("num_tasks", "dataset_size__n_train_patients", "arm")
        .agg(
            pl.col("test_metric__auroc").mean().alias("auroc_mean"),
            pl.col("test_metric__auroc").std().alias("auroc_std"),
            pl.len().alias("n_draws"),
        )
    )
    wide = per_arm.pivot(
        on="arm", index=["num_tasks", "dataset_size__n_train_patients"], values="auroc_mean"
    )
    if {"retrieval", REFERENCE_ARM} <= set(wide.columns):
        wide = wide.with_columns((pl.col("retrieval") - pl.col(REFERENCE_ARM)).alias("retrieval_minus_ref"))
    return per_arm.join(wide, on=["num_tasks", "dataset_size__n_train_patients"]).sort(
        "num_tasks", "dataset_size__n_train_patients", "arm"
    )


def main() -> None:
    results = collect_results()
    results.write_parquet("outputs/scaling_laws_results.parquet")
    print(results.group_by("num_tasks", "arm").len().sort("num_tasks", "arm"))

    print(f"\n##### Main grid: {MAIN_NUM_TASKS} tasks #####")
    comparison = fit_and_compare(results.filter(pl.col("num_tasks") == MAIN_NUM_TASKS))
    comparison.write_parquet("outputs/scaling_laws_comparison.parquet")
    retrieval_vs_reference = (
        comparison.filter(pl.col("system") != REFERENCE_ARM)
        .select(
            pl.col("model_size__n_params", "dataset_size__n_train_patients").cast(pl.Int64),
            "domain",
            "difference__q025",
            "difference__median",
            "difference__q975",
            "p_better_than_reference",
        )
        .sort("model_size__n_params", "dataset_size__n_train_patients")
    )
    ablation = task_count_ablation(results)
    ablation.write_parquet("outputs/task_count_ablation.parquet")
    with pl.Config(tbl_rows=-1, tbl_cols=-1, float_precision=4):
        print(f"\n=== AUROC difference vs {REFERENCE_ARM} (main grid) ===")
        print(retrieval_vs_reference)
        print(f"\n=== Task-count ablation (dim={ABLATION_DIM}) ===")
        print(ablation)


if __name__ == "__main__":
    Path("outputs").mkdir(exist_ok=True)
    main()
