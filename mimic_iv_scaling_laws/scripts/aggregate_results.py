"""Fit the scaling laws from this experiment's outputs/ and summarize the task-count ablation.

Reads test/auroc/mean from every outputs/<arm>_dim<D>_trainN<P>_tasks<T>_test/draw<d> eval.

- Main grid (each T in MAIN_TASK_COUNTS): fits ``AUROC ~ E + A * N^-alpha + B * D^-beta`` per arm with
  simple-scaling-laws (N = patient-encoder parameter count, D = training patients) and compares
  the arms at every grid point plus 10x beyond the largest model / dataset.
- Task-count ablation (dim = ABLATION_DIM, every T): mean +- std test AUROC over draws per
  (T, training patients, arm), and the retrieval - patient_only difference.

Writes outputs/scaling_laws_results.parquet (one row per eval), outputs/scaling_laws_comparison.parquet
(main-grid fitted AUROC + p_better_than_reference per arm per point) and
outputs/task_count_ablation.parquet and outputs/knowledge_control.parquet (the retrieval arm
against its random-docs / null-docs controls), and prints the fitted laws and the tables.
"""

import csv
import glob
import itertools
import re
from pathlib import Path

import polars as pl
from medrap.model.encoders import TimeDeltaRoPEPatientEncoder
from simple_scaling_laws import compare, fit

RUN_RE = re.compile(
    r"outputs/(?P<arm>\w+?)_dim(?P<dim>\d+)_trainN(?P<n_train>\d+)_tasks(?P<num_tasks>\d+)_test/draw(?P<draw>\d+)$"
)
REFERENCE_ARM = "patient_only"
MAIN_TASK_COUNTS = (25, 625)  # keep in sync with scripts/grid.sh
MAIN_NUM_TASKS = MAIN_TASK_COUNTS[0]  # default for plot_results.py
ABLATION_DIM = 32
SCALING_ARMS = ("patient_only", "retrieval")
CONTROL_ARMS = ("retrieval_random_docs", "retrieval_null_docs")
CONTROL_NUM_TASKS = 625
N_DRAWS = 1000


def test_metrics(eval_dir: str) -> tuple[float, float]:
    """(test AUROC averaged over tasks, test loss) from an eval's CSV logger.

    Both arms log ``test/loss`` as binary cross-entropy against their final predicted probability
    (retrieval's is the marginal over its retrieved documents), so the two are on the same scale.
    """
    files = sorted(glob.glob(f"{eval_dir}/loggers/csv/*/metrics.csv"))
    with open(files[-1]) as f:
        rows = [r for r in csv.DictReader(f) if r.get("test/auroc/mean")]
    return float(rows[-1]["test/auroc/mean"]), float(rows[-1]["test/loss"])


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
                **dict(zip(("test_metric__auroc", "test_loss__bce"), test_metrics(eval_dir), strict=True)),
            }
        )
    return pl.DataFrame(rows)


def fit_and_compare(results: pl.DataFrame) -> pl.DataFrame:
    """Fit each scaling arm on one task count's results and compare it with the reference arm."""
    results = results.filter(pl.col("arm").is_in(SCALING_ARMS))
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
        results.filter(
            pl.col("model_size__n_params") == encoder_n_params(ABLATION_DIM),
            pl.col("arm").is_in(SCALING_ARMS),
        )
        .group_by("num_tasks", "dataset_size__n_train_patients", "arm")
        .agg(
            pl.col("test_metric__auroc").mean().alias("auroc_mean"),
            pl.col("test_metric__auroc").std().alias("auroc_std"),
            pl.len().alias("n_draws"),
        )
    )
    wide = per_arm.pivot(on="arm", index=["num_tasks", "dataset_size__n_train_patients"], values="auroc_mean")
    if {"retrieval", REFERENCE_ARM} <= set(wide.columns):
        wide = wide.with_columns((pl.col("retrieval") - pl.col(REFERENCE_ARM)).alias("retrieval_minus_ref"))
    return per_arm.join(wide, on=["num_tasks", "dataset_size__n_train_patients"]).sort(
        "num_tasks", "dataset_size__n_train_patients", "arm"
    )


def knowledge_control(results: pl.DataFrame) -> pl.DataFrame:
    """Mean/std test AUROC per arm on the control slice (CONTROL_NUM_TASKS, dims the controls ran at)."""
    control = results.filter(pl.col("num_tasks") == CONTROL_NUM_TASKS)
    params = control.filter(pl.col("arm").is_in(CONTROL_ARMS))["model_size__n_params"].unique()
    return (
        control.filter(pl.col("model_size__n_params").is_in(params))
        .group_by("arm", "model_size__n_params", "dataset_size__n_train_patients")
        .agg(
            pl.col("test_metric__auroc").mean().alias("auroc_mean"),
            pl.col("test_metric__auroc").std().alias("auroc_std"),
            pl.len().alias("n_draws"),
        )
        .sort("model_size__n_params", "dataset_size__n_train_patients", "arm")
    )


def main() -> None:
    results = collect_results()
    results.write_parquet("outputs/scaling_laws_results.parquet")
    print(results.group_by("num_tasks", "arm").len().sort("num_tasks", "arm"))

    comparisons = []
    for num_tasks in MAIN_TASK_COUNTS:
        main_grid = results.filter(pl.col("num_tasks") == num_tasks)
        if main_grid.filter(pl.col("arm").is_in(SCALING_ARMS))["arm"].n_unique() < len(SCALING_ARMS):
            print(f"\n##### Main grid: {num_tasks} tasks -- skipped, not every arm has results #####")
            continue
        print(f"\n##### Main grid: {num_tasks} tasks #####")
        comparisons.append(fit_and_compare(main_grid).with_columns(pl.lit(num_tasks).alias("num_tasks")))
    comparison = pl.concat(comparisons)
    comparison.write_parquet("outputs/scaling_laws_comparison.parquet")
    retrieval_vs_reference = (
        comparison.filter(pl.col("system") != REFERENCE_ARM)
        .select(
            "num_tasks",
            pl.col("model_size__n_params", "dataset_size__n_train_patients").cast(pl.Int64),
            "domain",
            "difference__q025",
            "difference__median",
            "difference__q975",
            "p_better_than_reference",
        )
        .sort("num_tasks", "model_size__n_params", "dataset_size__n_train_patients")
    )
    ablation = task_count_ablation(results)
    ablation.write_parquet("outputs/task_count_ablation.parquet")
    control = knowledge_control(results)
    control.write_parquet("outputs/knowledge_control.parquet")
    with pl.Config(tbl_rows=-1, tbl_cols=-1, float_precision=4):
        print(f"\n=== AUROC difference vs {REFERENCE_ARM} (main grids) ===")
        print(retrieval_vs_reference)
        print(f"\n=== Task-count ablation (dim={ABLATION_DIM}) ===")
        print(ablation)
        print(f"\n=== Knowledge-vs-capacity control ({CONTROL_NUM_TASKS} tasks) ===")
        print(control)


if __name__ == "__main__":
    Path("outputs").mkdir(exist_ok=True)
    main()
