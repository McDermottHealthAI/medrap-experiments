"""Bias-recalibration check: is the test-loss gap the documents, or an unlearned base rate?

For every run with predictions from scripts/predict_probabilities.sh, fits one logit offset per
task on the tuning split (the bias that minimizes tuning BCE, all other weights frozen) and
reports held_out BCE three ways, each averaged over (patient, task) pairs like test/loss:

- ``raw``: the model as trained (should reproduce the logged test/loss).
- ``recal``: after the per-task tuning-fit offset. If the arms' gap shrinks here, the original
  gap was mostly "has the head learned each task's base rate yet", not discrimination.
- ``prior``: a constant prediction of each task's train-split positive rate (the floor any
  model must beat).

Writes outputs/bias_check.parquet and prints per-cell means and paired retrieval - patient_only
differences.

Usage:  cd mimic_iv_scaling_laws && .venv/bin/python scripts/bias_check.py
"""

import glob
import re
from pathlib import Path

import numpy as np
import polars as pl

RUN_RE = re.compile(
    r"predictions/(?P<arm>\w+?)_dim(?P<dim>\d+)_trainN(?P<n>\d+)_tasks(?P<t>\d+)/draw(?P<draw>\d+)/"
)
EPS = 1e-7


def bce(p: np.ndarray, y: np.ndarray) -> float:
    """Mean binary cross-entropy over the non-NaN entries of ``y``."""
    m = ~np.isnan(y)
    p = np.clip(p[m], EPS, 1 - EPS)
    return float(-(y[m] * np.log(p) + (1 - y[m]) * np.log(1 - p)).mean())


def fit_offsets(z: np.ndarray, y: np.ndarray, iters: int = 50) -> np.ndarray:
    """Per-task logit offset b minimizing BCE(sigmoid(z + b), y) on tuning, by Newton's method."""
    b = np.zeros(z.shape[1])
    m = ~np.isnan(y)
    yy = np.nan_to_num(y)
    for _ in range(iters):
        s = 1 / (1 + np.exp(-(z + b)))
        g = ((s - yy) * m).sum(0) / m.sum(0)
        h = (s * (1 - s) * m).sum(0) / m.sum(0) + 1e-12
        b -= g / h
    return b


def labels(tasks_dir: Path, split: str) -> pl.DataFrame:
    return pl.read_parquet(tasks_dir / f"{split}.parquet")


def logged_loss(run: str, draw: int) -> float | None:
    files = sorted(glob.glob(f"outputs/{run}_test/draw{draw}/loggers/csv/*/metrics.csv"))
    if not files:
        return None
    col = pl.read_csv(files[-1])["test/loss"].drop_nulls()
    return float(col[-1]) if col.len() else None


def one_run(path: str) -> dict:
    m = RUN_RE.search(path)
    arm, dim, n, t, draw = m["arm"], int(m["dim"]), int(m["n"]), int(m["t"]), int(m["draw"])
    tasks_dir = Path(f"data/tasks_n{t}_30d_trainN{n}/draw{draw}/tasks")
    cols = [f"task_{i}" for i in range(t)]
    preds = pl.read_parquet(path).select("subject_id", "prediction_time", "probabilities")

    def split_arrays(split: str) -> tuple[np.ndarray, np.ndarray]:
        df = labels(tasks_dir, split).join(preds, on=["subject_id", "prediction_time"], how="inner")
        p = np.array(df["probabilities"].to_list(), dtype=np.float64)
        y = df.select(cols).to_numpy().astype(np.float64)
        return p, y

    p_tu, y_tu = split_arrays("tuning")
    p_ho, y_ho = split_arrays("held_out")
    z_tu, z_ho = (np.log(np.clip(p, EPS, 1 - EPS) / (1 - np.clip(p, EPS, 1 - EPS))) for p in (p_tu, p_ho))
    b = fit_offsets(z_tu, y_tu)
    prior = labels(tasks_dir, "train").select(pl.col(cols).mean()).to_numpy()[0]
    run = f"{arm}_dim{dim}_trainN{n}_tasks{t}"
    return {
        "arm": arm,
        "dim": dim,
        "n_train": n,
        "num_tasks": t,
        "draw": draw,
        "n_held_out": len(y_ho),
        "loss_logged": logged_loss(run, draw),
        "loss_raw": bce(p_ho, y_ho),
        "loss_recal": bce(1 / (1 + np.exp(-(z_ho + b))), y_ho),
        "loss_prior": bce(np.broadcast_to(prior, y_ho.shape), y_ho),
        "mean_pred": float(np.nanmean(p_ho)),
        "pos_rate": float(np.nanmean(y_ho)),
        "mean_abs_offset": float(np.abs(b).mean()),
    }


def main() -> None:
    rows = [one_run(p) for p in sorted(glob.glob("outputs/predictions/*/draw*/probabilities.parquet"))]
    df = pl.DataFrame(rows)
    df.write_parquet("outputs/bias_check.parquet")
    metrics = [
        "loss_logged",
        "loss_raw",
        "loss_recal",
        "loss_prior",
        "mean_pred",
        "pos_rate",
        "mean_abs_offset",
    ]
    with pl.Config(tbl_rows=-1, tbl_cols=-1, float_precision=4, tbl_width_chars=220):
        print(f"{df.height} runs\n")
        print(
            df.group_by("arm", "dim", "n_train")
            .agg(pl.col(metrics).mean(), pl.len().alias("draws"))
            .sort("dim", "n_train", "arm")
        )
        paired = (
            df.filter(pl.col("arm") == "retrieval")
            .join(
                df.filter(pl.col("arm") == "patient_only"),
                on=["dim", "n_train", "num_tasks", "draw"],
                suffix="_po",
            )
            .with_columns(
                (pl.col("loss_raw") - pl.col("loss_raw_po")).alias("diff_raw"),
                (pl.col("loss_recal") - pl.col("loss_recal_po")).alias("diff_recal"),
            )
            .group_by("dim", "n_train")
            .agg(
                pl.col("diff_raw").mean(),
                pl.col("diff_recal").mean(),
                pl.col("diff_recal").std().alias("diff_recal_std"),
                (pl.col("diff_recal") < 0).sum().alias("retrieval_wins_recal"),
                pl.len().alias("draws"),
            )
            .sort("dim", "n_train")
        )
        print("\nretrieval - patient_only, held_out BCE (negative = retrieval better):")
        print(paired)


if __name__ == "__main__":
    main()
