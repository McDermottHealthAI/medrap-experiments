"""Scaling-law headroom vs retrieval value: the experiment's core question.

For every main-grid cell (width M, training patients N) of the baseline arm, fit
``y = A - B * x^-alpha`` (A <= 1, B, alpha > 0) separately along each axis, using only
what a practitioner at that cell would have:

- data axis: test AUROC vs training patients, sizes <= N at width M;
- capacity axis: test AUROC vs patient-encoder parameters, widths <= M at N patients.

Each fit needs >= 3 sizes. The predicted headroom is ``A - mean observed AUROC`` at the cell,
with a 95% interval from refitting on draws resampled with replacement at every x. Headroom
is then compared with what retrieval adds at the same cell: total value (retrieval arm -
baseline) and, where the null-docs control exists, knowledge value (retrieval - null docs).

Writes <outputs>/headroom_tasks<T>.parquet and <outputs>/headroom_tasks<T>_correlations.parquet and
prints both. A retrieval arm's value is only computed on cells where all N_DRAWS draws exist.

Usage:
    cd mimic_iv_scaling_laws && .venv/bin/python scripts/headroom.py [--tasks 625] [--outputs outputs]
        [--baseline patient_only] [--retrieval retrieval retrieval_qwen3]
    v2 grid:  .venv/bin/python scripts/headroom.py --outputs outputs_v2 --tasks T \
                  --retrieval retrieval_null_docs retrieval_qwen3
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import polars as pl
from aggregate_results import collect_results, encoder_n_params
from scipy import stats
from scipy.optimize import OptimizeWarning, curve_fit

# Plain widths plus the v2 multi-layer specs (WIDTHxLAYERSxHEADS).
CAPACITIES = ("4", "8", "16", "32", "64", "128", "128x2x4", "256x4x8")
N_BOOT = 200
N_DRAWS = 5
MIN_POINTS = 3


def law(x: np.ndarray, a: float, b: float, alpha: float) -> np.ndarray:
    return a - b * np.power(x, -alpha)


def fit_asymptote(x: np.ndarray, y: np.ndarray) -> float:
    """Asymptote A of y = A - B x^-alpha fit to (x, y); NaN if the fit fails."""
    xs = x / x.min()  # unit-free x keeps B well scaled
    p0 = (min(1.0, y.max() + 0.05), max(1e-3, y.max() - y.min()), 0.5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OptimizeWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        try:
            (a, _b, _alpha), _ = curve_fit(
                law, xs, y, p0=p0, bounds=([0.0, 0.0, 1e-3], [1.0, 10.0, 5.0]), maxfev=20000
            )
        except (RuntimeError, ValueError):
            return float("nan")
    return float(a)


def headroom(points: pl.DataFrame, x_col: str, rng: np.random.Generator) -> tuple[float, float, float]:
    """(headroom, lo, hi) at the largest x in ``points`` (columns x_col, draw, auroc)."""
    xs = sorted(points[x_col].unique())
    groups = [points.filter(pl.col(x_col) == x)["auroc"].to_numpy() for x in xs]
    x_all = np.concatenate([np.full(len(g), x, dtype=float) for x, g in zip(xs, groups, strict=True)])
    y_all = np.concatenate(groups)
    y_cell = groups[-1].mean()
    point = fit_asymptote(x_all, y_all) - y_cell
    boots = []
    for _ in range(N_BOOT):
        res = [rng.choice(g, size=len(g), replace=True) for g in groups]
        a = fit_asymptote(x_all, np.concatenate(res))
        if np.isfinite(a):
            boots.append(a - res[-1].mean())
    lo, hi = np.percentile(boots, [2.5, 97.5]) if len(boots) >= 20 else (np.nan, np.nan)
    return float(point), float(lo), float(hi)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tasks", type=int, default=625)
    parser.add_argument("--outputs", default="outputs")
    parser.add_argument("--baseline", default="patient_only")
    parser.add_argument("--retrieval", nargs="+", default=["retrieval", "retrieval_qwen3"])
    args = parser.parse_args()

    dims = {encoder_n_params(d): d for d in CAPACITIES}
    res = (
        collect_results(root=args.outputs)
        .filter(pl.col("num_tasks") == args.tasks)
        .select(
            "arm",
            pl.col("model_size__n_params").alias("params"),
            pl.col("model_size__n_params").replace_strict(dims).alias("dim"),
            pl.col("dataset_size__n_train_patients").alias("n"),
            pl.col("optimizer_seed").alias("draw"),
            pl.col("test_metric__auroc").alias("auroc"),
        )
    )
    base = res.filter(pl.col("arm") == args.baseline)
    rng = np.random.default_rng(0)
    rows = []
    for params, dim in sorted(base.select("params", "dim").unique().iter_rows()):
        for n in sorted(base["n"].unique()):
            row = {"dim": dim, "params": params, "n": n}
            data_pts = base.filter((pl.col("dim") == dim) & (pl.col("n") <= n))
            cap_pts = base.filter((pl.col("n") == n) & (pl.col("params") <= params))
            for name, pts, x_col in (("data", data_pts, "n"), ("capacity", cap_pts, "params")):
                ok = pts[x_col].n_unique() >= MIN_POINTS
                h, lo, hi = headroom(pts, x_col, rng) if ok else (np.nan, np.nan, np.nan)
                row |= {f"headroom_{name}": h, f"headroom_{name}_lo": lo, f"headroom_{name}_hi": hi}
            rows.append(row)
    out = pl.DataFrame(rows)

    def mean_auroc(arm: str) -> pl.DataFrame:
        return (
            res.filter(pl.col("arm") == arm)
            .group_by("dim", "n")
            .agg(pl.when(pl.len() == N_DRAWS).then(pl.col("auroc").mean()).alias(arm))
        )

    for arm in dict.fromkeys([args.baseline, *args.retrieval, "retrieval_null_docs"]):
        if res.filter(pl.col("arm") == arm).height:
            out = out.join(mean_auroc(arm), on=["dim", "n"], how="left")
    for arm in args.retrieval:
        if arm in out.columns:
            out = out.with_columns((pl.col(arm) - pl.col(args.baseline)).alias(f"value_{arm}"))
            if "retrieval_null_docs" in out.columns and arm != "retrieval_null_docs":
                out = out.with_columns(
                    (pl.col(arm) - pl.col("retrieval_null_docs")).alias(f"knowledge_{arm}")
                )
    # Reference: the naive gap 1 - AUROC needs no fit. If headroom correlates no better than this,
    # the scaling-law fit adds nothing beyond "how weak is the baseline here".
    out = out.with_columns((1 - pl.col(args.baseline)).alias("naive_gap"))
    out = out.with_columns(
        (
            pl.col("headroom_data").is_not_nan()
            & (pl.col("headroom_data") >= 1 - pl.col(args.baseline) - 1e-3)
        ).alias("data_fit_at_bound"),
        (
            pl.col("headroom_capacity").is_not_nan()
            & (pl.col("headroom_capacity") >= 1 - pl.col(args.baseline) - 1e-3)
        ).alias("capacity_fit_at_bound"),
    )
    out = out.sort("params", "n")
    Path(args.outputs).mkdir(exist_ok=True)
    out.write_parquet(Path(args.outputs) / f"headroom_tasks{args.tasks}.parquet")

    corr = []
    for h in ("headroom_data", "headroom_capacity", "naive_gap"):
        for v in [c for c in out.columns if c.startswith(("value_", "knowledge_"))]:
            d = out.select(h, v).drop_nulls().drop_nans()
            if d.height >= 4:
                rho, p = stats.spearmanr(d[h], d[v])
                corr.append(
                    {"headroom": h, "retrieval_metric": v, "cells": d.height, "spearman": rho, "p": p}
                )
    corr = pl.DataFrame(corr)
    corr.write_parquet(Path(args.outputs) / f"headroom_tasks{args.tasks}_correlations.parquet")
    with pl.Config(tbl_rows=-1, tbl_cols=-1, float_precision=3, tbl_width_chars=250):
        print(f"{args.tasks} tasks, baseline={args.baseline}\n")
        print(
            out.select(
                "dim",
                "n",
                "headroom_data",
                "headroom_data_lo",
                "headroom_data_hi",
                "headroom_capacity",
                "headroom_capacity_lo",
                "headroom_capacity_hi",
                *[c for c in out.columns if c.startswith(("value_", "knowledge_"))],
            )
        )
        print("\nSpearman correlation, headroom vs retrieval value across cells:")
        print(corr)
        print(
            f"\nfits with asymptote at the A=1 bound: data {out['data_fit_at_bound'].sum()}/"
            f"{out['headroom_data'].is_not_nan().sum()}, capacity {out['capacity_fit_at_bound'].sum()}/"
            f"{out['headroom_capacity'].is_not_nan().sum()}"
        )


if __name__ == "__main__":
    main()
