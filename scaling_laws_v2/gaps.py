"""Paired retrieval vs patient-only gaps in the v1 grid (results/scaling_laws/runs.csv).

    python scaling_laws_v2/gaps.py [RUNS_CSV]

One row per matched pair (same W, D, T, draw: same tasks, patients and seed). Prints, by W, by D and
overall:
    gap %        test loss of #2 / #1 - 1 (positive: retrieval is worse)
    #2 worse     share of pairs where #2 has the higher loss
    swap %       #2 test loss with a random document / with its top document - 1
and the #4 (random documents) pairs against #1 and #2 on the control slice.
"""

import sys
from pathlib import Path

import polars as pl

KEY = ["W", "D", "T", "draw"]


def arm(runs: pl.DataFrame, config: str, ev: str, name: str) -> pl.DataFrame:
    rows = runs.filter((pl.col("config") == config) & (pl.col("eval") == ev))
    return rows.select(*KEY, pl.col("loss").alias(name))


def summary(pairs: pl.DataFrame, by: list[str]) -> pl.DataFrame:
    return (
        pairs.group_by(by)
        .agg(
            pl.len().alias("pairs"),
            ((pl.col("l2") / pl.col("l1") - 1) * 100).mean().round(1).alias("gap %"),
            ((pl.col("l2") > pl.col("l1")).mean() * 100).round(0).alias("#2 worse %"),
            ((pl.col("swap") / pl.col("l2") - 1) * 100).mean().round(0).alias("swap %"),
        )
        .sort(pl.col(by[0]).cast(pl.String).str.extract(r"^(\d+)").cast(pl.Int64, strict=False), *by)
    )


def main(path: Path) -> None:
    runs = pl.read_csv(path)
    pairs = (
        arm(runs, "patient_only", "test", "l1")
        .join(arm(runs, "retrieval_learned_query", "test", "l2"), on=KEY)
        .join(arm(runs, "retrieval_learned_query", "test_random_doc", "swap"), on=KEY, how="left")
        .with_columns(pl.lit("all").alias("all"))
    )
    pl.Config.set_tbl_rows(20)
    pl.Config.set_tbl_hide_dataframe_shape(True)
    for by in (["W"], ["D"], ["all"]):
        print(summary(pairs, by))

    control = (
        arm(runs, "retrieval_random_docs", "test", "l4")
        .join(arm(runs, "patient_only", "test", "l1"), on=KEY)
        .join(arm(runs, "retrieval_learned_query", "test", "l2"), on=KEY)
    )
    print(
        control.select(
            pl.len().alias("pairs"),
            ((pl.col("l4") / pl.col("l1") - 1) * 100).mean().round(1).alias("#4 vs #1 %"),
            ((pl.col("l2") / pl.col("l4") - 1) * 100).mean().round(1).alias("#2 vs #4 %"),
        )
    )


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent.parent / "results/scaling_laws/runs.csv"))
