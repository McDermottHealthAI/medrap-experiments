"""Check whether "code already occurred before the prediction anchor" alone
predicts the label almost as well as the trained model's AUROC.

This is a zero-training diagnostic for task difficulty: for each task code,
compute a trivial binary "history-positive" score per labeled subject (did
this code occur at any time strictly before prediction_time in that split's
raw event data?), then compute the AUC that trivial score alone would achieve
against the real label. If this trivial-score AUC is close to the trained
model's val/auroc/mean, the task is mostly "is this patient already on/being
monitored for X" (recurring/standing orders, chronic meds, scheduled labs),
not genuine forecasting of a new event -- the model may not be learning much
beyond history-echoing.

AUC for a binary (0/1) score is computed via the closed-form for two-point
ROC curves (equivalent to the Mann-Whitney U statistic with tie-averaging):
    AUC = (a*d + 0.5*a*c + 0.5*b*d) / ((a+b)*(c+d))
where a = #(score=1, y=1), b = #(score=0, y=1), c = #(score=1, y=0), d = #(score=0, y=0).

Usage:
    python scripts/check_history_leakage.py data/tasks/n250 /groups/mm6677_gp/data/MIMIC_MEDS/MEDS_cohort/intermediate
"""

import argparse
import json
from pathlib import Path

import polars as pl


def _history_auc(history_positive: pl.Series, label: pl.Series) -> tuple[float, int, int, int, int] | None:
    a = int(((history_positive == 1) & (label == 1)).sum())
    b = int(((history_positive == 0) & (label == 1)).sum())
    c = int(((history_positive == 1) & (label == 0)).sum())
    d = int(((history_positive == 0) & (label == 0)).sum())
    n_pos, n_neg = a + b, c + d
    if n_pos == 0 or n_neg == 0:
        return None
    auc = (a * d + 0.5 * a * c + 0.5 * b * d) / (n_pos * n_neg)
    return auc, a, b, c, d


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tasks_dir", type=Path, help="e.g. data/tasks/n250 (parent of tasks/)")
    parser.add_argument("meds_data_dir", type=Path, help="filtered MEDS root, e.g. .../MEDS_cohort/intermediate")
    parser.add_argument("--split", default="train", help="split to check (default: train)")
    args = parser.parse_args()

    tasks_subdir = args.tasks_dir / "tasks"
    code_index = json.loads((tasks_subdir / "code_index.json").read_text())

    labels_path = tasks_subdir / f"{args.split}.parquet"
    labels = pl.read_parquet(labels_path)
    task_cols = sorted(
        [c for c in labels.columns if c.startswith("task_")], key=lambda c: int(c.split("_")[1])
    )

    events_dir = args.meds_data_dir / "data" / args.split
    events = pl.scan_parquet(events_dir / "*.parquet").select(["subject_id", "code", "time"])

    print(f"=== history-leakage check: {args.split} split, {len(task_cols)} tasks, n_subjects={len(labels)} ===\n")

    aucs = []
    for col in task_cols:
        idx = col.split("_")[1]
        code = code_index.get(idx, col)

        code_events = events.filter(pl.col("code") == code).select(["subject_id", "time"])
        joined = (
            labels.lazy()
            .select(["subject_id", "prediction_time", col])
            .join(code_events, on="subject_id", how="left")
            .with_columns((pl.col("time") < pl.col("prediction_time")).fill_null(False).alias("occurred_before"))
            .group_by("subject_id", "prediction_time", col)
            .agg(pl.col("occurred_before").any().cast(pl.Int8).alias("history_positive"))
            .collect()
        )

        result = _history_auc(joined["history_positive"], joined[col])
        if result is None:
            print(f"  {col:>10} ({code:40s}) skipped (single-class)")
            continue
        auc, a, b, c, d = result
        aucs.append((code, auc))
        print(
            f"  {col:>10} ({code:40s}) history-only AUC={auc:.4f}  "
            f"[hist+/label+={a} hist-/label+={b} hist+/label-={c} hist-/label-={d}]"
        )

    if aucs:
        mean_auc = sum(a for _, a in aucs) / len(aucs)
        print(f"\nMean history-only AUC across {len(aucs)} tasks: {mean_auc:.4f}")
        print("Compare this to the trained model's val/auroc/mean for the same run.")
        print("If they're close, the model is mostly re-deriving 'already on this' rather than forecasting.")


if __name__ == "__main__":
    main()
