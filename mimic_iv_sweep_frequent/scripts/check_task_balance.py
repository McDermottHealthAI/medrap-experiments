"""Check per-task, per-split positive rates for degenerate or off-band tasks.

Two severities:
  DEGENERATE (critical) -- rate is exactly 0 or 1, i.e. single-class on that
    split. AUROC is undefined and the task gets dropped from val/auroc/mean
    entirely. This should never happen post- McDermottHealthAI/MedRAP#89 /
    #graceful-degradation-fix -- every selected code is guaranteed non-
    degenerate (>=1 positive and >=1 negative) on every split it's checked on.
    If you see this, something upstream is wrong (e.g. splits arg mismatch
    between generate_labels.sh's --splits and what _sample_task_codes checked).
  OUTSIDE IDEAL BAND (informational) -- rate is outside [--min-rate, --max-rate]
    but not degenerate. Expected for some tasks when the ideal band was
    undersupplied and the fallback filled slots with the closest non-degenerate
    codes (see generate_tasks log output for how many were relaxed). Not a bug,
    just a signal that task has less learning signal than the ideal band tasks.

Usage:
    python scripts/check_task_balance.py data/tasks/n8/tasks [--min-rate 0.01] [--max-rate 0.5]
"""

import argparse
import json
from pathlib import Path

import polars as pl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tasks_dir", type=Path, help="e.g. data/tasks/n8/tasks")
    parser.add_argument("--min-rate", type=float, default=0.01, help="ideal-band lower bound")
    parser.add_argument("--max-rate", type=float, default=0.5, help="ideal-band upper bound")
    args = parser.parse_args()

    code_index = json.loads((args.tasks_dir / "code_index.json").read_text())
    degenerate: set[str] = set()
    off_band: set[str] = set()

    for split in ("train", "tuning", "held_out"):
        path = args.tasks_dir / f"{split}.parquet"
        if not path.exists():
            continue
        df = pl.read_parquet(path)
        task_cols = [c for c in df.columns if c.startswith("task_")]
        rates = df.select([pl.col(c).mean().alias(c) for c in task_cols]).row(0, named=True)

        print(f"\n=== {split} (n={len(df)}) ===")
        for col in sorted(task_cols, key=lambda c: int(c.split("_")[1])):
            idx = col.split("_")[1]
            code = code_index.get(idx, col)
            rate = rates[col]
            flag = ""
            if rate is None or rate <= 0.0 or rate >= 1.0:
                flag = "  <-- DEGENERATE (single-class! should not happen)"
                degenerate.add(code)
            elif rate < args.min_rate or rate > args.max_rate:
                flag = "  <-- outside ideal band"
                off_band.add(code)
            print(f"  {col:>10} ({code:40s}) pos_rate={rate!s:>10s}{flag}")

    print()
    if degenerate:
        print(f"{len(degenerate)} task(s) DEGENERATE (single-class on some split) -- investigate:")
        for code in sorted(degenerate):
            print(f"  - {code}")
    else:
        print("No degenerate tasks -- every task has both classes on every split. Good.")

    if off_band:
        print(f"\n{len(off_band)} task(s) outside the ideal [{args.min_rate}, {args.max_rate}] band")
        print("(expected if generate_tasks logged a fallback warning; not necessarily a problem):")
        for code in sorted(off_band):
            print(f"  - {code}")


if __name__ == "__main__":
    main()
