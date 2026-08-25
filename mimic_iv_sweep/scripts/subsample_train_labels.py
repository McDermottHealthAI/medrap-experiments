"""Subsample the train split of a generated task-labels directory, leaving
tuning/held_out untouched.

Motivation: the capacity-starved retrieval experiment (results/capacity_starved_retrieval/)
tested whether retrieval helps once a smaller *model* can't represent the task
on its own. This subsamples the *training data* instead -- same idea, orthogonal
axis: does retrieval help once the model doesn't have enough *examples* to learn
the task on its own, even at full model capacity?

Only train.parquet is subsampled (uniformly at random by subject, fixed seed
per draw for reproducibility). tuning.parquet and held_out.parquet are copied
byte-for-byte unchanged, so every subsample level is evaluated against the
exact same validation/test data -- only the training signal shrinks.

code_index.json and metadata.json are copied unchanged too (the task-code
selection and anchor sampling that produced them is unaffected by subsampling
after the fact).

Usage:
    python scripts/subsample_train_labels.py \
        data/tasks_zach_uniform_event_n25_30d/draw1/tasks \
        data/tasks_zach_uniform_event_n25_30d_train10pct/draw1/tasks \
        --fraction 0.10 --seed 101
"""

import argparse
import shutil
from pathlib import Path

import polars as pl


def subsample_train_labels(
    *,
    source_tasks_dir: Path,
    output_tasks_dir: Path,
    fraction: float,
    seed: int,
) -> None:
    """Write a copy of ``source_tasks_dir`` with ``train.parquet`` subsampled.

    Args:
        source_tasks_dir: Directory containing the full-size
            ``{train,tuning,held_out}.parquet``, ``code_index.json``, and
            ``metadata.json``.
        output_tasks_dir: Directory to write the subsampled copy to.
        fraction: Fraction of train subjects to keep, in ``(0, 1]``.
        seed: Random seed for the subsample draw.

    Raises:
        ValueError: If ``fraction`` is not in ``(0, 1]``.
    """
    if not (0.0 < fraction <= 1.0):
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")

    output_tasks_dir.mkdir(parents=True, exist_ok=True)

    train = pl.read_parquet(source_tasks_dir / "train.parquet")
    n_keep = max(1, round(train.height * fraction))
    subsampled = train.sample(n=n_keep, seed=seed, shuffle=True)
    subsampled.write_parquet(output_tasks_dir / "train.parquet")
    print(f"train.parquet: {train.height} -> {subsampled.height} rows (fraction={fraction})")

    for name in ("tuning.parquet", "held_out.parquet", "code_index.json", "metadata.json"):
        src = source_tasks_dir / name
        if src.exists():
            shutil.copy2(src, output_tasks_dir / name)
            print(f"copied {name} unchanged")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_tasks_dir", type=Path, help="e.g. data/tasks_.../draw1/tasks")
    parser.add_argument("output_tasks_dir", type=Path, help="e.g. data/tasks_..._train10pct/draw1/tasks")
    parser.add_argument("--fraction", type=float, required=True, help="fraction of train subjects to keep")
    parser.add_argument("--seed", type=int, required=True, help="subsample random seed")
    args = parser.parse_args()

    subsample_train_labels(
        source_tasks_dir=args.source_tasks_dir,
        output_tasks_dir=args.output_tasks_dir,
        fraction=args.fraction,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
