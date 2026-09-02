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

    # or target an exact absolute row count instead of a fraction:
    python scripts/subsample_train_labels.py \
        data/tasks_zach_uniform_event_n25_30d/draw1/tasks \
        data/tasks_zach_uniform_event_n25_30d_trainN100/draw1/tasks \
        --n-rows 100 --seed 201
"""

import argparse
import shutil
from pathlib import Path

import polars as pl


def subsample_train_labels(
    *,
    source_tasks_dir: Path,
    output_tasks_dir: Path,
    seed: int,
    fraction: float | None = None,
    n_rows: int | None = None,
) -> None:
    """Write a copy of ``source_tasks_dir`` with ``train.parquet`` subsampled.

    Args:
        source_tasks_dir: Directory containing the full-size
            ``{train,tuning,held_out}.parquet``, ``code_index.json``, and
            ``metadata.json``.
        output_tasks_dir: Directory to write the subsampled copy to.
        seed: Random seed for the subsample draw.
        fraction: Fraction of train subjects to keep, in ``(0, 1]``. Exactly
            one of ``fraction``/``n_rows`` must be given.
        n_rows: Exact number of train rows to keep. Exactly one of
            ``fraction``/``n_rows`` must be given.

    Raises:
        ValueError: If neither or both of ``fraction``/``n_rows`` are given,
            or if ``fraction`` is not in ``(0, 1]``.
    """
    if (fraction is None) == (n_rows is None):
        raise ValueError("exactly one of fraction or n_rows must be given")
    if fraction is not None and not (0.0 < fraction <= 1.0):
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")

    output_tasks_dir.mkdir(parents=True, exist_ok=True)

    train = pl.read_parquet(source_tasks_dir / "train.parquet")
    n_keep = n_rows if n_rows is not None else max(1, round(train.height * fraction))
    subsampled = train.sample(n=n_keep, seed=seed, shuffle=True)
    subsampled.write_parquet(output_tasks_dir / "train.parquet")
    target_desc = f"n_rows={n_rows}" if n_rows is not None else f"fraction={fraction}"
    print(f"train.parquet: {train.height} -> {subsampled.height} rows ({target_desc})")

    for name in ("tuning.parquet", "held_out.parquet", "code_index.json", "metadata.json"):
        src = source_tasks_dir / name
        if src.exists():
            shutil.copy2(src, output_tasks_dir / name)
            print(f"copied {name} unchanged")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_tasks_dir", type=Path, help="e.g. data/tasks_.../draw1/tasks")
    parser.add_argument("output_tasks_dir", type=Path, help="e.g. data/tasks_..._train10pct/draw1/tasks")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--fraction", type=float, help="fraction of train subjects to keep")
    target.add_argument("--n-rows", type=int, help="exact number of train rows to keep")
    parser.add_argument("--seed", type=int, required=True, help="subsample random seed")
    args = parser.parse_args()

    subsample_train_labels(
        source_tasks_dir=args.source_tasks_dir,
        output_tasks_dir=args.output_tasks_dir,
        fraction=args.fraction,
        n_rows=args.n_rows,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
