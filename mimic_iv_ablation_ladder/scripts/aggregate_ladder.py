"""Assemble the ablation ladder's FOUR-column AUROC table from the four eval arms.

`aggregate_results.py` (copied into this directory from `mimic_iv_sweep_frequent/`) is
structurally a TWO-arm tool: one --baseline, one --treatment, one delta column. This
experiment needs four arms side by side, so this module imports that file's helpers --
run discovery, metrics.csv parsing, the acceptance gates, cell keying, the Student-t
routines and the formatters -- and drives THREE paired comparisons against the shared
`real_docs` baseline, then assembles the table itself:

    C1  baseline patient_only  vs treatment real_docs    does retrieval help at all?
    C2  baseline real_docs     vs treatment random_docs  is retrieval SELECTING useful docs,
                                                         or just adding machinery?
    C3  baseline real_docs     vs treatment null_docs    do document CONTENTS contribute?

Nothing about discovery, gating or cell-keying is reimplemented here; if the gates change
in `aggregate_results.py` they change here too.

Fixed protocol (not configurable, because getting either wrong fails silently)
------------------------------------------------------------------------------
* `--split test`. Every column is scored by `medrap-eval eval_mode=test` on the MEDS
  held_out split, and those runs emit `test/auroc/*` and NO `val/*` column at all -- the
  aggregator's default `--split val` would find nothing and print an empty table rather
  than an error. --split test also auto-suppresses the within-run volatility columns and
  the best-validation sensitivity readout, since each eval run logs exactly one row.
* rep component `(seed|rep)\\d+`. The regex is `re.fullmatch`ed against each path
  component, so `rep\\d+` alone would leave `seed1001` in the cell key and the arms would
  never match. Both dimensions must collapse: a table cell is one (selection, N), averaged
  over the 5 model seeds and -- for `random_docs` only -- the 3 inference-time reps.
  `seed` and `rep` measure different things (training noise vs. the unseeded random-document
  draw) and both are still reported separately: see the cross-seed sigma section and the
  per-(seed, N) secondary test.

Coverage
--------
Unmatched cells are SILENTLY IGNORED by the cell-keying logic, so a naming drift between
the eval scripts and this aggregator produces a quietly empty table rather than an error.
That is why the report opens with an explicit coverage block: expected (seed, N) cells per
arm, how many were found, which glob resolved them, and every missing combination listed by
name. A table built from 12 of 40 cells must visibly say so.

The report degrades gracefully. An arm with zero runs is reported as such in coverage, its
column prints `n/a`, its comparisons print "nothing to test", and the remaining columns are
still assembled.

Usage
-----
    python scripts/aggregate_ladder.py --selection random -o RESULTS.md
    python scripts/aggregate_ladder.py --selection frequent --outputs-root ./outputs

stdlib + numpy/pandas only -- the Student-t helpers come from `aggregate_results.py`, which
is itself scipy-free.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import importlib.util
import io
import math
import re
import sys
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------------------
# import aggregate_results.py by path, so this works from any cwd

_HERE = Path(__file__).resolve().parent


def _load_aggregate_results():
    path = _HERE / "aggregate_results.py"
    if not path.is_file():
        raise SystemExit(
            f"ERROR: {path} not found. Copy it from "
            f"mimic_iv_sweep_frequent/scripts/aggregate_results.py first."
        )
    spec = importlib.util.spec_from_file_location("aggregate_results", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["aggregate_results"] = module
    spec.loader.exec_module(module)
    return module


agg = _load_aggregate_results()

# --------------------------------------------------------------------------------------
# the grid, fixed by the experiment design

SPLIT = "test"
REP_COMPONENT = r"(seed|rep)\d+"

MODEL_SEEDS = (1001, 2002, 3003, 4004, 5005)
NS = (1, 2, 4, 8, 16, 32, 64, 128)
REPS = (1, 2, 3)
SELECTIONS = ("random", "frequent")

EXPECTED_CELLS = len(MODEL_SEEDS) * len(NS)  # 40 (seed, N) cells per arm

ARM_ORDER = ("patient_only", "real_docs", "random_docs", "null_docs")

# arm key -> (markdown column label, directory token, has a rep dimension)
ARMS = {
    "patient_only": ("patient_only", "patient_only", False),
    "real_docs": ("marginalized (binary)", "real_docs", False),
    "random_docs": ("frozen_random", "random_docs", True),
    "null_docs": ("frozen_null", "null_docs", False),
}

# (id, baseline arm, treatment arm, the question it answers, its pairing structure)
COMPARISONS = (
    ("C1", "patient_only", "real_docs",
     "does retrieval help at all?",
     "between-arm (separately trained models)"),
    ("C2", "real_docs", "random_docs",
     "is retrieval SELECTING useful documents, or just adding machinery?",
     "within-checkpoint (weights frozen, documents swapped)"),
    ("C3", "real_docs", "null_docs",
     "do document CONTENTS contribute anything?",
     "within-checkpoint, deterministic (content-free corpus)"),
)

GATE_DESCRIPTIONS = (
    "n_valid_tasks == n_tasks   -- no task in the label set was unscorable",
    "effective_k_mean >= 2.0    -- retrieval did not collapse onto a single document "
    "(checked only where the column exists, i.e. on the marginalized arms)",
    "top1_mode_frac <= 0.5      -- fewer than half of patients retrieve the same top "
    "document (a gate only where retrieval reaches the loss; informational elsewhere)",
)

_SEED_RE = re.compile(r"seed(\d+)")
_N_RE = re.compile(r"n(\d+)")
_REP_RE = re.compile(r"rep(\d+)")


# --------------------------------------------------------------------------------------
# glob resolution


def arm_glob_candidates(outputs_root: str, token: str, selection: str, has_rep: bool) -> list[str]:
    """Ordered candidate globs for one arm; the first that matches anything wins.

    The canonical layout, matching `EVAL_OUTPUT_DIR` in scripts/eval_ladder_*.sh, is

        <outputs_root>/eval_<token>/<selection>/seed<SEED>/n<N>[/rep<REP>]

    The training arms write to `patient_only_ladder/` and `marginalized_binary_ladder/`,
    which carry no `eval` in the name and are therefore never picked up here -- important,
    because those runs log `val/auroc/*` and no `test/*` column at all, so they would be
    read as gate failures rather than as the wrong arm.

    The remaining candidates absorb the plausible drifts -- an `eval_ladder_` prefix, the
    selection folded into the directory name instead of being its own path component, a
    differently ordered `eval`/`<token>` name -- so a rename in the eval scripts degrades
    into a reported fallback rather than a silently empty column.
    """
    tail = "seed*/n*" + ("/rep*" if has_rep else "")
    names = [
        f"eval_{token}",
        f"eval_ladder_{token}",
        f"eval_{token}_{selection}",
        f"eval_ladder_{token}_{selection}",
        f"*eval*{token}*",
        f"*{token}*eval*",
    ]
    out: list[str] = []
    for name in names:
        out.append(f"{outputs_root}/{name}/{selection}/{tail}")
        out.append(f"{outputs_root}/{name}/{tail}")
    if has_rep:
        # Last resort: the rep dimension is encoded somewhere other than its own directory.
        out.append(f"{outputs_root}/eval_{token}/{selection}/seed*/n*")
    seen: set[str] = set()
    unique = []
    for pattern in out:
        if pattern not in seen:
            seen.add(pattern)
            unique.append(pattern)
    return unique


def resolve_arm(patterns: list[str]) -> tuple[list[Path], str]:
    """Return (run dirs, the glob that produced them) for the first candidate that hits.

    `discover_runs` warns on stderr for every glob that matches nothing, which is the
    normal case while walking the candidate list -- so its stderr is swallowed here and the
    resolved pattern is reported in the coverage block instead.
    """
    for pattern in patterns:
        sink = io.StringIO()
        with contextlib.redirect_stderr(sink):
            run_dirs = agg.discover_runs([pattern])
        if run_dirs:
            return run_dirs, pattern
    return [], patterns[0] if patterns else "<none>"


# --------------------------------------------------------------------------------------
# coordinates and cells


def coords(run_dir: Path) -> tuple[int | None, int | None, int | None]:
    """(model seed, N, rep) read off the run directory's path components."""
    seed = n = rep = None
    for part in run_dir.parts:
        match = _SEED_RE.fullmatch(part)
        if match:
            seed = int(match.group(1))
        match = _N_RE.fullmatch(part)
        if match:
            n = int(match.group(1))
        match = _REP_RE.fullmatch(part)
        if match:
            rep = int(match.group(1))
    return seed, n, rep


def cell_n(cell: str) -> str:
    """Cell key -> the N shown in the table's first column ('random/n16' -> '16')."""
    for part in reversed(cell.split("/")):
        match = _N_RE.fullmatch(part)
        if match:
            return match.group(1)
    return agg.display_cell(cell)


def mean_final(runs: list) -> float:
    values = [r.final for r in runs if np.isfinite(r.final)]
    return float(np.mean(values)) if values else float("nan")


def per_seed_means(runs: list) -> dict[int, float]:
    """seed -> mean final AUROC over that seed's runs (averaging reps where present)."""
    grouped: dict[int, list] = {}
    for run in runs:
        seed, _, _ = coords(run.run_dir)
        if seed is not None:
            grouped.setdefault(seed, []).append(run)
    return {seed: mean_final(rs) for seed, rs in grouped.items()}


# --------------------------------------------------------------------------------------
# report sections


def print_header(selection: str, outputs_root: str) -> None:
    print(f"# Ablation ladder -- 4-column AUROC table (code selection: `{selection}`)")
    print()
    print(f"Generated : {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Outputs   : `{outputs_root}`")
    print(f"Readout   : final logged `{agg.M.headline}` "
          f"({agg.ARM_NOTE[agg.M.split]}).")
    print(f"Split     : `{SPLIT}` -- every column is `medrap-eval eval_mode=test` on the MEDS "
          f"held_out split.")
    print(f"Cell key  : (code selection, N), collapsing path components matching "
          f"`{REP_COMPONENT}` -- i.e. averaged over the {len(MODEL_SEEDS)} model seeds and, "
          f"for `frozen_random` only, the {len(REPS)} inference-time reps.")
    print()


def print_coverage(
    selection: str,
    arms: dict[str, list],
    resolved: dict[str, str],
) -> None:
    print("## Coverage")
    print()
    print(f"Grid: {len(SELECTIONS)} code selections x {len(MODEL_SEEDS)} model seeds x "
          f"{len(NS)} N values. This report covers selection=`{selection}` only.")
    print(f"Expected cells per arm: **{EXPECTED_CELLS}** (seed, N) "
          f"[`frozen_random` additionally x{len(REPS)} reps = "
          f"{EXPECTED_CELLS * len(REPS)} runs].")
    print()
    print(f"    model seeds : {', '.join(str(s) for s in MODEL_SEEDS)}")
    print(f"    N values    : {', '.join(str(n) for n in NS)}")
    print()

    rows = []
    missing_report: list[tuple[str, list[str]]] = []
    for key in ARM_ORDER:
        label, _, has_rep = ARMS[key]
        runs = arms[key]
        expected = {
            (seed, n) for seed in MODEL_SEEDS for n in NS
        }
        found_cells = set()
        found_runs = set()
        for run in runs:
            seed, n, rep = coords(run.run_dir)
            found_cells.add((seed, n))
            found_runs.add((seed, n, rep))
        missing = sorted(expected - found_cells)
        extra = sorted(found_cells - expected)
        expected_runs = EXPECTED_CELLS * (len(REPS) if has_rep else 1)
        if not runs:
            status = "**MISSING -- no runs found**"
        elif missing:
            status = f"**INCOMPLETE ({len(missing)} missing)**"
        elif len(found_runs) < expected_runs:
            status = f"partial reps ({len(found_runs)}/{expected_runs} runs)"
        else:
            status = "complete"
        rows.append([
            f"`{key}`",
            label,
            str(len(runs)),
            f"{len(found_cells)}/{EXPECTED_CELLS}",
            status,
            f"`{resolved[key]}`",
        ])
        if not runs:
            # Listing all 40 combinations for an arm that produced nothing is noise; the
            # actionable fact is that the glob resolved to nothing at all.
            missing_report.append(
                (key, [f"ALL {EXPECTED_CELLS} (seed, N) cells -- the arm produced no runs"])
            )
        elif missing:
            missing_report.append(
                (key, [f"(seed={seed}, N={n})" for seed, n in missing])
            )
        if extra:
            missing_report.append(
                (f"{key} UNEXPECTED", [f"(seed={seed}, N={n})" for seed, n in extra])
            )

    header = ["arm", "column", "runs", "(seed, N) cells", "status", "resolved glob"]
    print("| " + " | ".join(header) + " |")
    print("| " + " | ".join("---" for _ in header) + " |")
    for row in rows:
        print("| " + " | ".join(row) + " |")
    print()

    if missing_report:
        print("Missing / unexpected cells, listed explicitly:")
        print()
        for key, items in missing_report:
            print(f"* `{key}`: {', '.join(items)}")
        print()
    else:
        print("No missing cells: every arm covers the full "
              f"{len(MODEL_SEEDS)} x {len(NS)} grid.")
        print()

    empty = [k for k in ARM_ORDER if not arms[k]]
    if empty:
        print(f"**{len(empty)} arm(s) produced no runs at all: "
              f"{', '.join('`' + k + '`' for k in empty)}.** Their columns read `n/a` and "
              f"every comparison that involves them is skipped. The remaining columns below "
              f"are still assembled from what exists.")
        print()


def print_gates(arms: dict[str, list], verbose: bool) -> None:
    print("## Acceptance gates")
    print()
    print("A failing run is reported and excluded from the headline aggregate, but still "
          "appears in the table:")
    print()
    for line in GATE_DESCRIPTIONS:
        print(f"* `{line}`")
    print()
    print("```")
    for key in ARM_ORDER:
        label, _, _ = ARMS[key]
        runs = arms[key]
        if not runs:
            print(f"--- acceptance gates: {label} ---")
            print("    0 run(s) found -- nothing to gate.")
            continue
        if verbose:
            agg.print_run_table(label, runs, agg.common_root([r.run_dir for r in runs]), False)
        agg.print_gate_report(label, runs)
    print("```")
    print()


def print_table(arms_by_cell: dict[str, dict[str, list]], cells: list[str]) -> None:
    print("## Results")
    print()
    print("**What \"valid task\" means:** a sampled task counts as valid only if the scored "
          "split contains at least one positive *and* one negative example -- AUROC is "
          "undefined otherwise, so those tasks are excluded from the mean rather than "
          "scored as 0.")
    print()
    print("Comparisons (all three share the `marginalized (binary)` column as their "
          "reference):")
    print()
    for cid, base, treat, question, structure in COMPARISONS:
        base_label = ARMS[base][0]
        treat_label = ARMS[treat][0]
        print(f"* **{cid}** = `{treat_label}` − `{base_label}` -- {structure}; "
              f"{question}")
    print()

    header = [
        "N",
        "patient_only",
        "marginalized (binary)",
        "frozen_random",
        "frozen_null",
        "Δ C1",
        "Δ C2",
        "Δ C3",
        "valid tasks",
    ]
    print("| " + " | ".join(header) + " |")
    print("| " + " | ".join("---" for _ in header) + " |")

    def cell_value(arm: str, cell: str) -> float:
        runs = arms_by_cell[arm].get(cell, [])
        return mean_final(runs)

    def shown_delta(base: str, treat: str, cell: str) -> float:
        # Repo convention: the printed delta is the difference of the two DISPLAYED
        # 4-decimal values, so the table is internally consistent. The paired statistics
        # below run at full precision and can differ by 1 in the last digit.
        b, t = cell_value(base, cell), cell_value(treat, cell)
        if not (np.isfinite(b) and np.isfinite(t)):
            return float("nan")
        return round(t, 4) - round(b, 4)

    def valid_tasks(cell: str) -> str:
        seen: list[str] = []
        for arm in ARM_ORDER:
            for run in arms_by_cell[arm].get(cell, []):
                if run.valid_tasks_str not in seen:
                    seen.append(run.valid_tasks_str)
        if not seen:
            return "?/?"
        return ", ".join(sorted(seen))

    flagged: list[str] = []
    for cell in cells:
        row = [cell_n(cell)]
        row += [agg.fmt(cell_value(arm, cell)) for arm in ARM_ORDER]
        row += [
            agg.fmt_signed(shown_delta(base, treat, cell))
            for _, base, treat, _, _ in COMPARISONS
        ]
        row.append(valid_tasks(cell))
        print("| " + " | ".join(row) + " |")
        cell_runs = [r for arm in ARM_ORDER for r in arms_by_cell[arm].get(cell, [])]
        if cell_runs and not all(r.valid for r in cell_runs):
            flagged.append(cell_n(cell))

    print()
    print(f"Values are the mean final `{agg.M.headline}` over the "
          f"{len(MODEL_SEEDS)} model seeds (and over the {len(REPS)} inference reps for "
          f"`frozen_random`). Deltas are the difference of the two displayed 4-decimal "
          f"values (repo convention); the paired statistics below are computed at full "
          f"precision, so they can differ in the last digit.")
    if flagged:
        print()
        print(f"**Gate failures in cells: {', '.join(flagged)}** -- these rows are reported "
              f"for reproducibility but at least one of their runs failed an acceptance "
              f"gate; see the gate report above for the per-run reason. They are excluded "
              f"from the headline paired statistics below.")
    print()


def paired_stats(deltas: list[float], label: str, indent: str = "    ") -> None:
    n = len(deltas)
    print(f"{indent}{label}: n={n}")
    if n == 0:
        print(f"{indent}    nothing to test.")
        return
    arr = np.asarray(deltas, dtype=float)
    mean = float(arr.mean())
    if n == 1:
        print(f"{indent}    mean delta = {mean:+.4f}   sd = n/a   (single pair, no test)")
        return
    sd = float(arr.std(ddof=1))
    se = sd / math.sqrt(n)
    tstat = mean / se if se > 0 else float("nan")
    df = n - 1
    crit = agg.t_ppf(0.975, df)
    lo, hi = mean - crit * se, mean + crit * se
    print(f"{indent}    mean delta = {mean:+.4f}   sd = {sd:.4f}   se = {se:.4f}")
    print(f"{indent}    paired t({df}) = {tstat:.3f}   two-sided p = "
          f"{agg.t_two_sided_p(tstat, df):.4f}")
    print(f"{indent}    95% t-CI   = [{lo:+.4f}, {hi:+.4f}]")
    neg = int((arr < 0).sum())
    print(f"{indent}    {neg}/{n} blocks negative (treatment worse), "
          f"{n - neg}/{n} non-negative")


def print_comparisons(arms_by_cell: dict[str, dict[str, list]]) -> None:
    print("## Paired comparisons")
    print()
    print("The headline test blocks on N: one paired observation per (code selection, N) "
          "cell, each side already averaged over its seeds and reps. The secondary test "
          "un-collapses the seed dimension and pairs at (seed, N), which is the "
          f"{EXPECTED_CELLS}-pair structure the grid was sized for -- it has more power but "
          "assumes the seeds are exchangeable between the two arms, which holds for C2 and "
          "C3 (same checkpoint) and is an assumption for C1 (separately trained).")
    print()
    print("```")
    for cid, base, treat, question, structure in COMPARISONS:
        base_label, treat_label = ARMS[base][0], ARMS[treat][0]
        print(f"=== {cid}: {treat_label} - {base_label} ===")
        print(f"    question  : {question}")
        print(f"    structure : {structure}")

        base_cells, treat_cells = arms_by_cell[base], arms_by_cell[treat]
        cells = sorted(set(base_cells) & set(treat_cells), key=agg.sort_key)
        unmatched = sorted(set(base_cells) ^ set(treat_cells), key=agg.sort_key)
        print(f"    matched cells: {len(cells)}")
        if unmatched:
            print(f"    UNMATCHED cells (ignored): {', '.join(unmatched)}")
        if not cells:
            print("    nothing to test -- at least one arm has no matching runs.")
            print()
            continue

        deltas_all, deltas_valid = [], []
        for cell in cells:
            b, t = mean_final(base_cells[cell]), mean_final(treat_cells[cell])
            delta = t - b
            if not np.isfinite(delta):
                continue
            deltas_all.append(delta)
            if all(r.valid for r in base_cells[cell] + treat_cells[cell]):
                deltas_valid.append(delta)
        paired_stats(deltas_valid, "paired delta over N blocks, gate-passing only (HEADLINE)")
        paired_stats(deltas_all, "paired delta over N blocks, ALL matched (descriptive)")

        seed_deltas = []
        for cell in cells:
            base_seeds = per_seed_means(base_cells[cell])
            treat_seeds = per_seed_means(treat_cells[cell])
            for seed in sorted(set(base_seeds) & set(treat_seeds)):
                delta = treat_seeds[seed] - base_seeds[seed]
                if np.isfinite(delta):
                    seed_deltas.append(delta)
        paired_stats(seed_deltas, "paired delta over (seed, N) blocks, ALL matched (secondary)")
        print()
    print("```")
    print()


def print_cross_seed_sigma(arms_by_cell: dict[str, dict[str, list]]) -> None:
    print("## Realized cross-seed sigma")
    print()
    print("Training noise does not shrink when the labels improve, and it was unmeasured "
          "under these labels before this run. Below is the sd of the per-seed cell value "
          "across the model seeds, within each (selection, N) cell: `mean sd` averages "
          "those sds, `pooled sd` is the square root of the mean variance. Right-size any "
          "follow-up experiment against `pooled sd`, not against the old 0.011 figure "
          "(which included estimator noise these labels largely remove).")
    print()
    print("```")
    header = ["arm", "cells", "seeds/cell", "mean sd", "pooled sd", "min sd", "max sd"]
    rows = []
    for key in ARM_ORDER:
        label, _, _ = ARMS[key]
        sds, seed_counts = [], []
        for cell, runs in arms_by_cell[key].items():
            values = [v for v in per_seed_means(runs).values() if np.isfinite(v)]
            seed_counts.append(len(values))
            if len(values) > 1:
                sds.append(float(np.std(np.asarray(values, dtype=float), ddof=1)))
        if not sds:
            rows.append([label, str(len(arms_by_cell[key])), "n/a", "n/a", "n/a", "n/a", "n/a"])
            continue
        arr = np.asarray(sds, dtype=float)
        rows.append([
            label,
            str(len(sds)),
            f"{np.mean(seed_counts):.1f}",
            agg.fmt(float(arr.mean())),
            agg.fmt(float(math.sqrt(float((arr ** 2).mean())))),
            agg.fmt(float(arr.min())),
            agg.fmt(float(arr.max())),
        ])
    widths = [
        max(len(header[i]), *(len(r[i]) for r in rows)) if rows else len(header[i])
        for i in range(len(header))
    ]
    print("    " + "  ".join(h.ljust(w) for h, w in zip(header, widths)))
    print("    " + "  ".join("-" * w for w in widths))
    for row in rows:
        print("    " + "  ".join(c.ljust(w) for c, w in zip(row, widths)))
    print("```")
    print()


# --------------------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Always --split test and --rep-component '(seed|rep)\\d+'; both are fixed "
               "here because getting either wrong yields a quietly empty table.",
    )
    parser.add_argument("--selection", required=True, choices=list(SELECTIONS),
                        help="code selection to report; picks the labels/outputs branch")
    parser.add_argument("--outputs-root", default="./outputs",
                        help="root of the run output tree (default: ./outputs)")
    parser.add_argument("-o", "--output", default=None, metavar="RESULTS.md",
                        help="write the report to this file instead of stdout")
    parser.add_argument("--verbose", action="store_true",
                        help="also emit aggregate_results.py's per-run table for each arm")
    for key in ARM_ORDER:
        parser.add_argument(f"--{key.replace('_', '-')}-glob", action="append", default=[],
                            metavar="GLOB", dest=f"glob_{key}",
                            help=f"explicit run-directory glob for the {key} arm "
                                 f"(repeatable; overrides auto-resolution)")
    parser.add_argument("--gate-effective-k", type=float, default=2.0,
                        help="minimum retrieval/test/differentiable/effective_k_mean "
                             "(default 2.0)")
    parser.add_argument("--gate-top1-mode-frac", type=float, default=0.5,
                        help="maximum retrieval/test/top1_mode_frac (default 0.5)")
    # Accepted, but only at their one correct value. The eval scripts' headers quote the
    # invocation as `--split test --rep-component '(seed|rep)\d+'`, so those flags must not
    # be a usage error -- and passing anything else must not be silently honoured, because
    # either wrong value yields a quietly empty table rather than a failure.
    parser.add_argument("--split", default=SPLIT, choices=sorted(agg.SPLITS),
                        help=f"fixed at '{SPLIT}': every ladder column is medrap-eval "
                             f"eval_mode=test, and those runs emit no val/* column at all")
    parser.add_argument("--rep-component", default=REP_COMPONENT, metavar="REGEX",
                        help=f"fixed at '{REP_COMPONENT}': both the seed and the rep "
                             f"directory levels must collapse for the four arms to match")
    args = parser.parse_args(argv)
    if args.split != SPLIT:
        parser.error(
            f"--split must be '{SPLIT}' for the ablation ladder (got '{args.split}'). "
            f"Every column is scored by medrap-eval eval_mode=test, which logs "
            f"test/auroc/* and no val/* column at all."
        )
    if args.rep_component != REP_COMPONENT:
        parser.error(
            f"--rep-component must be '{REP_COMPONENT}' for the ablation ladder (got "
            f"'{args.rep_component}'). The regex is re.fullmatch'ed per path component, so "
            f"anything narrower leaves seed<SEED> or rep<REP> in the cell key and the arms "
            f"stop matching -- silently, as an empty table."
        )
    return args


def build_report(args: argparse.Namespace) -> int:
    # aggregate_results.py reads every column name through its module-global `M`; it
    # defaults to the val split, so this assignment is what makes the imported loaders and
    # gates read test/auroc/* instead of finding nothing.
    agg.M = agg.SPLITS[args.split]
    outputs_root = args.outputs_root.rstrip("/")

    arms: dict[str, list] = {}
    resolved: dict[str, str] = {}
    for key in ARM_ORDER:
        _, token, has_rep = ARMS[key]
        override = getattr(args, f"glob_{key}")
        if override:
            sink = io.StringIO()
            with contextlib.redirect_stderr(sink):
                run_dirs = agg.discover_runs(override)
            pattern = " , ".join(override)
        else:
            run_dirs, pattern = resolve_arm(
                arm_glob_candidates(outputs_root, token, args.selection, has_rep)
            )
        resolved[key] = pattern
        arms[key] = [
            agg.load_run(d, args.gate_effective_k, args.gate_top1_mode_frac)
            for d in run_dirs
        ]

    # One keying pass over ALL four arms at once, so every comparison below reads the same
    # cell keys. assign_cell_keys chokes on arms with no runs, so only non-empty ones go in.
    non_empty = {k: v for k, v in arms.items() if v}
    if non_empty:
        agg.assign_cell_keys(non_empty, args.rep_component)

    arms_by_cell = {key: agg.group_by_cell(arms[key]) for key in ARM_ORDER}
    cells = sorted(
        {cell for by_cell in arms_by_cell.values() for cell in by_cell},
        key=agg.sort_key,
    )

    print_header(args.selection, outputs_root)
    print_coverage(args.selection, arms, resolved)
    print_gates(arms, args.verbose)
    if cells:
        print_table(arms_by_cell, cells)
    else:
        print("## Results")
        print()
        print("**No runs were found for any arm -- there is no table to print.** Check the "
              "resolved globs in the coverage block above against the eval scripts' "
              "`EVAL_OUTPUT_DIR`, and confirm the eval arrays have finished.")
        print()
    print_comparisons(arms_by_cell)
    print_cross_seed_sigma(arms_by_cell)
    return 0 if non_empty else 1


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.output is None:
        return build_report(args)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as handle:
        with contextlib.redirect_stdout(handle):
            status = build_report(args)
    print(f"wrote {out_path.resolve()}", file=sys.stderr)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
