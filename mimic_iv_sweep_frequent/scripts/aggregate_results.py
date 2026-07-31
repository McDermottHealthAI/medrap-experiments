"""Aggregate AUROC across run directories into the README results table.

Reads `<run_dir>/loggers/csv/version_0/metrics.csv` (the Lightning CSVLogger output)
for every run directory matched by the given globs and reports:

  * the **final** logged headline AUROC -- the pre-registered readout, and the one the
    published `mimic_iv_sweep_frequent` table quotes (verified: extracting the last
    `val/auroc/mean` from every metrics.csv reproduces all 16 cells exactly);
  * `*/auroc/n_tasks` / `*/auroc/n_valid_tasks` (a task is "valid" only if its scored
    split has >=1 positive and >=1 negative, otherwise AUROC is undefined and the task is
    dropped from the mean);
  * within-run volatility of the headline: range and sd of the last 5 validations. This
    matters -- some runs here swing by >0.2 AUROC across their own validations, which is
    larger than any effect the table claims to measure. Suppressed under --split test,
    where each run logs exactly one row;
  * acceptance-gate diagnostics: `effective_k_mean`, `top1_mode_frac`, `offdiag_cos_mean`
    and (val only) `grad_norm/train/query_projector`;
  * across-run aggregates (mean, sd, 95% t-CI) and, with --baseline/--treatment, a paired
    per-cell comparison (mean delta, sd, paired-t, 95% CI).

--split selects which metric family to read:

  * `--split val`  (default) -- medrap-train runs: `val/auroc/*` plus `*/train/*`
    diagnostics, many rows per run.
  * `--split test`           -- medrap-eval `eval_mode=test` runs, i.e. the
    eval_frozen_{real,random,null}_docs_n1248.sh ablations and
    eval_marginalized_binary_top1_n1248.sh: `test/auroc/*` plus `*/test/*` diagnostics,
    exactly one row per run. Those runs emit NO `val/*` column, so the default would find
    nothing.

Acceptance gates (a failing run is reported, and excluded from the headline aggregate, but
is still listed and still appears in the per-cell table):

  * `n_valid_tasks != n_tasks`  -- part of the label set was unscorable;
  * `effective_k_mean < 2.0`    -- retrieval collapsed onto one document, so marginalizing
                                   over K is arithmetically a no-op (checked only when the
                                   column exists, i.e. on marginalized runs);
  * `top1_mode_frac > 0.5`      -- more than half of patients retrieve the *same* top doc.
                                   Enforced as a gate ONLY on runs that also log the
                                   differentiable-retrieval column. Where that column is
                                   absent, retrieval never reaches the loss (e.g.
                                   fusion=passthrough in the patient_only arm), so a
                                   degenerate retriever cannot bias the AUROC; the value is
                                   reported as an informational NOTE and the run still
                                   counts toward the headline aggregate.

Usage
-----
Single arm::

    python scripts/aggregate_results.py \\
        'outputs/patient_only_n1248/n*' --arm-label patient_only

Paired comparison (emits the README markdown table)::

    python scripts/aggregate_results.py \\
        --baseline  'outputs/patient_only_n1248/n*'        --baseline-label  patient_only \\
        --treatment 'outputs/marginalized_binary_n1248/n*' --treatment-label 'marginalized (binary)'

Frozen-document ablation vs. its control (both are eval_mode=test, so --split test), with
the random-docs reps collapsed into a mean +/- sd per N::

    python scripts/aggregate_results.py --split test --rep-component 'rep\\d+' \\
        --baseline  'outputs/frozen_real_docs_eval_n1248/n*'   --baseline-label  'real docs' \\
        --treatment 'outputs/frozen_random_docs_eval_n1248/n*/rep*' --treatment-label 'random docs'

Add --sensitivity-best for the OPTIONAL best-validation column (never the headline; see the
warning the flag prints). It is ignored under --split test.

stdlib + pandas/numpy only -- no scipy (the Student-t helpers below are self-contained).
"""

from __future__ import annotations

import argparse
import glob as globlib
import math
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------------------
# metric names

METRICS_RELDIR = Path("loggers") / "csv"


@dataclass(frozen=True)
class MetricNames:
    """Column names for one split.

    Training runs (medrap-train) log `val/*` plus `*/train/*` diagnostics over many
    validation passes. Eval runs (medrap-eval eval_mode=test) log `test/*` plus
    `*/test/*` diagnostics, in exactly ONE row -- so there is no within-run volatility
    to report and no gradient-norm column at all.
    """

    split: str
    headline: str
    n_tasks: str
    n_valid_tasks: str
    effective_k: str
    top1_mode_frac: str
    offdiag_cos: str
    grad_query_proj: str | None

    @property
    def diagnostics(self) -> dict:
        """Column -> short header for the per-run text table."""
        out = {
            self.effective_k: "eff_k",
            self.top1_mode_frac: "top1_mode",
            self.offdiag_cos: "offdiag_cos",
        }
        if self.grad_query_proj is not None:
            out[self.grad_query_proj] = "grad_qproj"
        return out

    @property
    def is_single_row(self) -> bool:
        """True when a run logs one row, making last-N volatility meaningless."""
        return self.split == "test"


SPLITS = {
    "val": MetricNames(
        split="val",
        headline="val/auroc/mean",
        n_tasks="val/auroc/n_tasks",
        n_valid_tasks="val/auroc/n_valid_tasks",
        effective_k="retrieval/train/differentiable/effective_k_mean",
        top1_mode_frac="retrieval/train/top1_mode_frac",
        offdiag_cos="query/train/offdiag_cos_mean",
        grad_query_proj="grad_norm/train/query_projector",
    ),
    # medrap-eval eval_mode=test. Verified against a real eval metrics.csv
    # (outputs/marginalized_binary_top1_eval_n1248/n4/loggers/csv/version_0/metrics.csv):
    # its columns are test/auroc/{mean,n_tasks,n_valid_tasks},
    # retrieval/test/differentiable/effective_k_mean, retrieval/test/top1_mode_frac and
    # query/test/offdiag_cos_mean, in a single row. There is no grad_norm/* column --
    # trainer.test() computes no gradients.
    "test": MetricNames(
        split="test",
        headline="test/auroc/mean",
        n_tasks="test/auroc/n_tasks",
        n_valid_tasks="test/auroc/n_valid_tasks",
        effective_k="retrieval/test/differentiable/effective_k_mean",
        top1_mode_frac="retrieval/test/top1_mode_frac",
        offdiag_cos="query/test/offdiag_cos_mean",
        grad_query_proj=None,
    ),
}

# Active split, selected by --split and set once in main(). Everything below reads column
# names through this object rather than through module-level string constants.
M: MetricNames = SPLITS["val"]

VOLATILITY_WINDOW = 5

# Measured upward bias of max-over-validations, quoted by --sensitivity-best. For V roughly
# independent validations the expected maximum sits ~1.74 sd above the run's own mean at
# V=15, so the bias is proportional to each arm's noise and is NOT comparable across arms.
BEST_BIAS_NOISY = (0.139, 0.080)  # (bias, within-run sd) noisy arm, 15 validations
BEST_BIAS_QUIET = (0.043, 0.025)  # (bias, within-run sd) quiet arm, 15 validations


# --------------------------------------------------------------------------------------
# Student-t helpers (no scipy)


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (Numerical Recipes 6.4)."""
    tiny, eps, max_iter = 1e-30, 3e-16, 300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_sf(t: float, df: float) -> float:
    """P(T > t) for Student's t with `df` degrees of freedom."""
    if df <= 0:
        return float("nan")
    x = df / (df + t * t)
    tail = 0.5 * _betai(df / 2.0, 0.5, x)
    return tail if t > 0 else 1.0 - tail


def t_two_sided_p(t: float, df: float) -> float:
    if not math.isfinite(t) or df <= 0:
        return float("nan")
    return 2.0 * t_sf(abs(t), df)


def t_ppf(p: float, df: float) -> float:
    """Inverse CDF of Student's t, by bisection on the CDF. Adequate for CI endpoints."""
    if df <= 0:
        return float("nan")
    lo, hi = -1e4, 1e4
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if (1.0 - t_sf(mid, df)) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


@dataclass
class Summary:
    n: int
    mean: float
    sd: float
    ci_lo: float
    ci_hi: float

    def __str__(self) -> str:
        if self.n == 0:
            return "n=0  (nothing to aggregate)"
        if self.n == 1:
            return f"n=1  mean={self.mean:.4f}  sd=n/a  95% CI=n/a (single run)"
        return (
            f"n={self.n}  mean={self.mean:.4f}  sd={self.sd:.4f}  "
            f"95% t-CI=[{self.ci_lo:.4f}, {self.ci_hi:.4f}]"
        )


def summarize(values) -> Summary:
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    n = int(arr.size)
    if n == 0:
        return Summary(0, float("nan"), float("nan"), float("nan"), float("nan"))
    mean = float(arr.mean())
    if n == 1:
        return Summary(1, mean, float("nan"), float("nan"), float("nan"))
    sd = float(arr.std(ddof=1))
    half = t_ppf(0.975, n - 1) * sd / math.sqrt(n)
    return Summary(n, mean, sd, mean - half, mean + half)


# --------------------------------------------------------------------------------------
# run loading


@dataclass
class Run:
    run_dir: Path
    metrics_csv: Path
    cell: str = ""
    rep: str = ""  # rep/seed path components, when --rep-component collapsed them
    n_validations: int = 0
    final: float = float("nan")
    best: float = float("nan")
    run_sd: float = float("nan")  # sd over all validations, for the best-value bias check
    last5_range: float = float("nan")
    last5_sd: float = float("nan")
    n_tasks: float = float("nan")
    n_valid_tasks: float = float("nan")
    diagnostics: dict = field(default_factory=dict)
    gate_failures: list = field(default_factory=list)
    # Diagnostics worth printing that are NOT grounds for exclusion (see load_run).
    observations: list = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.gate_failures

    @property
    def valid_tasks_str(self) -> str:
        if not np.isfinite(self.n_tasks):
            return "?/?"
        return f"{int(self.n_valid_tasks)}/{int(self.n_tasks)}"


def find_metrics_csv(path: Path) -> Path | None:
    """Resolve a user-supplied path to its metrics.csv."""
    if path.is_file() and path.name == "metrics.csv":
        return path
    preferred = path / METRICS_RELDIR / "version_0" / "metrics.csv"
    if preferred.is_file():
        return preferred
    candidates = sorted(
        (path / METRICS_RELDIR).glob("version_*/metrics.csv"),
        key=lambda p: int(p.parent.name.split("_")[-1]),
    )
    return candidates[-1] if candidates else None


def discover_runs(patterns: list[str]) -> list[Path]:
    """Expand globs to run directories that actually contain a metrics.csv."""
    found: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        matches = sorted(globlib.glob(os.path.expanduser(pattern), recursive=True))
        if not matches:
            print(f"WARNING: glob matched nothing: {pattern}", file=sys.stderr)
        for match in matches:
            path = Path(match)
            csv = find_metrics_csv(path)
            if csv is None:
                print(f"WARNING: no metrics.csv under {path}", file=sys.stderr)
                continue
            run_dir = path if path.is_dir() else csv.parents[3]
            run_dir = run_dir.resolve()
            if run_dir in seen:
                continue
            seen.add(run_dir)
            found.append(run_dir)
    return found


def _final(df: pd.DataFrame, column: str) -> float:
    if column not in df.columns:
        return float("nan")
    series = df[column].dropna()
    return float(series.iloc[-1]) if len(series) else float("nan")


def load_run(run_dir: Path, gate_effective_k: float, gate_top1: float) -> Run:
    csv = find_metrics_csv(run_dir)
    if csv is None:
        raise FileNotFoundError(f"no metrics.csv under {run_dir}")
    df = pd.read_csv(csv)
    run = Run(run_dir=run_dir, metrics_csv=csv)

    if M.headline not in df.columns:
        run.gate_failures.append(f"{M.headline} column absent from {csv}")
        return run
    series = df[M.headline].dropna()
    run.n_validations = int(len(series))
    if run.n_validations == 0:
        run.gate_failures.append(f"{M.headline} logged 0 times (run produced no {M.split} pass)")
        return run

    run.final = float(series.iloc[-1])
    run.best = float(series.max())
    run.run_sd = float(series.std(ddof=1)) if run.n_validations > 1 else float("nan")
    if not M.is_single_row:
        window = series.tail(VOLATILITY_WINDOW)
        run.last5_range = float(window.max() - window.min())
        run.last5_sd = float(window.std(ddof=1)) if len(window) > 1 else float("nan")

    run.n_tasks = _final(df, M.n_tasks)
    run.n_valid_tasks = _final(df, M.n_valid_tasks)
    run.diagnostics = {name: _final(df, name) for name in M.diagnostics if name in df.columns}

    # --- acceptance gates -------------------------------------------------------------
    if np.isfinite(run.n_tasks) and np.isfinite(run.n_valid_tasks):
        if run.n_valid_tasks != run.n_tasks:
            dropped = int(run.n_tasks - run.n_valid_tasks)
            run.gate_failures.append(
                f"{M.n_valid_tasks}={int(run.n_valid_tasks)} != {M.n_tasks}={int(run.n_tasks)} "
                f"({dropped} task(s) unscorable, dropped from the mean)"
            )
    else:
        run.gate_failures.append(
            f"{M.n_tasks}/{M.n_valid_tasks} not logged -- cannot verify task validity"
        )

    eff_k = run.diagnostics.get(M.effective_k)
    if eff_k is not None and np.isfinite(eff_k) and eff_k < gate_effective_k:
        run.gate_failures.append(
            f"{M.effective_k}={eff_k:.4f} < {gate_effective_k} -- retrieval collapsed onto "
            f"one document, marginalization over K is a no-op"
        )

    top1 = run.diagnostics.get(M.top1_mode_frac)
    if top1 is not None and np.isfinite(top1) and top1 > gate_top1:
        message = (
            f"{M.top1_mode_frac}={top1:.4f} > {gate_top1} -- most patients retrieve the same "
            f"top document"
        )
        if eff_k is None:
            # No differentiable-retrieval column means retrieval never reaches the loss
            # (e.g. fusion=passthrough in the patient_only arm), so a degenerate retriever
            # provably cannot bias this run's AUROC. Report the observation, but do NOT
            # fail the run on it -- doing so excluded every patient_only run from the
            # headline aggregate on a criterion that does not apply to them.
            run.observations.append(
                message + " [no differentiable-retrieval column in this run, so retrieval is "
                "logged but does not reach the loss -- informational, NOT a gate failure]"
            )
        else:
            run.gate_failures.append(message)

    return run


# --------------------------------------------------------------------------------------
# cell keys (match runs across arms)


def _relparts(run_dir: Path, root: Path) -> list[str]:
    return list(run_dir.relative_to(root).parts)


def common_root(run_dirs: list[Path]) -> Path:
    if not run_dirs:
        return Path("/")
    if len(run_dirs) == 1:
        return run_dirs[0].parent
    return Path(os.path.commonpath([str(p) for p in run_dirs]))


def assign_cell_keys(
    arms: dict[str, list[Run]], rep_component: str | None = None
) -> tuple[Path, int | None]:
    """Give every run a cell key, matching cells across arms.

    Cells are matched on the run-directory path components *other than* the one that
    differs only in the arm name. We find that component by trying to drop each path
    position in turn and keeping whichever choice keeps keys unique inside each arm while
    matching the most cells across arms.

    `rep_component` is an optional regex; path components matching it are also dropped, so
    that seeds/reps of the same condition collapse into one cell and get aggregated. Off by
    default -- without it every rep is its own cell, which is what you want when reps are
    matched across arms (e.g. `draw1` trained on the same labels in both arms).

    Returns (common root, dropped component index or None).
    """
    rep_re = re.compile(rep_component) if rep_component else None
    all_runs = [r for runs in arms.values() for r in runs]
    root = common_root([r.run_dir for r in all_runs])
    parts = {r.run_dir: _relparts(r.run_dir, root) for r in all_runs}
    max_depth = max((len(p) for p in parts.values()), default=0)

    def keys_for(drop: int | None) -> dict[Path, str]:
        out = {}
        for run_dir, comps in parts.items():
            kept = [
                c
                for i, c in enumerate(comps)
                if (drop is None or i != drop) and not (rep_re and rep_re.fullmatch(c))
            ]
            out[run_dir] = "/".join(kept) or "."
        return out

    def score(drop: int | None) -> tuple[int, int]:
        keys = keys_for(drop)
        per_arm = []
        for runs in arms.values():
            arm_keys = [keys[r.run_dir] for r in runs]
            # Repeated keys inside an arm are reps only when --rep-component asked for it;
            # otherwise they mean this position is not the arm component.
            if rep_re is None and len(set(arm_keys)) != len(arm_keys):
                return (-1, 0)
            per_arm.append(set(arm_keys))
        shared = set.intersection(*per_arm) if per_arm else set()
        return (len(shared), -(drop if drop is not None else max_depth))

    candidates = [None] + list(range(max_depth)) if len(arms) > 1 else [None]
    best_drop = max(candidates, key=score)
    if score(best_drop)[0] <= 0 and len(arms) > 1:
        best_drop = None  # nothing matched; fall back to full relative paths

    keys = keys_for(best_drop)
    for run in all_runs:
        run.cell = keys[run.run_dir]
        if rep_re:
            run.rep = "/".join(c for c in parts[run.run_dir] if rep_re.fullmatch(c))
    return root, best_drop


def sort_key(cell: str):
    """Sort cells numerically where the components are numeric (n1 < n2 < ... < n128)."""
    out = []
    for part in cell.split("/"):
        digits = "".join(ch for ch in part if ch.isdigit())
        prefix = "".join(ch for ch in part if not ch.isdigit())
        out.append((prefix, int(digits) if digits else -1))
    return out


def display_cell(cell: str) -> str:
    """`n16` -> `16`; leaves anything else alone."""
    if cell.startswith("n") and cell[1:].isdigit():
        return cell[1:]
    return cell


# --------------------------------------------------------------------------------------
# reporting


def fmt(value: float, digits: int = 4) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    if abs(value) < 1e-4 and value != 0:
        return f"{value:.2e}"
    return f"{value:.{digits}f}"


def fmt_signed(value: float, digits: int = 4) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:+.{digits}f}"


def print_run_table(label: str, runs: list[Run], root: Path, show_best: bool) -> None:
    print(f"=== {label}: {len(runs)} run(s) ===")
    print(f"    root: {root}")
    diag_present = [name for name in M.diagnostics if any(name in r.diagnostics for r in runs)]

    # A test run logs exactly one row, so the last-5 window and the max-over-validations
    # columns carry no information -- suppress them rather than print n/a or a value equal
    # to `final`.
    show_volatility = not M.is_single_row
    show_best = show_best and not M.is_single_row

    has_reps = any(r.rep for r in runs)
    header = ["cell", "final", f"#{M.split}"]
    if show_volatility:
        header += ["last5_rng", "last5_sd"]
    header += ["valid/total"]
    if has_reps:
        header.insert(1, "rep")
    if show_best:
        header.insert(2 + has_reps, "best*")
        header.insert(3 + has_reps, "run_sd")
    header += [M.diagnostics[name] for name in diag_present] + ["gates"]
    rows = []
    for run in sorted(runs, key=lambda r: (sort_key(r.cell), sort_key(r.rep))):
        row = [run.cell, fmt(run.final), str(run.n_validations)]
        if show_volatility:
            row += [fmt(run.last5_range), fmt(run.last5_sd)]
        row += [run.valid_tasks_str]
        if has_reps:
            row.insert(1, run.rep)
        if show_best:
            row.insert(2 + has_reps, fmt(run.best))
            row.insert(3 + has_reps, fmt(run.run_sd))
        row += [fmt(run.diagnostics.get(name, float("nan"))) for name in diag_present]
        row.append("PASS" if run.valid else f"INVALID({len(run.gate_failures)})")
        rows.append(row)

    widths = [max(len(header[i]), *(len(r[i]) for r in rows)) if rows else len(header[i])
              for i in range(len(header))]
    print("    " + "  ".join(h.ljust(w) for h, w in zip(header, widths)))
    print("    " + "  ".join("-" * w for w in widths))
    for row in rows:
        print("    " + "  ".join(c.ljust(w) for c, w in zip(row, widths)))


def print_gate_report(label: str, runs: list[Run]) -> None:
    failed = [r for r in runs if not r.valid]
    noted = [r for r in runs if r.observations]
    print(f"\n--- acceptance gates: {label} ---")
    if not failed:
        print(f"    all {len(runs)} run(s) pass.")
    for run in sorted(failed, key=lambda r: (sort_key(r.cell), sort_key(r.rep))):
        print(f"    EXCLUDED {run.cell}{'/' + run.rep if run.rep else ''}  ({run.run_dir})")
        for reason in run.gate_failures:
            print(f"             reason: {reason}")
    if noted:
        print("    informational (these runs still count toward the headline aggregate):")
        for run in sorted(noted, key=lambda r: (sort_key(r.cell), sort_key(r.rep))):
            for note in run.observations:
                print(f"        NOTE {run.cell}{'/' + run.rep if run.rep else ''}: {note}")
    if not failed:
        return
    print(f"    {len(failed)}/{len(runs)} run(s) excluded from the headline aggregate.")
    if len(failed) == len(runs):
        print(
            "    *** NO RUN PASSES THE GATES. The headline aggregate is empty; every number\n"
            "        below is DESCRIPTIVE ONLY and must not be read as an architecture result. ***"
        )


def print_aggregates(label: str, runs: list[Run], show_best: bool) -> None:
    show_best = show_best and not M.is_single_row
    print(f"\n--- aggregate: {label} (final {M.headline}) ---")
    valid = [r for r in runs if r.valid]
    print(f"    gate-passing runs : {summarize([r.final for r in valid])}")
    print(f"    ALL runs (descr.) : {summarize([r.final for r in runs])}")
    if show_best:
        print(f"    OPTIONAL best-val, ALL runs (biased, see warning): "
              f"{summarize([r.best for r in runs])}")
        bias = [r.best - r.final for r in runs if np.isfinite(r.best - r.final)]
        sds = [r.run_sd for r in runs if np.isfinite(r.run_sd)]
        if bias:
            print(f"        observed best-minus-final for this arm: mean +{np.mean(bias):.4f} "
                  f"(max +{np.max(bias):.4f}); mean within-run sd {np.mean(sds):.4f}")
            print("        ^ that inflation is a property of this arm's noise, not of its quality.")

    by_cell: dict[str, list[Run]] = {}
    for run in runs:
        by_cell.setdefault(run.cell, []).append(run)
    if any(len(v) > 1 for v in by_cell.values()):
        print("    per-cell aggregate over seeds/reps (gate-passing runs):")
        for cell in sorted(by_cell, key=sort_key):
            cell_runs = [r for r in by_cell[cell] if r.valid]
            print(f"        {cell:>16}: {summarize([r.final for r in cell_runs])}")


def print_paired(
    base_label: str,
    treat_label: str,
    base: dict[str, list[Run]],
    treat: dict[str, list[Run]],
    show_best: bool,
) -> list[str]:
    """Paired per-cell comparison. Returns the sorted list of matched cells."""
    show_best = show_best and not M.is_single_row
    cells = sorted(set(base) & set(treat), key=sort_key)
    unmatched = sorted((set(base) ^ set(treat)), key=sort_key)
    print(f"\n=== paired comparison: {treat_label} - {base_label} ===")
    print(f"    matched cells: {len(cells)}")
    if unmatched:
        print(f"    UNMATCHED cells (ignored): {', '.join(unmatched)}")

    def cell_mean(runs: list[Run], attr: str) -> float:
        vals = [getattr(r, attr) for r in runs if np.isfinite(getattr(r, attr))]
        return float(np.mean(vals)) if vals else float("nan")

    header = ["cell", base_label, treat_label, "delta", "both gates pass"]
    rows = []
    deltas_all, deltas_valid = [], []
    for cell in cells:
        b, t = cell_mean(base[cell], "final"), cell_mean(treat[cell], "final")
        delta = t - b
        both_ok = all(r.valid for r in base[cell] + treat[cell])
        rows.append([cell, fmt(b), fmt(t), fmt_signed(delta), "yes" if both_ok else "no"])
        if np.isfinite(delta):
            deltas_all.append(delta)
            if both_ok:
                deltas_valid.append(delta)

    widths = [max(len(header[i]), *(len(r[i]) for r in rows)) if rows else len(header[i])
              for i in range(len(header))]
    print("    " + "  ".join(h.ljust(w) for h, w in zip(header, widths)))
    print("    " + "  ".join("-" * w for w in widths))
    for row in rows:
        print("    " + "  ".join(c.ljust(w) for c, w in zip(row, widths)))

    def paired_stats(name: str, deltas: list[float]) -> None:
        n = len(deltas)
        print(f"\n    paired delta, {name}: n={n}")
        if n == 0:
            print("        nothing to test.")
            return
        arr = np.asarray(deltas, dtype=float)
        mean = float(arr.mean())
        if n == 1:
            print(f"        mean delta={mean:+.4f}  sd=n/a  (single pair, no test)")
            return
        sd = float(arr.std(ddof=1))
        se = sd / math.sqrt(n)
        tstat = mean / se if se > 0 else float("nan")
        df = n - 1
        crit = t_ppf(0.975, df)
        lo, hi = mean - crit * se, mean + crit * se
        print(f"        mean delta = {mean:+.4f}   sd = {sd:.4f}   se = {se:.4f}")
        print(f"        paired t({df}) = {tstat:.3f}   two-sided p = {t_two_sided_p(tstat, df):.4f}")
        print(f"        95% t-CI   = [{lo:+.4f}, {hi:+.4f}]")
        neg = int((arr < 0).sum())
        print(f"        {neg}/{n} cells negative ({treat_label} worse), "
              f"{n - neg}/{n} non-negative")

    paired_stats("gate-passing cells only (HEADLINE)", deltas_valid)
    paired_stats("ALL matched cells (descriptive)", deltas_all)

    if show_best:
        best_deltas = [
            cell_mean(treat[c], "best") - cell_mean(base[c], "best")
            for c in cells
            if np.isfinite(cell_mean(treat[c], "best") - cell_mean(base[c], "best"))
        ]
        paired_stats("OPTIONAL best-validation readout, ALL cells (BIASED)", best_deltas)

    return cells


def print_markdown_table(
    base_label: str,
    treat_label: str,
    base: dict[str, list[Run]],
    treat: dict[str, list[Run]],
    cells: list[str],
    cell_header: str,
    show_best: bool,
) -> None:
    show_best = show_best and not M.is_single_row
    print("\n--- markdown results table ---\n")

    def cell_mean(runs: list[Run], attr: str) -> float:
        vals = [getattr(r, attr) for r in runs if np.isfinite(getattr(r, attr))]
        return float(np.mean(vals)) if vals else float("nan")

    def valid_tasks(cell: str) -> str:
        b = {r.valid_tasks_str for r in base[cell]}
        t = {r.valid_tasks_str for r in treat[cell]}
        bs = ", ".join(sorted(b)) or "?"
        ts = ", ".join(sorted(t)) or "?"
        return bs if b == t else f"{bs} vs {ts}"

    header = [cell_header, base_label, treat_label, f"Δ ({treat_label} − {base_label})", "valid tasks"]
    if show_best:
        header.append(f"Δ best-val (OPTIONAL, biased) ")
    print("| " + " | ".join(h.strip() for h in header) + " |")
    print("| " + " | ".join("---" for _ in header) + " |")

    def shown_delta(cell: str, attr: str) -> float:
        # Repo convention: Δ is the difference of the two *displayed* 4-decimal values, so
        # the printed table is internally consistent. The paired statistics above use full
        # precision, so the two can differ by 1 in the last digit.
        b, t = cell_mean(base[cell], attr), cell_mean(treat[cell], attr)
        if not (np.isfinite(b) and np.isfinite(t)):
            return float("nan")
        return round(t, 4) - round(b, 4)

    flagged = []
    for cell in cells:
        b, t = cell_mean(base[cell], "final"), cell_mean(treat[cell], "final")
        row = [display_cell(cell), fmt(b), fmt(t), fmt_signed(shown_delta(cell, "final")),
               valid_tasks(cell)]
        if show_best:
            row.append(fmt_signed(shown_delta(cell, "best")))
        print("| " + " | ".join(row) + " |")
        if not all(r.valid for r in base[cell] + treat[cell]):
            flagged.append(display_cell(cell))

    print(f"\nHeadline readout: final logged `{M.headline}` ({ARM_NOTE[M.split]}).")
    print("Δ is the difference of the two displayed 4-decimal values (repo convention); the paired\n"
          "statistics above are computed at full precision, so they can differ in the last digit.")
    if flagged:
        print(
            f"**Gate failures in cells: {', '.join(flagged)}** -- these rows are reported for\n"
            f"reproducibility but at least one of their runs failed an acceptance gate; see the\n"
            f"gate report above for the per-run reason. They are excluded from the headline aggregate."
        )


ARM_NOTE = {
    "val": "last of the validations logged in each run",
    "test": "the single row each medrap-eval eval_mode=test run logs",
}


BEST_WARNING = f"""
!!! OPTIONAL SENSITIVITY COLUMN -- NOT THE HEADLINE !!!
    The best-validation column takes the max over each run's validations. Max-over-validations
    is UPWARD-BIASED, by an amount PROPORTIONAL TO THAT ARM'S OWN NOISE: for 15 roughly
    independent validations the expected maximum lands about 1.74 within-run sd above the run's
    mean. Measured:
      * arm with within-run sd {BEST_BIAS_NOISY[1]:.3f} -> +{BEST_BIAS_NOISY[0]:.3f}
      * arm with within-run sd {BEST_BIAS_QUIET[1]:.3f} -> +{BEST_BIAS_QUIET[0]:.3f}
    So a noisier arm gains ~3x more from this readout for purely statistical reasons -- switching
    to "best" can manufacture, erase, or flip a delta between two arms that differ only in
    volatility. Use it as a sensitivity check beside the pre-registered final-value headline,
    never as the headline, and never compare two arms on it without matching their volatility
    (compare the `run_sd` column first).
"""


# --------------------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Headline = FINAL logged <split>/auroc/mean (pre-registered). Use --split test for "
               "medrap-eval eval_mode=test runs. See --sensitivity-best.",
    )
    parser.add_argument(
        "run_globs", nargs="*", help="run-directory glob(s) for a single-arm report, e.g. 'outputs/patient_only_n1248/n*'"
    )
    parser.add_argument("--arm-label", default="arm", help="label for the positional run globs")
    parser.add_argument("--baseline", action="append", default=[], metavar="GLOB",
                        help="baseline run-directory glob (repeatable); pairs with --treatment")
    parser.add_argument("--treatment", action="append", default=[], metavar="GLOB",
                        help="treatment run-directory glob (repeatable); pairs with --baseline")
    parser.add_argument("--baseline-label", default=None, help="column label for the baseline arm")
    parser.add_argument("--treatment-label", default=None, help="column label for the treatment arm")
    parser.add_argument("--cell-header", default="N", help="first column header of the markdown table")
    parser.add_argument("--rep-component", default=None, metavar="REGEX",
                        help="treat path components fully matching REGEX (e.g. 'draw\\d+', 'seed\\d+') "
                             "as seeds/reps: they are dropped from the cell key so runs sharing a cell "
                             "are aggregated (mean/sd/95%% CI) instead of being separate cells")
    parser.add_argument("--split", choices=sorted(SPLITS), default="val",
                        help="which split's metrics to read: 'val' for medrap-train runs "
                             "(val/auroc/* + */train/* diagnostics, many rows), 'test' for "
                             "medrap-eval eval_mode=test runs such as the eval_frozen_*_docs "
                             "ablations (test/auroc/* + */test/* diagnostics, one row). "
                             "Default: val")
    parser.add_argument("--sensitivity-best", action="store_true",
                        help="add the OPTIONAL best-validation column (prints a bias warning); "
                             "ignored with --split test, where each run logs a single row")
    parser.add_argument("--gate-effective-k", type=float, default=2.0,
                        help="minimum retrieval/train/differentiable/effective_k_mean (default 2.0)")
    parser.add_argument("--gate-top1-mode-frac", type=float, default=0.5,
                        help="maximum retrieval/train/top1_mode_frac (default 0.5)")
    args = parser.parse_args(argv)
    if not args.run_globs and not (args.baseline and args.treatment):
        parser.error("give run-directory glob(s), or both --baseline and --treatment")
    if bool(args.baseline) != bool(args.treatment):
        parser.error("--baseline and --treatment must be given together")
    return args


def group_by_cell(runs: list[Run]) -> dict[str, list[Run]]:
    out: dict[str, list[Run]] = {}
    for run in runs:
        out.setdefault(run.cell, []).append(run)
    return out


def main(argv: list[str] | None = None) -> int:
    global M
    args = parse_args(argv)
    M = SPLITS[args.split]
    print(f"reading `{M.headline}` and the {M.split}-split diagnostics (--split {args.split}).")
    if args.sensitivity_best and M.is_single_row:
        print("NOTE: --sensitivity-best ignored with --split test (each run logs one row, so\n"
              "      max-over-rows is identical to the final value).")
        args.sensitivity_best = False
    if args.sensitivity_best:
        print(BEST_WARNING)

    arms: dict[str, list[Run]] = {}
    order: list[str] = []

    def load(label: str, patterns: list[str]) -> None:
        run_dirs = discover_runs(patterns)
        runs = [load_run(d, args.gate_effective_k, args.gate_top1_mode_frac) for d in run_dirs]
        arms[label] = runs
        order.append(label)

    base_label = treat_label = None
    if args.baseline:
        base_label = args.baseline_label or "baseline"
        treat_label = args.treatment_label or "treatment"
        if base_label == treat_label:
            treat_label += " (treatment)"
        load(base_label, args.baseline)
        load(treat_label, args.treatment)
    if args.run_globs:
        label = args.arm_label if args.arm_label not in arms else f"{args.arm_label} (extra)"
        load(label, args.run_globs)

    if not any(arms.values()):
        print("ERROR: no runs found.", file=sys.stderr)
        return 1

    root, dropped = assign_cell_keys({k: v for k, v in arms.items() if v}, args.rep_component)
    print(f"cell key = run path relative to {root}, "
          + (f"dropping path component #{dropped} (the arm name)" if dropped is not None
             else "using the full relative path")
          + (f", and dropping rep components matching /{args.rep_component}/"
             if args.rep_component else ""))
    print()

    for label in order:
        runs = arms[label]
        if not runs:
            print(f"=== {label}: 0 run(s) ===\n")
            continue
        print_run_table(label, runs, root, args.sensitivity_best)
        print_gate_report(label, runs)
        print_aggregates(label, runs, args.sensitivity_best)
        print()

    if base_label is not None and arms[base_label] and arms[treat_label]:
        base = group_by_cell(arms[base_label])
        treat = group_by_cell(arms[treat_label])
        cells = print_paired(base_label, treat_label, base, treat, args.sensitivity_best)
        if cells:
            print_markdown_table(base_label, treat_label, base, treat, cells,
                                 args.cell_header, args.sensitivity_best)

    if args.sensitivity_best:
        print(BEST_WARNING)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
