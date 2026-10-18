"""Presentation figures and tables for the scaling-law experiment.

Reads the same outputs/<arm>_dim<D>_trainN<P>_tasks<T>_test/draw<d> evals as
aggregate_results.py (only each eval's CSV logger is needed, so a local copy of those files is
enough) and writes, under <out>/:

- figures/fig1_data_scaling: AUROC vs training patients, one panel per encoder width.
- figures/fig2_capacity_scaling: AUROC vs patient-encoder parameters, one panel per data size.
- figures/fig3_retrieval_advantage: retrieval - patient_only AUROC over the main grid.
- figures/fig4_task_count_ablation: (a) both arms vs training patients, one panel per task
  count; (b) retrieval - patient_only vs training patients, one line per task count.
- tables/*.md and tables/*.tex: mean +- std AUROC per arm, the difference grid, the task-count
  ablation, and (once every main-grid cell exists) the fitted law per arm.

Points are means over draws with +-1 std error bars. Once both arms have the full main grid,
the fitted separable power law (simple-scaling-laws, median and 95% band) is drawn through
them; before that, means are joined by plain lines.

Usage:
    cd mimic_iv_scaling_laws && .venv/bin/python scripts/plot_results.py [--out results]
    # or against a local copy of the eval CSVs:
    cd ~/Desktop/scaling_laws_results && <venv>/bin/python <repo>/scripts/plot_results.py
"""

import argparse
import contextlib
import io
import itertools
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.ticker import FuncFormatter

sys.path.insert(0, str(Path(__file__).parent))
from aggregate_results import (
    ABLATION_DIM,
    CONTROL_ARMS,
    CONTROL_NUM_TASKS,
    MAIN_NUM_TASKS,
    N_DRAWS,
    REFERENCE_ARM,
    collect_results,
    encoder_n_params,
)
from simple_scaling_laws import compare, fit

ARMS = ["patient_only", "retrieval"]
ARM_LABEL = {
    "patient_only": "Patient only",
    "retrieval": "Retrieval",
    "retrieval_random_docs": "Retrieval, random docs",
    "retrieval_null_docs": "Retrieval, null docs",
}
# Reference data-viz palette (validated: light mode, adjacent + all pairs for two slots).
ARM_COLOR = {"patient_only": "#2a78d6", "retrieval": "#eb6834"}
# Control figure: categorical slots 1-4 in fixed order (validated adjacent pairs; slots 3-4 sit
# below 3:1 on the surface, so that figure ships with marker shapes and a table view).
CONTROL_COLOR = {**ARM_COLOR, "retrieval_random_docs": "#1baf7a", "retrieval_null_docs": "#eda100"}
CONTROL_MARKER = {
    "patient_only": "o",
    "retrieval": "o",
    "retrieval_random_docs": "s",
    "retrieval_null_docs": "^",
}
TASK_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]  # ordinal, light -> dark
SURFACE, INK, INK_2, GRID, NEUTRAL = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#f0efec"
DIVERGING = LinearSegmentedColormap.from_list(
    "patient_vs_retrieval", [ARM_COLOR["patient_only"], NEUTRAL, ARM_COLOR["retrieval"]]
)

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.size": 10,
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": GRID,
        "axes.linewidth": 1,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "xtick.minor.size": 0,
        "legend.frameon": False,
        "lines.linewidth": 2,
        "lines.solid_capstyle": "round",
    }
)


def compact(x: float, _pos: object = None) -> str:
    for div, suffix in ((1e6, "M"), (1e3, "K")):
        if x >= div:
            v = x / div
            return f"{v:.0f}{suffix}" if v == int(v) else f"{v:.1f}{suffix}"
    return f"{x:.0f}"


def summarize(results: pl.DataFrame) -> pl.DataFrame:
    """Mean/std/count of test AUROC over draws per (arm, dim, n_train, num_tasks)."""
    dims = {encoder_n_params(d): d for d in (4, 8, 16, 32, 64, 128)}
    return (
        results.with_columns(pl.col("model_size__n_params").replace_strict(dims).alias("dim"))
        .rename({"dataset_size__n_train_patients": "n_train"})
        .group_by("arm", "dim", "model_size__n_params", "n_train", "num_tasks")
        .agg(
            pl.col("test_metric__auroc").mean().alias("mean"),
            pl.col("test_metric__auroc").std().fill_null(0.0).alias("std"),
            pl.len().alias("n"),
        )
        .sort("arm", "num_tasks", "dim", "n_train")
    )


def fitted_curves(results: pl.DataFrame, summary: pl.DataFrame) -> tuple[dict, pl.DataFrame | None]:
    """Fit each arm on the main grid if it is complete; return per-arm summaries and dense curves."""
    main = summary.filter(pl.col("num_tasks") == MAIN_NUM_TASKS)
    dims, ns = sorted(main["dim"].unique()), sorted(main["n_train"].unique())
    complete = all(main.filter(pl.col("arm") == a).height == len(dims) * len(ns) for a in ARMS)
    if not complete or len(dims) < 3 or len(ns) < 3:
        return {}, None
    data = results.filter(pl.col("num_tasks") == MAIN_NUM_TASKS).drop("num_tasks")
    models = {}
    for arm in ARMS:
        with contextlib.redirect_stdout(io.StringIO()):
            models[arm] = fit(data.filter(pl.col("arm") == arm).drop("arm"), n_draws=N_DRAWS, seed=0)
    n_grid = np.geomspace(min(ns), max(ns), 60)
    p_grid = np.geomspace(encoder_n_params(min(dims)), encoder_n_params(max(dims)), 60)
    pts = [(encoder_n_params(d), n) for d in dims for n in n_grid] + [(p, n) for n in ns for p in p_grid]
    points = {
        "point_id": [f"p{i}" for i in range(len(pts))],
        "model_size__n_params": [float(p) for p, _ in pts],
        "dataset_size__n_train_patients": [float(n) for _, n in pts],
    }
    curves = compare(models, points, target="test_metric__auroc", reference=REFERENCE_ARM)
    return {arm: m.summary() for arm, m in models.items()}, curves


def log_x(ax, lo: float, hi: float) -> None:
    """Log x-axis over the full grid range (so partial results don't rescale it), compact ticks."""
    ax.set_xscale("log")
    ax.set_xlim(lo / 1.4, hi * 1.4)
    ax.xaxis.set_major_formatter(FuncFormatter(compact))
    ax.xaxis.set_minor_formatter(FuncFormatter(lambda *_: ""))


def errorbars(ax, x, df: pl.DataFrame, arm: str) -> None:
    ax.errorbar(
        x,
        df["mean"],
        yerr=df["std"],
        fmt="o",
        ms=6,
        color=ARM_COLOR[arm],
        mec=SURFACE,
        mew=1.5,
        elinewidth=1.2,
        capsize=0,
        zorder=3,
    )


def curve(ax, curves: pl.DataFrame | None, arm: str, xcol: str, fixed: tuple[str, float], fallback) -> None:
    if curves is not None:
        c = curves.filter(
            (pl.col("system") == arm) & ((pl.col(fixed[0]) - fixed[1]).abs() < 1e-6 * fixed[1])
        ).sort(xcol)
        ax.fill_between(c[xcol], c["value__q025"], c["value__q975"], color=ARM_COLOR[arm], alpha=0.1, lw=0)
        ax.plot(c[xcol], c["value__median"], color=ARM_COLOR[arm], zorder=2)
    else:
        ax.plot(*fallback, color=ARM_COLOR[arm], zorder=2, alpha=0.9)


def arm_legend(fig, note: str) -> None:
    handles = [
        plt.Line2D([], [], color=ARM_COLOR[a], marker="o", ms=6, mec=SURFACE, label=ARM_LABEL[a])
        for a in ARMS
    ]
    fig.legend(handles=handles, loc="upper right", ncol=2, bbox_to_anchor=(0.99, 1.0))
    fig.text(0.01, 0.005, note, color=INK_2, fontsize=8, va="bottom")


def save(fig, out: Path, name: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_scaling(summary, curves, out, *, axis: str) -> None:
    """axis='data': x = training patients, panel per dim. axis='capacity': x = params, panel per N."""
    main = summary.filter(pl.col("num_tasks") == MAIN_NUM_TASKS)
    if axis == "data":
        panels, panel_col, x_col, fit_x, fixed_col = (
            sorted(main["dim"].unique()),
            "dim",
            "n_train",
            "dataset_size__n_train_patients",
            "model_size__n_params",
        )
        nrows, ncols = 2, 3
    else:
        panels, panel_col, x_col, fit_x, fixed_col = (
            sorted(main["n_train"].unique()),
            "n_train",
            "model_size__n_params",
            "model_size__n_params",
            "dataset_size__n_train_patients",
        )
        nrows, ncols = 1, len(panels)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.3 * ncols, 2.9 * nrows + 0.5), sharey=True, squeeze=False
    )
    for ax, panel in zip(axes.flat, panels, strict=False):
        sub = main.filter(pl.col(panel_col) == panel)
        fixed_val = encoder_n_params(panel) if axis == "data" else panel
        for arm in ARMS:
            d = sub.filter(pl.col("arm") == arm).sort(x_col)
            if d.is_empty():
                continue
            curve(ax, curves, arm, fit_x, (fixed_col, fixed_val), (d[x_col], d["mean"]))
            errorbars(ax, d[x_col], d, arm)
        xs = (
            sorted(summary[x_col].unique())
            if axis == "data"
            else [encoder_n_params(d) for d in sorted(summary["dim"].unique())]
        )
        log_x(ax, xs[0], xs[-1])
        title = (
            f"dim {panel}  ({compact(encoder_n_params(panel))} params)"
            if axis == "data"
            else f"{compact(panel)} training patients"
        )
        ax.set_title(title, color=INK, fontsize=10, loc="left")
    for ax in axes.flat[len(panels) :]:
        ax.set_visible(False)
    xlabel = "Training patients" if axis == "data" else "Patient-encoder parameters"
    for ax in axes[-1]:
        ax.set_xlabel(xlabel)
    for row in axes:
        row[0].set_ylabel(f"Test AUROC (mean of {MAIN_NUM_TASKS} tasks)")
    head = (
        "AUROC vs training-set size, by encoder width"
        if axis == "data"
        else "AUROC vs encoder size, by training-set size"
    )
    fig.suptitle(head, x=0.01, ha="left", fontsize=12, color=INK)
    note = "Points: mean over 5 draws, bars ±1 std. " + (
        "Lines: fitted separable power law per arm (median, 95% band)."
        if curves is not None
        else "Lines join the means (fit shown once the grid is complete)."
    )
    arm_legend(fig, note)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    save(fig, out, "fig1_data_scaling" if axis == "data" else "fig2_capacity_scaling")


def advantage_grid(summary: pl.DataFrame) -> pl.DataFrame:
    main = summary.filter(pl.col("num_tasks") == MAIN_NUM_TASKS)
    wide = main.pivot(on="arm", index=["dim", "n_train"], values="mean")
    return wide.with_columns((pl.col("retrieval") - pl.col(REFERENCE_ARM)).alias("diff")).sort(
        "dim", "n_train"
    )


def fig_advantage(summary, out) -> None:
    g = advantage_grid(summary)
    dims, ns = sorted(g["dim"].unique()), sorted(g["n_train"].unique())
    mat = np.full((len(dims), len(ns)), np.nan)
    for r in g.iter_rows(named=True):
        if r["diff"] is not None:
            mat[dims.index(r["dim"]), ns.index(r["n_train"])] = r["diff"]
    lim = max(0.05, np.nanmax(np.abs(mat))) if np.isfinite(mat).any() else 0.1
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    im = ax.imshow(mat, cmap=DIVERGING, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto", origin="lower")
    ax.grid(False)
    for i, j in itertools.product(range(len(dims)), range(len(ns))):
        v = mat[i, j]
        if np.isnan(v):
            ax.text(j, i, "pending", ha="center", va="center", color=INK_2, fontsize=8, style="italic")
            continue
        rgb = np.array(im.cmap(im.norm(v))[:3])
        lum = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
        ax.text(j, i, f"{v:+.3f}", ha="center", va="center", fontsize=9, color=INK if lum > 0.5 else SURFACE)
    ax.set_xticks(range(len(ns)), [compact(n) for n in ns])
    ax.set_yticks(range(len(dims)), [str(d) for d in dims])
    ax.set_xlabel("Training patients")
    ax.set_ylabel("Encoder width (dim)")
    for spine in ax.spines.values():
        spine.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=INK_2)
    cb.set_label("Retrieval − patient only (AUROC)", color=INK_2)
    ax.set_title(
        f"Retrieval advantage, {MAIN_NUM_TASKS} tasks (orange: retrieval better, blue: patient only better)",
        loc="left",
        fontsize=10,
        color=INK,
    )
    fig.tight_layout()
    save(fig, out, "fig3_retrieval_advantage")


def ablation_table(summary: pl.DataFrame) -> pl.DataFrame:
    ab = summary.filter(pl.col("dim") == ABLATION_DIM)
    wide = ab.pivot(on="arm", index=["num_tasks", "n_train"], values="mean")
    return wide.with_columns((pl.col("retrieval") - pl.col(REFERENCE_ARM)).alias("diff")).sort(
        "num_tasks", "n_train"
    )


def fig_ablation(summary, out) -> None:
    ab = summary.filter(pl.col("dim") == ABLATION_DIM)
    ts = sorted(ab["num_tasks"].unique())
    fig, axes = plt.subplots(1, len(ts), figsize=(3.0 * len(ts), 3.2), sharey=True, squeeze=False)
    for ax, t in zip(axes[0], ts, strict=True):
        for arm in ARMS:
            d = ab.filter((pl.col("num_tasks") == t) & (pl.col("arm") == arm)).sort("n_train")
            if d.is_empty():
                continue
            ax.plot(d["n_train"], d["mean"], color=ARM_COLOR[arm], zorder=2)
            errorbars(ax, d["n_train"], d, arm)
        log_x(ax, ab["n_train"].min(), ab["n_train"].max())
        ax.set_title(f"{t:,} tasks", loc="left", fontsize=10, color=INK)
        ax.set_xlabel("Training patients")
    axes[0][0].set_ylabel("Test AUROC (mean over tasks)")
    fig.suptitle(f"Task-count ablation at dim {ABLATION_DIM}", x=0.01, ha="left", fontsize=12, color=INK)
    arm_legend(fig, "Points: mean over 5 draws, bars ±1 std. Each task count draws its own random codes.")
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    save(fig, out, "fig4a_task_count_ablation")

    tab = ablation_table(summary).drop_nulls("diff")
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    ax.axhline(0, color=INK_2, lw=1, zorder=1)
    for t, color in zip(ts, TASK_RAMP[-len(ts) :] if len(ts) <= len(TASK_RAMP) else TASK_RAMP, strict=False):
        d = tab.filter(pl.col("num_tasks") == t).sort("n_train")
        if d.is_empty():
            continue
        ax.plot(
            d["n_train"],
            d["diff"],
            color=color,
            marker="o",
            ms=6,
            mec=SURFACE,
            mew=1.5,
            label=f"{t:,} tasks",
            zorder=2,
        )
    log_x(ax, ab["n_train"].min(), ab["n_train"].max())
    ax.set_xlabel("Training patients")
    ax.set_ylabel("Retrieval − patient only (AUROC)")
    ax.set_title(f"Retrieval advantage by task count, dim {ABLATION_DIM}", loc="left", fontsize=11, color=INK)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9)
    fig.tight_layout()
    save(fig, out, "fig4b_task_count_advantage")


def fig_control(summary, out) -> bool:
    """Retrieval vs its no-knowledge controls and patient only, CONTROL_NUM_TASKS tasks."""
    c = summary.filter(pl.col("num_tasks") == CONTROL_NUM_TASKS)
    dims = sorted(c.filter(pl.col("arm").is_in(CONTROL_ARMS))["dim"].unique())
    if not dims:
        return False
    arms = [*ARMS, *CONTROL_ARMS]
    ns = sorted(c.filter(pl.col("arm").is_in(CONTROL_ARMS))["n_train"].unique())
    fig, axes = plt.subplots(1, len(dims), figsize=(3.4 * len(dims), 3.4), sharey=True, squeeze=False)
    for ax, dim in zip(axes[0], dims, strict=True):
        for arm in arms:
            d = c.filter((pl.col("arm") == arm) & (pl.col("dim") == dim) & pl.col("n_train").is_in(ns)).sort(
                "n_train"
            )
            if d.is_empty():
                continue
            ax.plot(d["n_train"], d["mean"], color=CONTROL_COLOR[arm], zorder=2)
            ax.errorbar(
                d["n_train"],
                d["mean"],
                yerr=d["std"],
                fmt=CONTROL_MARKER[arm],
                ms=6,
                color=CONTROL_COLOR[arm],
                mec=SURFACE,
                mew=1.5,
                elinewidth=1.2,
                capsize=0,
                zorder=3,
            )
        log_x(ax, ns[0], ns[-1])
        ax.set_title(
            f"dim {dim}  ({compact(encoder_n_params(dim))} params)", loc="left", fontsize=10, color=INK
        )
        ax.set_xlabel("Training patients")
    axes[0][0].set_ylabel(f"Test AUROC (mean of {CONTROL_NUM_TASKS} tasks)")
    fig.suptitle(
        "Is the retrieval gain knowledge or extra capacity?", x=0.01, ha="left", fontsize=12, color=INK
    )
    handles = [
        plt.Line2D(
            [], [], color=CONTROL_COLOR[a], marker=CONTROL_MARKER[a], ms=6, mec=SURFACE, label=ARM_LABEL[a]
        )
        for a in arms
    ]
    fig.legend(handles=handles, loc="upper right", ncol=4, bbox_to_anchor=(0.99, 1.0), fontsize=9)
    fig.text(
        0.01,
        0.005,
        "Controls train and test the same retrieval model on random corpus documents or a content-free corpus. "
        "Points: mean over 5 draws, bars ±1 std.",
        color=INK_2,
        fontsize=8,
        va="bottom",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.9))
    save(fig, out, "fig5_knowledge_control")
    return True


def fmt(m, s) -> str:
    return "—" if m is None else f"{m:.3f} ± {s:.3f}"


def write_table(df: pl.DataFrame, out: Path, name: str, caption: str) -> None:
    cols = df.columns
    md = [f"**{caption}**", "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join("—" if v is None else str(v) for v in row) + " |" for row in df.iter_rows()]
    (out / f"{name}.md").write_text("\n".join(md) + "\n")
    tex = [
        "\\begin{table}[h]\\centering\\small",
        "\\begin{tabular}{" + "l" * len(cols) + "}\\toprule",
        " & ".join(c.replace("_", "\\_") for c in cols) + " \\\\\\midrule",
    ]
    tex += [
        " & ".join("--" if v is None else str(v).replace("±", "$\\pm$").replace("−", "$-$") for v in row)
        + " \\\\"
        for row in df.iter_rows()
    ]
    tex += ["\\bottomrule\\end{tabular}", f"\\caption{{{caption}}}", "\\end{table}"]
    (out / f"{name}.tex").write_text("\n".join(tex) + "\n")


def tables(summary, fits, out) -> None:
    main = summary.filter(pl.col("num_tasks") == MAIN_NUM_TASKS)
    ns = sorted(main["n_train"].unique())
    for arm in ARMS:
        rows = []
        for dim in sorted(main["dim"].unique()):
            r = {"dim": dim, "params": compact(encoder_n_params(dim))}
            for n in ns:
                c = main.filter((pl.col("arm") == arm) & (pl.col("dim") == dim) & (pl.col("n_train") == n))
                r[f"N={compact(n)}"] = fmt(c["mean"][0], c["std"][0]) if c.height else "—"
            rows.append(r)
        write_table(
            pl.DataFrame(rows),
            out,
            f"table1_{arm}_auroc",
            f"{ARM_LABEL[arm]}: test AUROC (mean ± std over draws), {MAIN_NUM_TASKS} tasks",
        )
    g = advantage_grid(summary)
    adv = (
        g.with_columns(pl.col("n_train").map_elements(lambda n: f"N={compact(n)}", return_dtype=pl.String))
        .with_columns(pl.col("diff").map_elements(lambda v: f"{v:+.3f}", return_dtype=pl.String))
        .pivot(on="n_train", index="dim", values="diff")
        .sort("dim")
    )
    write_table(
        adv,
        out,
        "table2_retrieval_advantage",
        f"Retrieval − patient only, test AUROC, {MAIN_NUM_TASKS} tasks",
    )
    ab = ablation_table(summary).with_columns(
        pl.col("n_train").map_elements(compact, return_dtype=pl.String),
        pl.col(REFERENCE_ARM, "retrieval").round(3),
        pl.col("diff").map_elements(lambda v: f"{v:+.3f}", return_dtype=pl.String),
    )
    write_table(
        ab, out, "table3_task_count_ablation", f"Task-count ablation at dim {ABLATION_DIM}: mean test AUROC"
    )
    c = summary.filter(pl.col("num_tasks") == CONTROL_NUM_TASKS)
    dims = sorted(c.filter(pl.col("arm").is_in(CONTROL_ARMS))["dim"].unique())
    if dims:
        ns = sorted(c.filter(pl.col("arm").is_in(CONTROL_ARMS))["n_train"].unique())
        rows = []
        for dim, n in itertools.product(dims, ns):
            r = {"dim": dim, "n_train": compact(n)}
            for arm in [*ARMS, *CONTROL_ARMS]:
                v = c.filter((pl.col("arm") == arm) & (pl.col("dim") == dim) & (pl.col("n_train") == n))
                r[ARM_LABEL[arm]] = fmt(v["mean"][0], v["std"][0]) if v.height else "—"
            rows.append(r)
        write_table(
            pl.DataFrame(rows),
            out,
            "table5_knowledge_control",
            f"Retrieval vs no-knowledge controls: test AUROC (mean ± std over draws), {CONTROL_NUM_TASKS} tasks",
        )
    if fits:
        text = "\n\n".join(f"## {ARM_LABEL[a]}\n\n```\n{s}\n```" for a, s in fits.items())
        (out / "table4_fitted_laws.md").write_text(f"# Fitted laws ({MAIN_NUM_TASKS} tasks)\n\n{text}\n")


def main() -> None:
    global MAIN_NUM_TASKS
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument(
        "--main-tasks", type=int, default=MAIN_NUM_TASKS, help="task count of the main grid to plot"
    )
    args = parser.parse_args()
    MAIN_NUM_TASKS = args.main_tasks
    figs, tabs = args.out / "figures", args.out / "tables"
    figs.mkdir(parents=True, exist_ok=True)
    tabs.mkdir(parents=True, exist_ok=True)

    results = collect_results()
    summary = summarize(results)
    fits, curves = fitted_curves(results, summary)
    print(
        f"{results.height} evals; main-grid fit: {'yes' if curves is not None else 'pending (grid incomplete)'}"
    )
    fig_scaling(summary, curves, figs, axis="data")
    fig_scaling(summary, curves, figs, axis="capacity")
    fig_advantage(summary, figs)
    fig_ablation(summary, figs)
    fig_control(summary, figs)
    tables(summary, fits, tabs)
    print(f"wrote {figs} and {tabs}")


if __name__ == "__main__":
    main()
