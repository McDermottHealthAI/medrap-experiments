"""Figures for the follow-up findings: bias check, v2 pilot, document value, headroom.

Complements plot_results.py (same style) with figures for results that the README otherwise
shows only as tables:

- fig6_bias_check: retrieval - patient_only held-out BCE as trained vs after a per-task logit
  offset (outputs/bias_check.parquet, from scripts/bias_check.py).
- fig7_bias_fix: patient_only round 1 vs v2 (head bias initialized from the base rate), AUROC and
  test loss vs training patients.
- fig8_v2_pilot: v2 arms vs training patients (patient_only, null docs, retrieval_qwen3).
- fig9_document_value: retrieval - null docs per cell, round 1 vs v2, one panel per retriever.
- fig10_headroom: headroom vs retrieval value across cells (outputs/headroom_tasks625.parquet, from
  scripts/headroom.py --tasks 625 --retrieval retrieval retrieval_qwen3).

``--metric loss`` redraws fig8-10 on test loss (BCE) as fig8-10 *_loss; fig6 is already loss and fig7
shows both. The loss headroom needs scripts/headroom.py --metric loss first.

Usage:  cd mimic_iv_scaling_laws && .venv/bin/python scripts/plot_findings.py [--out figures] [--metric loss]
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.ticker import FuncFormatter
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from aggregate_results import collect_results, encoder_n_params
from plot_results import GRID, INK, INK_2, SURFACE, compact, log_x

TASKS = 625
PILOT_DIMS = ("4", "64")
PILOT_NS = (1000, 10000, 100000)
DIMS = ("4", "8", "16", "32", "64", "128")
# Arm colors follow plot_results (patient_only blue, null docs yellow); retrieval_qwen3 takes the
# next free categorical slot (magenta). Validated together, all pairs, light surface: CVD ΔE 13.0,
# normal-vision ΔE 19.6; yellow and magenta sit below 3:1 contrast, so markers differ per arm and
# the README carries the table view.
ARM = {
    "patient_only": ("Patient only", "#2a78d6", "o"),
    "retrieval_null_docs": ("Retrieval, null docs", "#eda100", "^"),
    "retrieval_qwen3": ("Retrieval, frozen Qwen3 query", "#e87ba4", "D"),
}
# Two steps of one blue ramp for ordered comparisons (before -> after, round 1 -> v2).
LIGHT, DARK = "#6da7ec", "#184f95"
# What fig8-10 show; --metric picks one. Differences are always first arm - second arm, raw, so for
# loss a negative difference means the first arm is better.
METRICS = {
    "auroc": {"col": "auroc", "axis": "Test AUROC", "short": "AUROC", "suffix": "", "helps": "Positive"},
    "loss": {"col": "loss", "axis": "Test loss (BCE)", "short": "test loss", "suffix": "_loss", "helps": "Negative"},
}
METRIC = METRICS["auroc"]
WIDTH_RAMP = dict(zip(DIMS, ["#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"], strict=True))


def save(fig, out: Path, name: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}{METRIC['suffix']}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def note(fig, text: str) -> None:
    fig.text(0.01, 0.005, text, color=INK_2, fontsize=8, va="bottom")


def ci(d: np.ndarray) -> tuple[float, float]:
    """Mean and 95% t half-width."""
    return float(d.mean()), float(stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d)))


def load(root: str) -> pl.DataFrame:
    dims = {encoder_n_params(d): d for d in DIMS}
    return (
        collect_results(root=root)
        .filter(pl.col("num_tasks") == TASKS)
        .select(
            "arm",
            pl.col("model_size__n_params").replace_strict(dims, default=None).alias("dim"),
            pl.col("dataset_size__n_train_patients").alias("n"),
            pl.col("optimizer_seed").alias("draw"),
            pl.col("test_metric__auroc").alias("auroc"),
            pl.col("test_loss__bce").alias("loss"),
        )
        .filter(pl.col("dim").is_in(PILOT_DIMS) & pl.col("n").is_in(PILOT_NS))
    )


def paired(r: pl.DataFrame, a: str, b: str, metric: str = "auroc") -> pl.DataFrame:
    j = r.filter(pl.col("arm") == a).join(r.filter(pl.col("arm") == b), on=["dim", "n", "draw"], suffix="_b")
    rows = []
    for (dim, n), g in j.group_by("dim", "n"):
        if g.height >= 2:
            m, h = ci((g[metric] - g[f"{metric}_b"]).to_numpy())
            rows.append({"dim": dim, "n": n, "mean": m, "half": h})
    return pl.DataFrame(rows)


def zero_line(ax, vertical: bool = False) -> None:
    (ax.axvline if vertical else ax.axhline)(0, color=INK_2, lw=1, zorder=1)


def fig_bias_check(out: Path) -> None:
    bc = pl.read_parquet("outputs/bias_check.parquet")
    j = bc.filter(pl.col("arm") == "retrieval").join(
        bc.filter(pl.col("arm") == "patient_only"), on=["dim", "n_train", "num_tasks", "draw"], suffix="_po"
    )
    dims = sorted(j["dim"].unique())
    ns = sorted(j["n_train"].unique())
    fig, axes = plt.subplots(1, len(dims), figsize=(3.6 * len(dims), 2.9), squeeze=False)
    for ax, dim in zip(axes[0], dims, strict=True):
        for y, n in enumerate(ns):
            g = j.filter((pl.col("dim") == dim) & (pl.col("n_train") == n))
            raw = ci((g["loss_raw"] - g["loss_raw_po"]).to_numpy())
            rec = ci((g["loss_recal"] - g["loss_recal_po"]).to_numpy())
            ax.plot([raw[0], rec[0]], [y, y], color=GRID, lw=2, zorder=1)
            for (m, h), c, mk in ((raw, DARK, "o"), (rec, LIGHT, "s")):
                ax.errorbar(m, y, xerr=h, fmt=mk, ms=7, color=c, mec=SURFACE, mew=1.5, elinewidth=1.2, zorder=3)
        zero_line(ax, vertical=True)
        ax.set_yticks(range(len(ns)), [f"{compact(n)} patients" for n in ns])
        ax.set_ylim(-0.6, len(ns) - 0.4)
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        ax.set_title(f"dim {dim}", loc="left", fontsize=10, color=INK)
        ax.set_xlabel("Retrieval − patient only, held-out BCE")
    fig.suptitle(
        "The small-model loss gap is an unlearned base rate: one offset per task removes most of it",
        x=0.01,
        ha="left",
        fontsize=12,
        color=INK,
    )
    handles = [
        plt.Line2D([], [], color=DARK, marker="o", ls="", ms=7, mec=SURFACE, label="As trained"),
        plt.Line2D([], [], color=LIGHT, marker="s", ls="", ms=7, mec=SURFACE, label="After a per-task logit offset"),
    ]
    fig.legend(handles=handles, loc="upper left", ncol=2, bbox_to_anchor=(0.005, 0.92), fontsize=9)
    note(
        fig,
        "Round 1, 25 tasks. Negative = retrieval lower loss. Offsets fit on the tuning split, everything else "
        "frozen. Points: mean over 5 paired draws, bars 95% CI. Each panel has its own x-scale.",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.88))
    save(fig, out, "fig6_bias_check")


def fig_bias_fix(v1: pl.DataFrame, v2: pl.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(2, len(PILOT_DIMS), figsize=(7.2, 5.6), sharex=True, sharey="row")
    for col, dim in enumerate(PILOT_DIMS):
        for row, metric in enumerate(("auroc", "loss")):
            ax = axes[row][col]
            for r, c, mk, label in ((v1, LIGHT, "o", "Round 1"), (v2, DARK, "s", "v2: bias from base rate")):
                s = (
                    r.filter((pl.col("arm") == "patient_only") & (pl.col("dim") == dim))
                    .group_by("n")
                    .agg(pl.col(metric).mean().alias("m"), pl.col(metric).std().alias("s"))
                    .sort("n")
                )
                ax.plot(s["n"], s["m"], color=c, zorder=2, ls="--" if label == "Round 1" else "-")
                ax.errorbar(
                    s["n"], s["m"], yerr=s["s"], fmt=mk, ms=6, color=c, mec=SURFACE, mew=1.5,
                    elinewidth=1.2, label=label, zorder=3,
                )
            log_x(ax, PILOT_NS[0], PILOT_NS[-1])
            if metric == "loss":
                ax.set_yscale("log")
                ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
                ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: f"{v:g}" if round(v, 3) in (0.02, 0.03, 0.05, 0.2, 0.3) else ""))
                ax.set_xlabel("Training patients")
            if row == 0:
                ax.set_title(f"Patient only, dim {dim}", loc="left", fontsize=10, color=INK)
        axes[0][0].set_ylabel("Test AUROC")
        axes[1][0].set_ylabel("Test loss (BCE, log)")
    fig.suptitle("Initializing the head bias from the base rate", x=0.01, ha="left", fontsize=12, color=INK)
    h, lab = axes[0][0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper left", ncol=2, bbox_to_anchor=(0.005, 0.95), fontsize=9)
    note(fig, f"{TASKS} tasks, same task draws in both rounds. Points: mean over 5 draws, bars ±1 std.")
    fig.tight_layout(rect=(0, 0.03, 1, 0.9))
    save(fig, out, "fig7_bias_fix")


def fig_v2_pilot(v2: pl.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, len(PILOT_DIMS), figsize=(7.6, 3.6), sharey=True)
    for ax, dim in zip(axes, PILOT_DIMS, strict=True):
        for arm, (label, color, marker) in ARM.items():
            s = (
                v2.filter((pl.col("arm") == arm) & (pl.col("dim") == dim))
                .group_by("n")
                .agg(pl.col(METRIC["col"]).mean().alias("m"), pl.col(METRIC["col"]).std().alias("s"))
                .sort("n")
            )
            ax.plot(s["n"], s["m"], color=color, zorder=2)
            ax.errorbar(
                s["n"], s["m"], yerr=s["s"], fmt=marker, ms=6, color=color, mec=SURFACE, mew=1.5,
                elinewidth=1.2, label=label, zorder=3,
            )
        log_x(ax, PILOT_NS[0], PILOT_NS[-1])
        ax.set_title(f"dim {dim}  ({compact(encoder_n_params(dim))} params)", loc="left", fontsize=10, color=INK)
        ax.set_xlabel("Training patients")
    axes[0].set_ylabel(f"{METRIC['axis']} (mean of {TASKS} tasks)")
    fig.suptitle("v2 pilot: all arms with the bias fix", x=0.01, ha="left", fontsize=12, color=INK)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper left", ncol=3, bbox_to_anchor=(0.005, 0.92), fontsize=9)
    note(fig, "Learned-query retrieval is omitted here (see fig9). Points: mean over 5 draws, bars ±1 std.")
    fig.tight_layout(rect=(0, 0.04, 1, 0.84))
    save(fig, out, "fig8_v2_pilot")


def fig_document_value(v1: pl.DataFrame, v2: pl.DataFrame, out: Path) -> None:
    retrievers = (("retrieval", "Learned query"), ("retrieval_qwen3", "Frozen Qwen3 query"))
    cells = [(d, n) for d in PILOT_DIMS for n in PILOT_NS]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.8), sharey=True, sharex=True)
    for ax, (arm, title) in zip(axes, retrievers, strict=True):
        for k, (r, c, mk, label) in enumerate(((v1, LIGHT, "o", "Round 1"), (v2, DARK, "s", "v2 (bias fix)"))):
            p = paired(r, arm, "retrieval_null_docs", METRIC["col"])
            for y, (dim, n) in enumerate(cells):
                row = p.filter((pl.col("dim") == dim) & (pl.col("n") == n))
                if row.is_empty():
                    continue
                ax.errorbar(
                    row["mean"][0], y + (k - 0.5) * 0.3, xerr=row["half"][0], fmt=mk, ms=6, color=c,
                    mec=SURFACE, mew=1.5, elinewidth=1.2, label=label if y == 0 else None, zorder=3,
                )
        zero_line(ax, vertical=True)
        ax.set_yticks(range(len(cells)), [f"dim {d}, {compact(n)}" for d, n in cells])
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        ax.set_xlabel(f"Retrieval − null docs, {METRIC['short']}")
    fig.suptitle(
        "Value of the documents themselves (architecture held fixed)", x=0.01, ha="left", fontsize=12, color=INK
    )
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper left", ncol=2, bbox_to_anchor=(0.005, 0.93), fontsize=9)
    note(fig, f"{TASKS} tasks. {METRIC['helps']} = documents help. Points: mean over 5 paired draws, bars 95% CI.")
    fig.tight_layout(rect=(0, 0.04, 1, 0.86))
    save(fig, out, "fig9_document_value")


def fig_headroom(out: Path) -> None:
    # headroom.py stores loss values sign-flipped (higher = better), so its value columns are
    # patient only - retrieval loss: positive = retrieval better under either metric.
    loss = METRIC["col"] == "loss"
    h = pl.read_parquet(f"outputs/headroom_tasks{TASKS}{METRIC['suffix']}.parquet").with_columns(
        pl.col("dim").cast(pl.String)
    )
    naive = "Test loss (no fit)" if loss else "1 − AUROC (no fit)"
    total = "Patient only − retrieval" if loss else "Retrieval − patient only"
    docs = "Null docs − frozen Qwen3" if loss else "Frozen Qwen3 − null docs"
    panels = (
        ("headroom_data", "value_retrieval", "Fitted data-axis headroom", total),
        ("naive_gap", "value_retrieval", naive, total),
        ("naive_gap", "knowledge_retrieval_qwen3", naive, docs),
    )
    fig, axes = plt.subplots(1, len(panels), figsize=(11, 3.7))
    for ax, (x, y, xl, yl) in zip(axes, panels, strict=True):
        d = h.select("dim", "n", x, y, "data_fit_at_bound").drop_nulls().drop_nans()
        for dim in DIMS:
            g = d.filter(pl.col("dim") == dim)
            if g.is_empty():
                continue
            hollow = g["data_fit_at_bound"].to_numpy() if x == "headroom_data" else np.zeros(g.height, bool)
            for flag in (False, True):
                gg = g.filter(pl.Series(hollow == flag))
                ax.scatter(
                    gg[x], gg[y], s=42, zorder=3, label=f"dim {dim}" if not flag else None,
                    facecolors="none" if flag else WIDTH_RAMP[dim], edgecolors=WIDTH_RAMP[dim],
                    linewidths=1.5 if flag else 0.8,
                )
        rho, p = stats.spearmanr(d[x], d[y])
        ax.set_title(f"Spearman ρ = {rho:.2f}  (p = {p:.2g}, {d.height} cells)", loc="left", fontsize=9, color=INK)
        zero_line(ax)
        ax.set_xlabel(xl)
        ax.set_ylabel(f"{yl}, {METRIC['short']}")
    fig.suptitle(f"Does headroom predict what retrieval adds? (round 1, 625 tasks, {METRIC['short']})", x=0.01,
                 ha="left", fontsize=12, color=INK)
    hd, lab = axes[1].get_legend_handles_labels()
    fig.legend(hd, lab, loc="upper left", ncol=6, bbox_to_anchor=(0.005, 0.93), fontsize=9)
    note(fig, "One point per (width, training patients) cell; mean over 5 draws; positive = retrieval better. Hollow: "
              "the data-axis fit's asymptote sits on its bound (" + ("loss 0" if loss else "AUROC 1")
              + "), so its headroom equals the naive gap.")
    fig.tight_layout(rect=(0, 0.04, 1, 0.84))
    save(fig, out, "fig10_headroom")


def main() -> None:
    global METRIC
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="figures")
    parser.add_argument("--metric", choices=sorted(METRICS), default="auroc", help="metric of fig8-10")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    v1, v2 = load("outputs"), load("outputs_v2")
    fig_bias_check(out)
    fig_bias_fix(v1, v2, out)
    METRIC = METRICS[args.metric]
    fig_v2_pilot(v2, out)
    fig_document_value(v1, v2, out)
    fig_headroom(out)
    print(f"wrote {out}/fig6-10{METRIC['suffix']}")


if __name__ == "__main__":
    main()
