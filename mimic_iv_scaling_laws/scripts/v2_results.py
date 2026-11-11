"""Tables and figures for the v2 grid (head bias initialized from the base rate), outputs_v2/.

Reads every finished v2 eval; a cell (arm, capacity, training patients, task count) is used only
once all 5 draws exist, so the output can be regenerated while the grid is still running.

- fig11_v2_data_scaling: AUROC vs training patients per capacity, all arms, 625 tasks.
- fig12_v2_value_grid: architecture value (null docs - patient_only), document value
  (retrieval_qwen3 - null docs) and total value (retrieval_qwen3 - patient_only) over
  capacity x training patients, 625 tasks.
- fig13_v2_task_count: patient_only AUROC and architecture value vs training patients, one
  line per task count, one panel per task-axis capacity.
- fig14_v2_headroom: headroom (from scripts/headroom.py, run per task count first) vs
  architecture and document value, 625 tasks.
- v2_tables.md: the README tables for the same results.

Usage:
    cd mimic_iv_scaling_laws
    for T in 5 25 125 625 2500; do
        .venv/bin/python scripts/headroom.py --outputs outputs_v2 --tasks $T \\
            --retrieval retrieval_null_docs retrieval_qwen3
    done
    .venv/bin/python scripts/v2_results.py [--out figures_v2]
"""

import argparse
import itertools
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.colors import TwoSlopeNorm
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from aggregate_results import collect_results, encoder_n_params
from plot_findings import ARM, ci, note, save, zero_line
from plot_results import DIVERGING, INK, INK_2, SURFACE, TASK_RAMP, compact, log_x

ROOT = "outputs_v2"
N_DRAWS = 5
MAIN_TASKS = 625
CAPACITIES = ("4", "8", "16", "32", "64", "128", "128x2x4", "256x4x8")  # scripts/grid.sh V2_CAPACITIES
TASK_COUNTS = (5, 25, 125, 625, 2500)
TASK_CAPACITIES = ("4", "16", "64", "128x2x4", "256x4x8")  # V2_TASK_CAPACITIES
NS = (1000, 3000, 10000, 30000, 100000, 153220)
# Sequential blue ramp, light -> dark, one step per capacity (magnitude, not identity).
CAP_RAMP = dict(
    zip(CAPACITIES, ["#b9d5f7", "#8bbaf1", "#5c9cea", "#3987e5", "#256abf", "#184f95", "#0d366b", "#071f40"], strict=True)
)
TASK_COLOR = dict(zip(TASK_COUNTS, TASK_RAMP, strict=True))
VALUES = {  # name: (minuend, subtrahend, label)
    "architecture": ("retrieval_null_docs", "patient_only", "Architecture: null docs − patient only"),
    "documents": ("retrieval_qwen3", "retrieval_null_docs", "Documents: Qwen3 − null docs"),
    "total": ("retrieval_qwen3", "patient_only", "Total: Qwen3 − patient only"),
}


def cap_label(cap: str) -> str:
    return f"{cap} ({compact(encoder_n_params(cap))})"


def load() -> pl.DataFrame:
    caps = {encoder_n_params(c): c for c in CAPACITIES}
    r = (
        collect_results(root=ROOT)
        .filter(pl.col("arm").is_in(list(ARM)))
        .select(
            "arm",
            pl.col("model_size__n_params").replace_strict(caps).alias("cap"),
            pl.col("dataset_size__n_train_patients").alias("n"),
            pl.col("num_tasks").alias("t"),
            pl.col("optimizer_seed").alias("draw"),
            pl.col("test_metric__auroc").alias("auroc"),
        )
    )
    complete = r.group_by("arm", "cap", "n", "t").len().filter(pl.col("len") == N_DRAWS).drop("len")
    return r.join(complete, on=["arm", "cap", "n", "t"])


def paired(r: pl.DataFrame, value: str, t: int) -> pl.DataFrame:
    """Per (cap, n): mean, 95% CI and wins of the paired per-draw difference for one value."""
    a, b, _ = VALUES[value]
    r = r.filter(pl.col("t") == t)
    j = r.filter(pl.col("arm") == a).join(r.filter(pl.col("arm") == b), on=["cap", "n", "draw"], suffix="_b")
    rows = []
    for (cap, n), g in j.group_by("cap", "n"):
        d = (g["auroc"] - g["auroc_b"]).to_numpy()
        m, h = ci(d)
        rows.append({"cap": cap, "n": n, "mean": m, "lo": m - h, "hi": m + h, "wins": int((d > 0).sum())})
    return pl.DataFrame(rows, schema={"cap": pl.String, "n": pl.Int64, "mean": pl.Float64, "lo": pl.Float64,
                                      "hi": pl.Float64, "wins": pl.Int64})


def md_cell(row: dict | None) -> str:
    if row is None:
        return "pending"
    s = f"{row['mean']:+.3f}"
    if row["lo"] > 0 or row["hi"] < 0:
        s = f"**{s}**"
    return f"{s} <sub>[{row['lo']:+.3f}, {row['hi']:+.3f}] · {row['wins']}/{N_DRAWS}</sub>"


def md_value_table(p: pl.DataFrame, caps, row_label=lambda c: f"{c}") -> list[str]:
    lines = ["| Capacity ↓ / patients → | " + " | ".join(compact(n) for n in NS) + " |", "|---" * (len(NS) + 1) + "|"]
    for cap in caps:
        cells = []
        for n in NS:
            g = p.filter((pl.col("cap") == cap) & (pl.col("n") == n))
            cells.append(md_cell(g.row(0, named=True) if g.height else None))
        lines.append(f"| {row_label(cap)} | " + " | ".join(cells) + " |")
    return lines


def tables(r: pl.DataFrame, out: Path) -> None:
    lines = [f"## Mean test AUROC, {MAIN_TASKS} tasks\n"]
    m = r.filter(pl.col("t") == MAIN_TASKS).group_by("arm", "cap", "n").agg(pl.col("auroc").mean())
    lines += ["| Capacity | Arm | " + " | ".join(compact(n) for n in NS) + " |", "|---" * (len(NS) + 2) + "|"]
    for cap in CAPACITIES:
        for i, arm in enumerate(ARM):
            vals = []
            for n in NS:
                v = m.filter((pl.col("arm") == arm) & (pl.col("cap") == cap) & (pl.col("n") == n))["auroc"]
                vals.append(f"{v[0]:.3f}" if len(v) else "pending")
            lines.append(f"| {cap_label(cap) if i == 0 else ''} | `{arm}` | " + " | ".join(vals) + " |")
    for value, (_, _, label) in VALUES.items():
        lines += [f"\n## {label}, {MAIN_TASKS} tasks\n", *md_value_table(paired(r, value, MAIN_TASKS), CAPACITIES, cap_label)]
    for t in TASK_COUNTS:
        lines += [f"\n## Architecture value, {t} tasks\n", *md_value_table(paired(r, "architecture", t), TASK_CAPACITIES)]
    lines.append("\n## patient_only mean AUROC by task count\n")
    po = r.filter(pl.col("arm") == "patient_only").group_by("cap", "n", "t").agg(pl.col("auroc").mean())
    lines += ["| Capacity | Tasks | " + " | ".join(compact(n) for n in NS) + " |", "|---" * (len(NS) + 2) + "|"]
    for cap in TASK_CAPACITIES:
        for i, t in enumerate(TASK_COUNTS):
            v = [po.filter((pl.col("cap") == cap) & (pl.col("n") == n) & (pl.col("t") == t))["auroc"] for n in NS]
            lines.append(f"| {cap if i == 0 else ''} | {t} | " + " | ".join(f"{x[0]:.3f}" if len(x) else "–" for x in v) + " |")
    lines.append("\n## Headroom fits and correlations by task count\n")
    lines += ["| Tasks | Data fits at A = 1 | Capacity fits at A = 1 | Headroom | Value | Cells | Spearman ρ | p |",
              "|---|---|---|---|---|---|---|---|"]
    for t in TASK_COUNTS:
        hp = Path(ROOT) / f"headroom_tasks{t}.parquet"
        if not hp.exists():
            continue
        h, c = pl.read_parquet(hp), pl.read_parquet(Path(ROOT) / f"headroom_tasks{t}_correlations.parquet")
        bounds = (f"{h['data_fit_at_bound'].sum()}/{h['headroom_data'].is_not_nan().sum()}",
                  f"{h['capacity_fit_at_bound'].sum()}/{h['headroom_capacity'].is_not_nan().sum()}")
        for i, row in enumerate(c.iter_rows(named=True)):
            lines.append(
                f"| {t if i == 0 else ''} | {bounds[0] if i == 0 else ''} | {bounds[1] if i == 0 else ''} | "
                f"{row['headroom']} | {row['retrieval_metric']} | {row['cells']} | {row['spearman']:+.2f} | {row['p']:.2g} |"
            )
    (out / "v2_tables.md").write_text("\n".join(lines) + "\n")


def fig_data_scaling(r: pl.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.2), sharex=True, sharey=True)
    for ax, cap in zip(axes.flat, CAPACITIES, strict=True):
        for arm, (label, color, marker) in ARM.items():
            s = (
                r.filter((pl.col("arm") == arm) & (pl.col("cap") == cap) & (pl.col("t") == MAIN_TASKS))
                .group_by("n")
                .agg(pl.col("auroc").mean().alias("m"), pl.col("auroc").std().alias("s"))
                .sort("n")
            )
            ax.plot(s["n"], s["m"], color=color, zorder=2)
            ax.errorbar(s["n"], s["m"], yerr=s["s"], fmt=marker, ms=5, color=color, mec=SURFACE, mew=1.2,
                        elinewidth=1.1, label=label, zorder=3)
        log_x(ax, NS[0], NS[-1])
        ax.set_title(f"capacity {cap_label(cap)}", loc="left", fontsize=10, color=INK)
    for ax in axes[1]:
        ax.set_xlabel("Training patients")
    for ax in axes[:, 0]:
        ax.set_ylabel("Test AUROC")
    fig.suptitle(f"v2 data scaling, {MAIN_TASKS} tasks: every capacity", x=0.01, ha="left", fontsize=12, color=INK)
    h, lab = axes[0][0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper left", ncol=3, bbox_to_anchor=(0.005, 0.955), fontsize=9)
    note(fig, "Capacity: width, or width × layers × heads; encoder parameters in parentheses. Points: mean over "
              "5 draws, bars ±1 std. Missing points: runs still pending.")
    fig.tight_layout(rect=(0, 0.03, 1, 0.91))
    save(fig, out, "fig11_v2_data_scaling")


def fig_value_grid(r: pl.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    lim = 0.1
    for ax, (value, (_, _, label)) in zip(axes, VALUES.items(), strict=True):
        p = paired(r, value, MAIN_TASKS)
        mat = np.full((len(CAPACITIES), len(NS)), np.nan)
        sig = np.zeros_like(mat, bool)
        for row in p.iter_rows(named=True):
            i, j = CAPACITIES.index(row["cap"]), NS.index(row["n"])
            mat[i, j], sig[i, j] = row["mean"], row["lo"] > 0 or row["hi"] < 0
        im = ax.imshow(np.clip(mat, -lim, lim), cmap=DIVERGING, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto",
                       origin="lower")
        ax.grid(False)
        for i, j in itertools.product(range(len(CAPACITIES)), range(len(NS))):
            v = mat[i, j]
            if np.isnan(v):
                ax.text(j, i, "pending", ha="center", va="center", color=INK_2, fontsize=7.5, style="italic")
                continue
            rgb = np.array(im.cmap(im.norm(np.clip(v, -lim, lim)))[:3])
            lum = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
            ax.text(j, i, f"{v:+.3f}", ha="center", va="center", fontsize=8,
                    fontweight="bold" if sig[i, j] else "normal", color=INK if lum > 0.5 else SURFACE)
        ax.set_xticks(range(len(NS)), [compact(n) for n in NS])
        ax.set_yticks(range(len(CAPACITIES)), CAPACITIES)
        ax.set_xlabel("Training patients")
        ax.set_title(label, loc="left", fontsize=10, color=INK)
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[0].set_ylabel("Capacity")
    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=INK_2)
    cb.set_label("Test AUROC difference (clipped at ±0.1)", color=INK_2)
    fig.suptitle(f"What retrieval adds, split into architecture and documents ({MAIN_TASKS} tasks)", x=0.01,
                 ha="left", fontsize=12, color=INK)
    note(fig, "Orange: the first arm is better; blue: the second is. Bold: 95% CI over 5 paired draws excludes 0. "
              "Architecture + documents = total.")
    fig.subplots_adjust(left=0.06, right=0.9, bottom=0.17, top=0.86, wspace=0.08)
    save(fig, out, "fig12_v2_value_grid")


def fig_task_count(r: pl.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(2, len(TASK_CAPACITIES), figsize=(15, 6.2), sharex=True, sharey="row")
    for col, cap in enumerate(TASK_CAPACITIES):
        top, bottom = axes[0][col], axes[1][col]
        for t in TASK_COUNTS:
            s = (
                r.filter((pl.col("arm") == "patient_only") & (pl.col("cap") == cap) & (pl.col("t") == t))
                .group_by("n").agg(pl.col("auroc").mean().alias("m")).sort("n")
            )
            top.plot(s["n"], s["m"], color=TASK_COLOR[t], marker="o", ms=5, mec=SURFACE, mew=1.2, label=f"{t} tasks")
            p = paired(r, "architecture", t).filter(pl.col("cap") == cap).sort("n")
            bottom.plot(p["n"], p["mean"], color=TASK_COLOR[t], marker="o", ms=5, mec=SURFACE, mew=1.2)
        zero_line(bottom)
        log_x(top, NS[0], NS[-1])
        top.set_title(f"capacity {cap_label(cap)}", loc="left", fontsize=10, color=INK)
        bottom.set_xlabel("Training patients")
    axes[0][0].set_ylabel("Patient only, test AUROC")
    axes[1][0].set_ylabel("Null docs − patient only")
    fig.suptitle("Task-count axis: does the number of tasks change scaling or the architecture's value?", x=0.01,
                 ha="left", fontsize=12, color=INK)
    h, lab = axes[0][0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper left", ncol=5, bbox_to_anchor=(0.005, 0.955), fontsize=9)
    note(fig, "Mean over 5 draws; each draw samples its own task set. Darker = more tasks. 625 tasks is the main grid.")
    fig.tight_layout(rect=(0, 0.03, 1, 0.91))
    save(fig, out, "fig13_v2_task_count")


def fig_headroom(out: Path) -> None:
    h = pl.read_parquet(Path(ROOT) / f"headroom_tasks{MAIN_TASKS}.parquet")
    xs = (("headroom_data", "Fitted data-axis headroom", "data_fit_at_bound"),
          ("headroom_capacity", "Fitted capacity-axis headroom", "capacity_fit_at_bound"),
          ("naive_gap", "1 − AUROC (no fit)", None))
    ys = (("value_retrieval_null_docs", "Architecture value"), ("knowledge_retrieval_qwen3", "Document value"))
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex="col", sharey="row")
    for (row, (y, yl)), (col, (x, xl, bound)) in itertools.product(enumerate(ys), enumerate(xs)):
        ax = axes[row][col]
        if y not in h.columns:
            continue
        d = h.select("dim", x, y, *([bound] if bound else [])).drop_nulls().drop_nans()
        for cap in CAPACITIES:
            g = d.filter(pl.col("dim") == cap)
            hollow = g[bound].to_numpy() if bound else np.zeros(g.height, bool)
            for flag in (False, True):
                gg = g.filter(pl.Series(hollow == flag))
                ax.scatter(gg[x], gg[y], s=40, zorder=3, label=cap if not flag else None,
                           facecolors="none" if flag else CAP_RAMP[cap], edgecolors=CAP_RAMP[cap],
                           linewidths=1.5 if flag else 0.8)
        if d.height >= 4:
            rho, p = stats.spearmanr(d[x], d[y])
            ax.set_title(f"ρ = {rho:+.2f}  (p = {p:.2g}, {d.height} cells)", loc="left", fontsize=9, color=INK)
        zero_line(ax)
        if row == 1:
            ax.set_xlabel(xl)
        if col == 0:
            ax.set_ylabel(f"{yl}, AUROC")
    fig.suptitle(f"v2: does headroom predict what retrieval adds? ({MAIN_TASKS} tasks)", x=0.01, ha="left",
                 fontsize=12, color=INK)
    hd, lab = axes[0][0].get_legend_handles_labels()
    fig.legend(hd, lab, loc="upper left", ncol=8, bbox_to_anchor=(0.005, 0.955), fontsize=9, title="Capacity",
               title_fontsize=9, alignment="left")
    note(fig, "One point per (capacity, training patients) cell; mean over 5 draws. Hollow: the fit's asymptote "
              "sits on the A = 1 bound, so its headroom equals 1 − AUROC. Document value: cells with all Qwen3 draws.")
    fig.tight_layout(rect=(0, 0.03, 1, 0.88))
    save(fig, out, "fig14_v2_headroom")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="figures_v2")
    out = Path(parser.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    r = load()
    print(r.group_by("arm", "t").len().sort("arm", "t"))
    tables(r, out)
    fig_data_scaling(r, out)
    fig_value_grid(r, out)
    fig_task_count(r, out)
    fig_headroom(out)
    print(f"wrote {out}/v2_tables.md and fig11-14")


if __name__ == "__main__":
    main()
