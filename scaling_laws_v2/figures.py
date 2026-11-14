"""Figures for RESULTS.md, one per question, all from paired per-draw test losses.

    python scaling_laws_v2/figures.py [OUTPUTS_DIR]

Δ = L(patient only) − L(model) as a % of patient only's test loss, paired per draw (same tasks,
patients and seed), mean over draws with a 95% t-interval; positive means the model is better.
Cold models are paired with patient only, warm-started ones with patient only continued from the
same checkpoint for the same epochs (patient_only_warm). Writes into figures/:

    readers.png         Δ for each reader, with its model trained on random passages
    k.png               Δ against the number of passages K, retrieved and random-trained
    encodings.png       Δ for each passage encoding, and GPU-hours per run
    retrieval_only.png  Δ for retrieval only (#R) by passage encoding and query
    stage0.png          patient only's test loss against learning rate (the seed configuration)
    compute.png         test loss against training compute, every grid cell, per W and arm
    isoflop.png         IsoFLOP curves and the compute-optimal frontier from the fitted laws
    delta.png           Δ between the grid's models against D: xres vs #1, #4 vs #1, xres vs #4

Needs results_grid.csv and results_compute.csv (grid.py, count_compute.py).
"""

import subprocess
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.lines import Line2D

from grid import EPOCHS, LOSS, centre, fit_quiet, t_interval, table
from ladder import RUN

HERE = Path(__file__).parent
OUT = HERE / "figures"
BLUE, ORANGE, RED, GREY, GREEN = "#0072B2", "#E69F00", "#D55E00", "#555555", "#009E73"
W_COLOUR = {"64x2x4": "#56B4E9", "128x2x4": "#0072B2", "256x4x8": "#08306B"}
CELLS = [("64x2x4", 10000), ("64x2x4", 153220), ("256x4x8", 10000), ("256x4x8", 153220)]
D_LABEL = {3000: "3K", 10000: "10K", 30000: "30K", 100000: "100K", 153220: "153K"}


def load(outputs: Path) -> pl.DataFrame:
    """Every test loss: run name (config + suffix), K, W, D, lr, dropout, draw, test."""
    rows = []
    for f in outputs.glob("*/draw*/test_*/loggers/csv/*/metrics.csv"):
        run, draw, test = f.parts[-7], f.parts[-6], f.parts[-5][5:]
        m = RUN.fullmatch(run)
        if not m:
            continue
        loss = pl.read_csv(f).select("test/loss").drop_nulls()
        if loss.height:
            rows.append({"name": m["config"] + "_K" + (m["K"] or "0") + (m["suffix"] or ""), "W": m["W"],
                         "D": int(m["D"]), "lr": m["lr"], "do": m["dropout"], "draw": draw, "test": test,
                         "loss": loss.item(-1, 0)})
    return pl.DataFrame(rows)


def delta(df: pl.DataFrame, name: str, w: str, d: int, test: str = "topK") -> tuple[float, float, int]:
    """Paired Δ% of run `name` against its baseline at (w, d); (nan, nan, 0) if not run."""
    base = "patient_only_K0_warm" if "_warm" in name else "patient_only_K0"
    sel = (pl.col("W") == w) & (pl.col("D") == d) & (pl.col("lr") == "1e-3") & (pl.col("do") == "0")
    a = df.filter(sel & (pl.col("name") == name) & (pl.col("test") == test)).select("draw", "loss")
    b = df.filter(sel & (pl.col("name") == base) & (pl.col("test") == "last")).select("draw", base="loss")
    p = a.join(b, on="draw")
    if p.is_empty():
        return float("nan"), float("nan"), 0
    pct = ((p["base"] - p["loss"]) / p["base"] * 100).to_list()
    return float(np.mean(pct)), t_interval(pct), len(pct)


def zero_line(ax, vertical=False):
    (ax.axvline if vertical else ax.axhline)(0, color="#999", lw=1, zorder=0)


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / name, dpi=140, bbox_inches="tight")
    plt.close(fig)


def dot_panels(df, rows, title, name, note=None, missing="not run"):
    """One panel per cell; one row per (label, run, random-trained run or None)."""
    fig, axes = plt.subplots(1, 4, figsize=(15, 0.7 * len(rows) + 1.8), sharey=True)
    y = np.arange(len(rows))[::-1]
    for ax, (w, d) in zip(axes, CELLS):
        for yi, (label, run, rand) in zip(y, rows):
            for r, colour, off, fill in ((run, BLUE, 0.15, True), (rand, ORANGE, -0.15, False)):
                if r is None:
                    continue
                m, ci, n = delta(df, r, w, d)
                if n:
                    ci = 0 if np.isnan(ci) else ci
                    ax.errorbar(m, yi + off, xerr=ci, fmt="o", ms=7, capsize=3, lw=1.5,
                                color=colour, mfc=colour if fill else "white", mew=1.5)
                    ax.annotate(f"{m:+.2f}", (m + ci, yi + off), textcoords="offset points", xytext=(5, -3),
                                fontsize=7, color=colour)
                elif r == run:
                    ax.text(0.5, yi, missing, transform=ax.get_yaxis_transform(), ha="center", va="center",
                            fontsize=8, color="#999", style="italic")
        ax.set_ylim(y.min() - 0.6, y.max() + 0.6)
        ax.margins(x=0.25)
        zero_line(ax, vertical=True)
        ax.set_title(f"W {w}, D {D_LABEL[d]}")
        ax.set_xlabel("Δ, % of patient-only loss\n(right of 0: better than patient only)")
        ax.grid(axis="x", alpha=0.3)
    axes[0].set_yticks(y, [r[0] for r in rows])
    handles = [Line2D([], [], color=BLUE, marker="o", ls="", label="retrieved passages")]
    if any(r[2] for r in rows):
        handles.append(Line2D([], [], color=ORANGE, marker="o", mfc="white", ls="",
                              label="same model trained on random passages (#4)"))
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(title, x=0.01, ha="left", fontsize=12)
    if note:
        fig.text(0.01, -0.1, note, fontsize=8, color="#555")
    fig.tight_layout()
    save(fig, name)


def readers(df):
    rows = [
        ("v1, warm, word pieces", "v1_masked_K4_warm", "v1_random_docs_masked_K4_warm"),
        ("v1, warm, last layer", "v1_masked_K4_warm_pre", "v1_random_docs_masked_K4_warm_pre"),
        ("v1, warm, all layers", "v1_masked_K4_warm_attend", "v1_random_docs_masked_K4_warm_attend"),
        ("residual, warm, last layer", "residual_K4_warm_pre", "residual_random_docs_K4_warm_pre"),
        ("xres, warm, last layer", "xres_K4_warm_pre", "xres_random_docs_K4_warm_pre"),
    ]
    dot_panels(df, rows, "Readers: Δ vs patient only (K 4, encoder warm-started)", "readers.png",
               "Not shown: v1 trained from scratch (cold), −4.5%, −13.4%, −5.6%, −18.1% in these four cells.")


def k_sweep(df):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, w in zip(axes, ("64x2x4", "256x4x8")):
        for reader, pre, rand, ls in (("v1", "v1_masked", "v1_random_docs_masked", "--"),
                                      ("xres", "xres", "xres_random_docs", "-")):
            for name, colour, fill, label in ((pre, BLUE, True, "retrieved"), (rand, ORANGE, False, "random-trained")):
                pts = [(k, *delta(df, f"{name}_K{k}_warm_pre", w, 153220)) for k in (4, 32, 128, 512)]
                pts = [p for p in pts if p[3]]
                if pts:
                    k, m, ci, _ = zip(*pts)
                    ax.errorbar(k, m, yerr=np.nan_to_num(ci), color=colour, ls=ls, marker="o", ms=6, capsize=3,
                                mfc=colour if fill else "white", mew=1.5, label=f"{reader}, {label}")
        if not delta(df, "xres_K512_warm_pre", w, 153220)[2]:
            ax.text(512, 0, "K 512\nrunning", ha="center", va="bottom", fontsize=8, color="#999", style="italic")
        zero_line(ax)
        ax.set_xscale("log", base=2)
        ax.set_xticks([4, 32, 128, 512], ["4", "32", "128", "512"])
        ax.set_xlabel("passages per patient, K")
        ax.set_title(f"W {w}, D 153K")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Δ, % of patient-only loss (positive: better)")
    axes[1].legend(fontsize=8)
    fig.suptitle("More passages: Δ vs patient only", x=0.01, ha="left")
    fig.tight_layout()
    save(fig, "k.png")


def gpu_hours(name_glob: str) -> float:
    """Median elapsed hours of completed jobs whose name contains name_glob (sacct, last 90 days)."""
    try:
        out = subprocess.run(["sacct", "-X", "-n", "-S", "now-90days", "-o", "JobName%120,State,ElapsedRaw"],
                             capture_output=True, text=True, timeout=60).stdout
    except Exception:
        return float("nan")
    secs = [int(p[2]) for p in (l.split() for l in out.splitlines())
            if len(p) == 3 and p[1] == "COMPLETED" and p[0].endswith(name_glob)]
    return float(np.median(secs)) / 3600 if secs else float("nan")


def encodings(df):
    rows = [("v1, word pieces", "v1_masked_K4_warm"), ("v1, last layer (precomputed)", "v1_masked_K4_warm_pre"),
            ("v1, all layers (on the fly)", "v1_masked_K4_warm_attend"),
            ("xres, last layer (precomputed)", "xres_K4_warm_pre"), ("xres, all layers (on the fly)", "xres_K4_warm_attend")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6), sharey=True, gridspec_kw={"width_ratios": [1, 1, 0.8]})
    y = np.arange(len(rows))[::-1]
    for ax, w in zip(axes[:2], ("64x2x4", "256x4x8")):
        for yi, (label, run) in zip(y, rows):
            m, ci, n = delta(df, run, w, 153220)
            if n:
                ax.errorbar(m, yi, xerr=np.nan_to_num(ci), fmt="o", ms=7, capsize=3, color=BLUE if "xres" in run else GREY)
                ax.annotate(f"{m:+.2f}", (m, yi), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7)
        zero_line(ax, vertical=True)
        ax.set_title(f"Δ vs patient only, W {w}, D 153K")
        ax.set_xlabel("Δ, % (right of 0: better)")
        ax.grid(axis="x", alpha=0.3)
    hours = [gpu_hours(f"{run.replace('_K4', '_W256x4x8_D153220_lr1e-3_do0_K4')}") for _, run in rows]
    axes[2].barh(y, hours, color=[BLUE if "xres" in r else GREY for _, r in rows], height=0.5)
    for yi, h in zip(y, hours):
        if not np.isnan(h):
            axes[2].annotate(f"{h:.1f} h", (h, yi), textcoords="offset points", xytext=(4, -3), fontsize=8)
    axes[2].set_xscale("log")
    axes[2].set_title("GPU-hours per run (256x4x8, D 153K)")
    axes[2].grid(axis="x", alpha=0.3)
    axes[0].set_yticks(y, [r[0] for r in rows])
    fig.suptitle("Passage encodings", x=0.01, ha="left")
    fig.tight_layout()
    save(fig, "encodings.png")


def retrieval_only(df):
    rows = [("word pieces, retrieval query (cold)", "retrieval_only_K4", None),
            ("last layer, retrieval query", "retrieval_only_K4_warm_pre", None),
            ("all layers, retrieval query", "retrieval_only_K4_warm_attend", None),
            ("last layer, learned query", "retrieval_only_learned_K4_warm_pre", None)]
    dot_panels(df, rows, "Retrieval only (#R): Δ vs patient only", "retrieval_only.png", missing="running")


def stage0(df):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    for ax, w in zip(axes, ("64x2x4", "128x2x4", "256x4x8")):
        for do, ls in (("0", "-"), ("0.1", "--")):
            runs = df.filter((pl.col("name") == "patient_only_K0") & (pl.col("W") == w) & (pl.col("D") == 153220)
                             & (pl.col("test") == "last") & pl.col("draw").is_in(["draw1", "draw2", "draw3"])
                             & pl.col("lr").is_in(["3e-4", "1e-3", "3e-3"]))
            # Each draw has its own tasks: centre on the draw's mean over this W's settings.
            runs = runs.with_columns(loss=pl.col("loss") - pl.col("loss").mean().over("draw") + pl.col("loss").mean())
            g = (runs.filter(pl.col("do") == do).group_by("lr").agg(m=pl.col("loss").mean(), s=pl.col("loss").std(), n=pl.len())
                 .with_columns(x=pl.col("lr").cast(pl.Float64)).sort("x"))
            ax.errorbar(g["x"], g["m"], yerr=g["s"] / g["n"].sqrt(), color=W_COLOUR[w], ls=ls, marker="o", capsize=3,
                        label=f"dropout {do}")
        ax.set_xscale("log")
        ax.set_xticks([3e-4, 1e-3, 3e-3], ["3e-4", "1e-3", "3e-3"])
        ax.minorticks_off()
        ax.set_title(f"W {w}")
        ax.set_xlabel("learning rate")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("test loss, D 153K\n(draw-centred, mean of 3 draws ± SE)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Patient-only sweep (the configuration retrieval starts from)", x=0.01, ha="left")
    fig.tight_layout()
    save(fig, "stage0.png")


ARM_STYLE = {"patient only": (GREY, "patient only (#1, warm)"), "xres": (BLUE, "xres, retrieved passages"),
             "xres random": (ORANGE, "xres trained on random passages (#4)")}


def compute(grid: pl.DataFrame):
    c = centre(grid).group_by("arm", "W", "D").agg(C=pl.col("C").first(), loss=pl.col("loss").mean())
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    markers = {"64x2x4": "o", "128x2x4": "s", "256x4x8": "^"}
    for arm, (colour, label) in ARM_STYLE.items():
        for w, mk in markers.items():
            g = c.filter((pl.col("arm") == arm) & (pl.col("W") == w)).sort("D")
            ax.plot(g["C"], g["loss"], color=colour, marker=mk, ms=6, lw=1.2, alpha=0.9)
        pts = sorted(c.filter(pl.col("arm") == arm).select("C", "loss").iter_rows())
        env, best = [], float("inf")
        for x, y in pts:
            if y < best:
                best, env = y, env + [(x, y)]
        ax.plot(*zip(*env), color=colour, lw=3, alpha=0.35)
    for w, mk in markers.items():
        g = c.filter((pl.col("arm") == "patient only") & (pl.col("W") == w)).sort("D")
        for x, y, d in g.select("C", "loss", "D").iter_rows():
            if w == "64x2x4":
                ax.annotate(f"D {D_LABEL[d]}", (x, y), textcoords="offset points", xytext=(-4, -12), fontsize=7, color="#555")
    ax.set_xscale("log")
    ax.set_xlabel("training compute, FLOPs (including the patient-only checkpoint a warm run starts from)")
    ax.set_ylabel("test loss (draw-centred, mean of draws)")
    handles = [Line2D([], [], color=col, lw=2, label=lab) for col, lab in ARM_STYLE.values()]
    handles += [Line2D([], [], color="#333", marker=mk, ls="", label=f"W {w}") for w, mk in markers.items()]
    handles += [Line2D([], [], color="#999", lw=3, alpha=0.5, label="lower envelope (best at that compute)")]
    ax.legend(handles=handles, fontsize=8)
    ax.set_title("Test loss against training compute (lines join D 3K → 153K for one W)", loc="left", fontsize=11)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save(fig, "compute.png")


def epochs_at(d):
    ds = sorted(EPOCHS)
    return np.interp(np.log(d), np.log(ds), [EPOCHS[x] for x in ds])


def isoflop(grid: pl.DataFrame):
    """From each arm's separable law L(N, D) and its measured compute per patient C(N) (log-linear in N
    over the three sizes): for each budget, loss against N with D = budget / (C(N) · epochs)."""
    c = centre(grid)
    budgets = np.logspace(13.5, 16.5, 7)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), gridspec_kw={"width_ratios": [1, 1, 1.1]})
    frontier = {}
    cmap = plt.get_cmap("viridis")
    for ax, arm in zip(axes[:2], ("patient only", "xres")):
        g = c.filter(pl.col("arm") == arm)
        model = fit_quiet(table(g), "separable-power", 200)
        per = g.group_by("W").agg(N=pl.col("N").first(), f=(pl.col("flops") + pl.col("cold_flops")).first())
        slope, icpt = np.polyfit(np.log(per["N"].to_numpy()), np.log(per["f"].to_numpy()), 1)
        ns = np.logspace(4, 7.5, 120)
        best = []
        for i, b in enumerate(budgets):
            f = np.exp(icpt) * ns**slope
            d = b / (f * 3.0)
            for _ in range(5):  # epochs depend on D
                d = b / (f * epochs_at(d))
            ok = (d >= 1000) & (d <= 1e7)
            pts = pl.DataFrame({"model_size__n": ns[ok], "dataset_size__d": d[ok]})
            if pts.is_empty():
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pred = model.predict(pts, targets=[LOSS])[f"{LOSS}__median"].to_numpy()
            ax.plot(ns[ok], pred, color=cmap(i / (len(budgets) - 1)), lw=2, label=f"{b:.0e} FLOPs")
            j = int(np.argmin(pred))
            ax.plot(ns[ok][j], pred[j], "o", color=cmap(i / (len(budgets) - 1)), mec="black", ms=7)
            best.append((b, pred[j]))
        frontier[arm] = best
        ax.axvspan(per["N"].min(), per["N"].max(), color="#eee", zorder=0)
        ax.set_xscale("log")
        ax.set_xlabel("model size N (trained parameters, without the code table)")
        ax.set_ylabel("predicted test loss")
        ax.set_title(f"IsoFLOP curves, {arm}", loc="left")
        ax.grid(alpha=0.3)
        ax.set_ylim(0.0175, 0.028)
    axes[0].legend(fontsize=7, title="training budget")
    for arm, colour in (("patient only", GREY), ("xres", BLUE)):
        b, l = zip(*frontier[arm])
        axes[2].plot(b, l, marker="o", color=colour, lw=2, label=arm)
    axes[2].set_xscale("log")
    axes[2].set_xlabel("training budget, FLOPs")
    axes[2].set_ylabel("best predicted test loss at that budget")
    axes[2].set_title("Compute-optimal frontier (minima of the isoFLOP curves)", loc="left")
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    fig.text(0.01, -0.04, "From the separable-power law per arm (the only form passing the quality gate), fitted on 3 model "
             "sizes; outside the shaded range of trained sizes it is extrapolation.", fontsize=8, color="#555")
    fig.tight_layout()
    save(fig, "isoflop.png")


def delta_grid(grid: pl.DataFrame):
    pairs = [("xres", "patient only", "xres vs patient only (#2 − #1)"),
             ("xres random", "patient only", "trained on random passages vs patient only (#4 − #1)"),
             ("xres", "xres random", "retrieved vs random-trained (#2 − #4): what the passages say")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, (a, b, title) in zip(axes, pairs):
        for w, colour in W_COLOUR.items():
            pts = []
            for d in sorted(grid["D"].unique()):
                x = grid.filter((pl.col("arm") == a) & (pl.col("W") == w) & (pl.col("D") == d)).select("draw", "loss")
                y = grid.filter((pl.col("arm") == b) & (pl.col("W") == w) & (pl.col("D") == d)).select("draw", base="loss")
                p = x.join(y, on="draw")
                if p.height:
                    pct = ((p["base"] - p["loss"]) / p["base"] * 100).to_list()
                    pts.append((d, np.mean(pct), t_interval(pct)))
            d, m, ci = zip(*pts)
            ax.errorbar(d, m, yerr=np.nan_to_num(ci), color=colour, marker="o", capsize=3, lw=2, label=f"W {w}")
        zero_line(ax)
        ax.set_xscale("log")
        ax.set_xticks(sorted(D_LABEL), [D_LABEL[d] for d in sorted(D_LABEL)])
        ax.set_xlabel("training patients D")
        ax.set_title(title, fontsize=10, loc="left")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("paired difference, % of loss (positive: first model better)")
    axes[0].legend()
    fig.tight_layout()
    save(fig, "delta.png")


if __name__ == "__main__":
    outputs = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "outputs"
    grid = pl.read_csv(HERE / "results_grid.csv")
    compute(grid)
    isoflop(grid)
    delta_grid(grid)
    if outputs.exists():
        df = load(outputs)
        readers(df)
        k_sweep(df)
        encodings(df)
        retrieval_only(df)
        stage0(df)
