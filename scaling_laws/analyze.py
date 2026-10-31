"""Collect the scaling-law runs, fit L(N, D) and draw plots P1-P7 of design.md.

    .venv/bin/python analyze.py [OUTPUT_DIR] [RESULTS_DIR]    # defaults: outputs, results

Writes to RESULTS_DIR (one folder per task count T):
    runs.csv           one row per test evaluation (config, W, D, T, draw, eval, loss, auroc)
    params.csv         N (encoder without its embedding table) and N_compute per config, W, T
    fits.csv           E, A, alpha, B, beta of L(N, D) = E + A / N^alpha + B / D^beta per config, T, with
                       R^2 over runs and over cell means, and RMSE over runs
    T<T>/P1.png ...    the plots

Choices for the open points in design.md:
    N          patient-encoder parameters without the code embedding table (a lookup).
    N_compute  all trainable parameters except embedding tables: encoder, query projector, fusion
               and head. Frozen parts (Qwen3 token table, Qwen3 query model) are not counted.
    C          6 * N_compute * D * epochs * S, with S = 256 (max_seq_len) and epochs from run.sh.
    P5, P6     read off the fitted law: our grid is not laid out on equal-compute lines.
"""

import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from hydra.utils import instantiate  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402

EPOCHS = {1000: 10, 3000: 7, 10000: 5, 30000: 4, 100000: 3, 153220: 3}
S = 256
CONFIGS = {
    "patient_only": "#1 Patient only",
    "retrieval_learned_query": "#2 Retrieval, learned query",
    "retrieval_qwen3_query": "#3 Retrieval, Qwen3 query",
    "retrieval_random_docs": "#4 Random documents",
}
RUN_DIR = re.compile(r"^(?P<config>.+)_W(?P<W>[^_]+)_D(?P<D>\d+)_T(?P<T>\d+)$")
EVAL_DIR = re.compile(r"^draw(?P<draw>\d+)_(?P<eval>test(?:_random_doc)?)$")


def collect(outputs: Path) -> pl.DataFrame:
    """One row per finished test evaluation."""
    rows = []
    for csv in sorted(outputs.glob("*/draw*_test*/loggers/csv/*/metrics.csv")):
        run, ev = RUN_DIR.match(csv.parents[3].parent.name), EVAL_DIR.match(csv.parents[3].name)
        if not (run and ev):
            continue
        df = pl.read_csv(csv, infer_schema_length=None).filter(pl.col("test/loss").is_not_null())
        if df.is_empty():
            continue
        rows.append(
            {
                "config": run["config"],
                "W": run["W"],
                "D": int(run["D"]),
                "T": int(run["T"]),
                "draw": int(ev["draw"]),
                "eval": ev["eval"],
                "loss": float(df["test/loss"][-1]),
                "auroc": float(df["test/auroc/mean"][-1]) if "test/auroc/mean" in df.columns else None,
            }
        )
    return pl.DataFrame(rows)


def count_params(train_dir: Path) -> tuple[int, int]:
    """(N, N_compute) for the model in one training run's resolved config."""
    import torch

    cfg = OmegaConf.load(train_dir / "resolved_config.yaml")
    parts = ["encoder", "head"]
    if cfg.fusion._target_.split(".")[-1] != "PassthroughFusion":
        parts += ["query_projector", "fusion"]

    def split(module: torch.nn.Module) -> tuple[int, int]:
        emb = sum(p.numel() for m in module.modules() if isinstance(m, torch.nn.Embedding) for p in m.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        return trainable - emb, emb

    n = n_compute = 0
    for part in parts:
        if part == "query_projector" and "Qwen3" in cfg.query_projector._target_:
            continue  # frozen Qwen3 text model: no trainable parameters
        non_emb, _ = split(instantiate(cfg[part]))
        n_compute += non_emb
        if part == "encoder":
            n = non_emb
    return n, n_compute


def params_table(outputs: Path, runs: pl.DataFrame) -> pl.DataFrame:
    rows = []
    for config, w, t in runs.select("config", "W", "T").unique().iter_rows():
        train_dir = next(outputs.glob(f"{config}_W{w}_D*_T{t}/draw*/resolved_config.yaml")).parent
        n, n_compute = count_params(train_dir)
        rows.append({"config": config, "W": w, "T": t, "N": n, "N_compute": n_compute})
    return pl.DataFrame(rows)


def law(p: np.ndarray, n: np.ndarray, d: np.ndarray) -> np.ndarray:
    e, a, alpha, b, beta = p
    return e + a / n**alpha + b / d**beta


def fit_law(n: np.ndarray, d: np.ndarray, loss: np.ndarray) -> np.ndarray | None:
    """Least squares on loss over a few starting points; parameters (E, A, alpha, B, beta)."""
    if len(np.unique(n)) < 3 or len(np.unique(d)) < 3:
        return None
    best = None
    for alpha0 in (0.2, 0.5, 1.0):
        for beta0 in (0.2, 0.5, 1.0):
            x0 = [np.log(loss.min() * 0.8), 0.0, alpha0, 0.0, beta0]

            def resid(x):
                p = [np.exp(x[0]), np.exp(x[1]), x[2], np.exp(x[3]), x[4]]
                return law(p, n, d) - loss

            r = least_squares(resid, x0, bounds=([-20, -30, 0, -30, 0], [5, 30, 3, 30, 3]))
            if best is None or r.cost < best.cost:
                best = r
    x = best.x
    return np.array([np.exp(x[0]), np.exp(x[1]), x[2], np.exp(x[3]), x[4]])


def r_squared(y: np.ndarray, fitted: np.ndarray) -> float:
    """1 - SS_residual / SS_total."""
    return float(1 - ((y - fitted) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def epochs_at(d: np.ndarray) -> np.ndarray:
    """Epochs schedule of run.sh, log-linear between its D values."""
    ds = np.array(sorted(EPOCHS))
    return np.interp(np.log(d), np.log(ds), [EPOCHS[x] for x in ds])


def d_for_budget(c: float, n_compute: float) -> float:
    """Training patients D that spend compute c with a model of n_compute (bisection in log D)."""
    lo, hi = np.log(10.0), np.log(1e9)
    for _ in range(80):
        mid = (lo + hi) / 2
        d = np.exp(mid)
        lo, hi = (mid, hi) if 6 * n_compute * d * epochs_at(d) * S < c else (lo, mid)
    return float(np.exp(lo))


def colors(k: int) -> list:
    return [plt.cm.viridis(x) for x in np.linspace(0.05, 0.9, max(k, 1))]


def width_key(w: str) -> tuple[int, int]:
    parts = [int(x) for x in w.split("x")]
    return parts[0], len(parts)


def plot_lines(ax, cell, x, group, fits=None, params=None):
    """Mean test loss (+/- s.e. over draws) vs x, one line per group value."""
    groups = sorted(cell[group].unique(), key=width_key if group == "W" else None)
    for col, g in zip(colors(len(groups)), groups):
        sub = cell.filter(pl.col(group) == g).sort(x)
        ax.errorbar(sub[x], sub["loss"], yerr=sub["sem"], color=col, marker="o", ms=3, lw=1.2,
                    capsize=2, label=f"{group}={g}")
        if fits is not None:
            xs = np.geomspace(sub[x].min(), sub[x].max(), 50)
            other = sub["N" if x == "D" else "D"][0]
            n, d = (np.full_like(xs, other), xs) if x == "D" else (xs, np.full_like(xs, other))
            ax.plot(xs, law(fits, n, d), color=col, ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel({"D": "training patients D", "N": "encoder parameters N"}[x])
    ax.grid(alpha=0.3)


def plot_random_doc(runs: pl.DataFrame, t: int, out: Path) -> None:
    """P7: random document - top document test loss, matched on (W, D, draw), for #2 and #3."""
    keys = ["config", "W", "N", "D", "draw"]
    sub = runs.filter((pl.col("T") == t) & pl.col("config").is_in(["retrieval_learned_query", "retrieval_qwen3_query"]))
    rd = sub.filter(pl.col("eval") == "test").select(*keys, "loss").join(
        sub.filter(pl.col("eval") == "test_random_doc").select(*keys, random="loss"), on=keys
    )
    configs = [c for c in CONFIGS if c in rd["config"].unique().to_list()]
    if not configs:
        return
    fig, axes = plt.subplots(len(configs), 2, figsize=(9, 3.4 * len(configs)), squeeze=False)
    for row, config in zip(axes, configs):
        d = rd.filter(pl.col("config") == config).with_columns(diff=pl.col("random") - pl.col("loss"))
        agg = d.group_by("W", "N", "D").agg(
            loss=pl.col("diff").mean(),
            sem=(pl.col("diff").std() / pl.col("diff").count().sqrt()).fill_null(0.0),
        )
        plot_lines(row[0], agg, "D", "W")
        plot_lines(row[1], agg, "N", "D")
        for ax in row:
            ax.axhline(0, color="k", lw=0.8)
            ax.set_ylabel(f"{CONFIGS[config].split(',')[0]}: random − top doc loss")
            ax.legend(fontsize=7)
    fig.suptitle(f"Random document − top document test loss (above 0: the top document helps), T = {t}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "P7.png", dpi=150)
    plt.close(fig)


def main(outputs: Path, results: Path) -> None:
    results.mkdir(parents=True, exist_ok=True)
    runs = collect(outputs)
    if runs.is_empty():
        print(f"no finished runs under {outputs}")
        return
    params = params_table(outputs, runs)
    runs = runs.join(params, on=["config", "W", "T"])
    runs = runs.with_columns(
        C=6 * pl.col("N_compute") * pl.col("D") * pl.col("D").replace_strict(EPOCHS) * S
    )
    runs.sort("config", "T", "W", "D", "draw", "eval").write_csv(results / "runs.csv")
    params.sort("config", "T", "N").write_csv(results / "params.csv")

    test = runs.filter(pl.col("eval") == "test")
    cells = test.group_by("config", "W", "N", "N_compute", "D", "T").agg(
        loss=pl.col("loss").mean(),
        sem=(pl.col("loss").std() / pl.col("loss").count().sqrt()).fill_null(0.0),
        auroc=pl.col("auroc").mean(),
        draws=pl.len(),
        C=pl.col("C").first(),
    )

    fits = {}
    fit_rows = []
    for (config, t), sub in test.group_by("config", "T"):
        n, d, y = sub["N"].to_numpy().astype(float), sub["D"].to_numpy().astype(float), sub["loss"].to_numpy()
        p = fit_law(n, d, y)
        if p is not None:
            fits[config, t] = p
            means = sub.group_by("N", "D").agg(pl.col("loss").mean())
            fit_rows.append({"config": config, "T": t, "runs": sub.height,
                             **dict(zip(["E", "A", "alpha", "B", "beta"], p.tolist())),
                             "r2_runs": r_squared(y, law(p, n, d)),
                             "r2_cells": r_squared(means["loss"].to_numpy(),
                                                   law(p, means["N"].to_numpy().astype(float),
                                                       means["D"].to_numpy().astype(float))),
                             "rmse": float(np.sqrt(((y - law(p, n, d)) ** 2).mean()))})
    if fit_rows:
        pl.DataFrame(fit_rows).sort("config", "T").write_csv(results / "fits.csv")

    for t in sorted(cells["T"].unique()):
        out = results / f"T{t}"
        out.mkdir(exist_ok=True)
        ct = cells.filter(pl.col("T") == t)
        configs = [c for c in CONFIGS if c in ct["config"].unique().to_list()]

        # P1, P2 (data), P3 (same with the fitted law, dashed)
        for name, x, group, with_fit in (("P1", "D", "W", False), ("P2", "N", "D", False),
                                         ("P3_D", "D", "W", True), ("P3_N", "N", "D", True)):
            fig, axes = plt.subplots(1, len(configs), figsize=(4.2 * len(configs), 3.6), sharey=True,
                                     squeeze=False)
            for ax, config in zip(axes[0], configs):
                plot_lines(ax, ct.filter(pl.col("config") == config), x, group,
                           fits.get((config, t)) if with_fit else None)
                ax.set_title(CONFIGS[config], fontsize=10)
            axes[0][0].set_ylabel("test loss")
            axes[0][0].legend(fontsize=7, loc="upper right")
            fig.suptitle(f"T = {t} tasks", fontsize=10)
            fig.tight_layout()
            fig.savefig(out / f"{name}.png", dpi=150)
            plt.close(fig)

        # P4: retrieval - patient only, matched on (W, D, draw); control split on its slice
        tt = test.filter(pl.col("T") == t)
        base = tt.filter(pl.col("config") == "patient_only").select("W", "D", "draw", base="loss")
        diffs = []
        for config in ("retrieval_learned_query", "retrieval_qwen3_query"):
            d = tt.filter(pl.col("config") == config).join(base, on=["W", "D", "draw"])
            if not d.is_empty():
                diffs.append(d.with_columns(diff=pl.col("loss") - pl.col("base")))
        if diffs:
            fig, axes = plt.subplots(len(diffs), 2, figsize=(9, 3.4 * len(diffs)), squeeze=False)
            for row, d in zip(axes, diffs):
                agg = d.group_by("W", "N", "D").agg(
                    loss=pl.col("diff").mean(),
                    sem=(pl.col("diff").std() / pl.col("diff").count().sqrt()).fill_null(0.0),
                )
                plot_lines(row[0], agg, "D", "W")
                plot_lines(row[1], agg, "N", "D")
                for ax in row:
                    ax.axhline(0, color="k", lw=0.8)
                    ax.set_ylabel(f"{CONFIGS[d['config'][0]].split(',')[0]} − #1 loss")
                row[1].legend(fontsize=7)
            fig.suptitle(f"Retrieval − Patient only test loss (below 0: retrieval better), T = {t}", fontsize=10)
            fig.tight_layout()
            fig.savefig(out / "P4.png", dpi=150)
            plt.close(fig)

        ctrl = tt.filter(pl.col("config") == "retrieval_random_docs")
        if not ctrl.is_empty():
            keys = ["W", "D", "draw"]
            m = (
                ctrl.select(*keys, r4="loss")
                .join(tt.filter(pl.col("config") == "retrieval_learned_query").select(*keys, r2="loss"), on=keys)
                .join(base.rename({"base": "r1"}), on=keys)
                .with_columns(knowledge=pl.col("r2") - pl.col("r4"), architecture=pl.col("r4") - pl.col("r1"))
            )
            if not m.is_empty():
                fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
                for ax, what, label in zip(axes, ("knowledge", "architecture"),
                                           ("#2 − #4 (knowledge)", "#4 − #1 (architecture)")):
                    agg = m.group_by("W", "D").agg(
                        loss=pl.col(what).mean(),
                        sem=(pl.col(what).std() / pl.col(what).count().sqrt()).fill_null(0.0),
                    )
                    plot_lines(ax, agg, "D", "W")
                    ax.axhline(0, color="k", lw=0.8)
                    ax.set_title(label, fontsize=10)
                axes[0].set_ylabel("test loss difference")
                axes[1].legend(fontsize=7)
                fig.suptitle(f"Control slice, T = {t} (below 0: the first config is better)", fontsize=10)
                fig.tight_layout()
                fig.savefig(out / "P4_control.png", dpi=150)
                plt.close(fig)

        plot_random_doc(runs, t, out)

        # P5: IsoFLOP curves from the fitted law, over the trained encoder sizes
        fitted = [c for c in configs if (c, t) in fits]
        if fitted:
            fig5, axes5 = plt.subplots(1, len(fitted), figsize=(4.2 * len(fitted), 3.6), squeeze=False)
            fig6, ax6 = plt.subplots(figsize=(5.5, 4))
            for ax, col6, config in zip(axes5[0], colors(len(fitted)), fitted):
                p = fits[config, t]
                sizes = params.filter((pl.col("config") == config) & (pl.col("T") == t)).sort("N")
                cs = ct.filter(pl.col("config") == config)["C"]
                budgets = np.geomspace(cs.min(), cs.max(), 6)
                frontier_c = np.geomspace(cs.min(), cs.max(), 40)
                for col, c in zip(colors(len(budgets)), budgets):
                    # Only sizes whose D for this budget lies in the trained range (no extrapolation).
                    pts = [(n, law(p, n, d)) for n, nc in sizes.select("N", "N_compute").iter_rows()
                           if min(EPOCHS) <= (d := d_for_budget(c, nc)) <= max(EPOCHS)]
                    if len(pts) < 2:
                        continue
                    ns, ls = zip(*pts)
                    ax.plot(ns, ls, color=col, marker="o", ms=3, label=f"C={c:.1e}")
                    i = int(np.argmin(ls))
                    ax.plot(ns[i], ls[i], "k*", ms=8)
                ax.set_xscale("log")
                ax.set_xlabel("encoder parameters N")
                ax.set_title(CONFIGS[config], fontsize=10)
                ax.grid(alpha=0.3)
                best = [min(law(p, n, d_for_budget(c, nc)) for n, nc in sizes.select("N", "N_compute").iter_rows())
                        for c in frontier_c]
                ax6.plot(frontier_c, best, color=col6, ls="--", lw=1)
                # Measured frontier: lowest cell-mean loss reached at or below each compute.
                runs_c = ct.filter(pl.col("config") == config).sort("C")
                ax6.scatter(runs_c["C"], runs_c["loss"], color=col6, s=6, alpha=0.4)
                ax6.step(runs_c["C"], np.minimum.accumulate(runs_c["loss"].to_numpy()), where="post",
                         color=col6, label=CONFIGS[config])
            axes5[0][0].set_ylabel("test loss (fitted law, trained D range only)")
            axes5[0][0].legend(fontsize=7)
            fig5.suptitle(f"IsoFLOP curves, T = {t} (★ best size per budget)", fontsize=10)
            fig5.tight_layout()
            fig5.savefig(out / "P5.png", dpi=150)
            ax6.set_xscale("log")
            ax6.set_xlabel("training compute C (FLOPs)")
            ax6.set_ylabel("test loss")
            ax6.set_title(f"Best loss vs compute, T = {t}\n(solid: best run so far; dashed: fitted law; dots: cells)",
                          fontsize=9)
            ax6.grid(alpha=0.3)
            ax6.legend(fontsize=7)
            fig6.tight_layout()
            fig6.savefig(out / "P6.png", dpi=150)
            plt.close(fig5)
            plt.close(fig6)

    cells.sort("config", "T", "W", "D").write_csv(results / "cells.csv")
    print(f"{runs.height} evaluations, {cells.height} cells, {len(fits)} fits -> {results}")


if __name__ == "__main__":
    args = sys.argv[1:]
    main(Path(args[0] if args else "outputs"), Path(args[1] if len(args) > 1 else "results"))
