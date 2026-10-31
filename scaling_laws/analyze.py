"""Collect the scaling-law runs, fit them with simple-scaling-laws and draw plots P1-P7 of design.md.

    .venv/bin/python analyze.py OUTPUT_DIR RESULTS_DIR     # collect run outputs, then fit and plot
    .venv/bin/python analyze.py RESULTS_DIR/runs.csv RESULTS_DIR   # fit and plot from collected runs

Fitting follows https://github.com/mmcdermott/simple-scaling-laws: the separable-power law
E + A / N^alpha + B / D^beta per configuration and T, on test loss and AUROC, with wild cluster
bootstrap intervals; paired comparisons between configurations; and compute-optimal allocation
from a fitted cost model.

Writes to RESULTS_DIR:
    runs.csv           one row per test evaluation (config, W, D, T, draw, eval, loss, auroc, N, C)
    params.csv         N (encoder without its embedding table) and N_compute per config, W, T
    cells.csv          mean over draws per config, W, D, T
    fits.csv           parameters with 95% intervals, R^2 and RMSE per config, T and target
    warnings.csv       every warning the package raised
    comparisons.csv    paired differences between configurations at every trained (N, D)
    allocation.csv     compute-optimal N, D and loss per budget
    fits/*.slaw        the fitted models (simple_scaling_laws.load)
    T<T>/P1.png ...    the plots

    N          patient-encoder parameters without the code embedding table (a lookup).
    N_compute  all trainable parameters except embedding tables: encoder, query projector, fusion
               and head. Frozen parts (Qwen3 token table, Qwen3 query model) are not counted.
    C          6 * N_compute * D * epochs * S, with S = 256 (max_seq_len) and epochs from run.sh.
"""

import json
import re
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from simple_scaling_laws import compare, fit, fit_cost_model  # noqa: E402

EPOCHS = {1000: 10, 3000: 7, 10000: 5, 30000: 4, 100000: 3, 153220: 3}
S = 256
CONFIGS = {
    "patient_only": "#1 Patient only",
    "retrieval_learned_query": "#2 Retrieval, learned query",
    "retrieval_qwen3_query": "#3 Retrieval, Qwen3 query",
    "retrieval_random_docs": "#4 Random documents",
}
SHORT = {c: name.split(" ")[0] for c, name in CONFIGS.items()}
LOSS, AUROC = "test_loss__cross_entropy", "test_metric__auroc"
N_COL, D_COL = "model_size__n_params", "dataset_size__n_subjects"
SEED, N_DRAWS = 0, 1000
QWEN3_GRID = {"W": ["4", "16", "64", "256x4x8"]}
CONTROL_GRID = {"W": ["4", "16", "64"], "D": [1000, 10000, 100000]}
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
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(train_dir / "resolved_config.yaml")
    parts = ["encoder", "head"]
    if cfg.fusion._target_.split(".")[-1] != "PassthroughFusion":
        parts += ["query_projector", "fusion"]

    def non_embedding(module: torch.nn.Module) -> int:
        emb = sum(p.numel() for m in module.modules() if isinstance(m, torch.nn.Embedding) for p in m.parameters())
        return sum(p.numel() for p in module.parameters() if p.requires_grad) - emb

    n = n_compute = 0
    for part in parts:
        if part == "query_projector" and "Qwen3" in cfg.query_projector._target_:
            continue  # frozen Qwen3 text model: no trainable parameters
        count = non_embedding(instantiate(cfg[part]))
        n_compute += count
        if part == "encoder":
            n = count
    return n, n_compute


def params_table(outputs: Path, runs: pl.DataFrame) -> pl.DataFrame:
    rows = []
    for config, w, t in runs.select("config", "W", "T").unique().iter_rows():
        train_dir = next(outputs.glob(f"{config}_W{w}_D*_T{t}/draw*/resolved_config.yaml")).parent
        n, n_compute = count_params(train_dir)
        rows.append({"config": config, "W": w, "T": t, "N": n, "N_compute": n_compute})
    return pl.DataFrame(rows)


def package_table(test: pl.DataFrame) -> pl.DataFrame:
    """Runs in the simple-scaling-laws input format: one row per run (one test evaluation each)."""
    return test.select(
        training_run_id=pl.format("{}_W{}_D{}_T{}_draw{}", "config", "W", "D", "T", "draw"),
        train_set_id=pl.format("T{}_D{}_draw{}", "T", "D", "draw"),
        test_set_id=pl.format("T{}_draw{}", "T", "draw"),
        optimizer_seed=pl.col("draw"),
        **{N_COL: pl.col("N").cast(pl.Float64), D_COL: pl.col("D").cast(pl.Float64)},
        compute__flops=pl.col("C").cast(pl.Float64),
        **{LOSS: pl.col("loss"), AUROC: pl.col("auroc")},
    )


def on_grid(frame: pl.DataFrame, grid: dict) -> pl.DataFrame:
    for col, values in grid.items():
        frame = frame.filter(pl.col(col).is_in(values))
    return frame


def cell_points(frame: pl.DataFrame) -> pl.DataFrame:
    """The trained (W, N, D) cells of a frame, as prediction points."""
    cells = frame.select("W", "N", "D").unique().sort("N", "D")
    return cells.with_columns(
        point_id=pl.format("W{}_D{}", "W", "D"),
        **{N_COL: pl.col("N").cast(pl.Float64), D_COL: pl.col("D").cast(pl.Float64)},
    )


def colors(k: int) -> list:
    return [plt.cm.viridis(x) for x in np.linspace(0.05, 0.9, max(k, 1))]


def width_key(w: str) -> tuple[int, int]:
    parts = [int(x) for x in w.split("x")]
    return parts[0], len(parts)


def groups_of(frame: pl.DataFrame, group: str) -> list:
    return sorted(frame[group].unique(), key=width_key if group == "W" else None)


def plot_lines(ax, cell, x, group, value="loss", sem="sem", label=True):
    """Value (+/- s.e. over draws) vs x, one line per group value."""
    for col, g in zip(colors(len(groups_of(cell, group))), groups_of(cell, group)):
        sub = cell.filter(pl.col(group) == g).sort(x)
        ax.errorbar(sub[x], sub[value], yerr=sub[sem], color=col, marker="o", ms=3, lw=1.2, capsize=2,
                    label=f"{group}={g}" if label else None)
    ax.set_xscale("log")
    ax.set_xlabel({"D": "training patients D", "N": "encoder parameters N"}[x])
    ax.grid(alpha=0.3)


def plot_bands(ax, frame, x, group, median, lo, hi, dots=None, dot_value=None):
    """Fitted median (line) with its 95% interval (band), one per group value; optional data dots."""
    for col, g in zip(colors(len(groups_of(frame, group))), groups_of(frame, group)):
        sub = frame.filter(pl.col(group) == g).sort(x)
        ax.plot(sub[x], sub[median], color=col, lw=1.2, label=f"{group}={g}")
        ax.fill_between(sub[x], sub[lo], sub[hi], color=col, alpha=0.2, lw=0)
        if dots is not None:
            d = dots.filter(pl.col(group) == g)
            ax.scatter(d[x], d[dot_value], color=col, s=10)
    ax.set_xscale("log")
    ax.set_xlabel({"D": "training patients D", "N": "encoder parameters N"}[x])
    ax.grid(alpha=0.3)


def mean_sem(frame: pl.DataFrame, by: list[str], value: str) -> pl.DataFrame:
    return frame.group_by(by).agg(
        loss=pl.col(value).mean(),
        sem=(pl.col(value).std() / pl.col(value).count().sqrt()).fill_null(0.0),
    )


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
        agg = mean_sem(d, ["W", "N", "D"], "diff")
        plot_lines(row[0], agg, "D", "W")
        plot_lines(row[1], agg, "N", "D")
        for ax in row:
            ax.axhline(0, color="k", lw=0.8)
            ax.set_ylabel(f"{SHORT[config]}: random − top doc loss")
            ax.legend(fontsize=7)
    fig.suptitle(f"Random document − top document test loss (above 0: the top document helps), T = {t}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "P7.png", dpi=150)
    plt.close(fig)


def fit_arm(test: pl.DataFrame):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fit(package_table(test), law="separable-power", n_draws=N_DRAWS, seed=SEED)


def fit_rows(model, config: str, t: int, test: pl.DataFrame, slaw: Path) -> list[dict]:
    """Parameters (raw scale) with 95% intervals, R^2 over runs and cell means, and RMSE per target."""
    stored = json.loads((slaw / "fits.json").read_text())
    stored = stored.get("targets", stored)
    cells = cell_points(test).join(test.group_by("N", "D").agg(loss=pl.col("loss").mean(), auroc=pl.col("auroc").mean()),
                                   on=["N", "D"])
    rows = []
    for target, column in ((LOSS, "loss"), (AUROC, "auroc")):
        raw, gof, ci = stored[target]["params_raw_scale"], stored[target]["goodness_of_fit"], model.conf_int(target)
        pred = model.predict(cells, targets=[target])[f"{target}__median"].to_numpy()
        y = cells[column].to_numpy()
        row = {"config": config, "T": t, "target": target.split("__")[-1], "runs": test.height}
        for name in ("E", "A", "alpha", "B", "beta"):
            row[name] = raw[name]
        for name in ("E", "alpha", "beta"):
            row[f"{name}_q025"], row[f"{name}_q975"] = ci[name]
        row["r2_runs"] = gof["r_squared"]
        row["r2_cells"] = float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())
        row["rmse"] = gof["rmse"]
        rows.append(row)
    return rows


def paired(models: dict, points: pl.DataFrame, reference: str) -> pl.DataFrame:
    """compare() on test loss at points, against reference; one row per non-reference system."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        table = compare(models, points.select("point_id", N_COL, D_COL), target=LOSS, reference=reference)
    return table.filter(pl.col("system") != reference).join(
        points.select("point_id", "W", "N", "D"), on="point_id"
    ).with_columns(reference=pl.lit(reference))


def law_lines(model, frame: pl.DataFrame, x: str) -> pl.DataFrame:
    """Fitted test loss along x for every trained value of the other axis."""
    group = "W" if x == "D" else "D"
    rows = []
    for g in groups_of(frame, group):
        sub = frame.filter(pl.col(group) == g)
        xs = np.geomspace(sub[x].min(), sub[x].max(), 40)
        other = float(sub["N" if x == "D" else "D"][0])
        n, d = (np.full_like(xs, other), xs) if x == "D" else (xs, np.full_like(xs, other))
        rows.append(pl.DataFrame({group: [g] * len(xs), "N": n, "D": d, N_COL: n, D_COL: d}))
    points = pl.concat(rows, how="diagonal_relaxed")
    pred = model.predict(points.drop(group, "N", "D"), targets=[LOSS])
    keep = list(dict.fromkeys([group, "N", "D"]))
    return pl.concat([points.select(keep), pred.select(pl.selectors.starts_with(LOSS))], how="horizontal")


def main(source: Path, results: Path) -> None:
    results.mkdir(parents=True, exist_ok=True)
    if source.suffix == ".csv":
        runs = pl.read_csv(source, schema_overrides={"W": pl.Utf8})
        params = pl.read_csv(results / "params.csv", schema_overrides={"W": pl.Utf8})
    else:
        runs = collect(source)
        if runs.is_empty():
            print(f"no finished runs under {source}")
            return
        params = params_table(source, runs)
        runs = runs.join(params, on=["config", "W", "T"]).with_columns(
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
    cells.sort("config", "T", "W", "D").write_csv(results / "cells.csv")

    # Fits on each configuration's full grid.
    (results / "fits").mkdir(exist_ok=True)
    models, rows, notes = {}, [], []
    for (config, t), sub in sorted(test.group_by("config", "T"), key=lambda kv: (list(CONFIGS).index(kv[0][0]), kv[0][1])):
        model = models[config, t] = fit_arm(sub)
        slaw = model.save(results / "fits" / f"{config}_T{t}.slaw")
        rows += fit_rows(model, config, t, sub, slaw)
        notes += [{"config": config, "T": t, "code": n.code, "severity": n.severity, "message": n.message}
                  for n in model.warnings]
    pl.DataFrame(rows).write_csv(results / "fits.csv")
    pl.DataFrame(notes).write_csv(results / "warnings.csv")

    # Paired comparisons (same grid, same seed) at every trained cell.
    comparisons = []
    for t in sorted(test["T"].unique()):
        tt = test.filter(pl.col("T") == t)
        found = []
        if ("retrieval_learned_query", t) in models:
            arms = {c: models[c, t] for c in ("patient_only", "retrieval_learned_query")}
            found.append(paired(arms, cell_points(tt.filter(pl.col("config") == "patient_only")), "patient_only"))
        if ("retrieval_qwen3_query", t) in models:
            grid = on_grid(tt, QWEN3_GRID)
            arms = {c: fit_arm(grid.filter(pl.col("config") == c))
                    for c in ("patient_only", "retrieval_learned_query", "retrieval_qwen3_query")}
            points = cell_points(grid.filter(pl.col("config") == "retrieval_qwen3_query"))
            for reference in ("patient_only", "retrieval_learned_query"):
                found.append(paired(arms, points, reference).filter(pl.col("system") == "retrieval_qwen3_query"))
        if ("retrieval_random_docs", t) in models:
            grid = on_grid(tt, CONTROL_GRID)
            arms = {c: fit_arm(grid.filter(pl.col("config") == c))
                    for c in ("patient_only", "retrieval_learned_query", "retrieval_random_docs")}
            points = cell_points(grid.filter(pl.col("config") == "retrieval_random_docs"))
            found.append(paired(arms, points, "retrieval_random_docs").filter(
                pl.col("system") == "retrieval_learned_query"))
            found.append(paired(arms, points, "patient_only").filter(pl.col("system") == "retrieval_random_docs"))
        comparisons += [f.with_columns(T=pl.lit(t)) for f in found]
    comparisons = pl.concat(comparisons, how="diagonal_relaxed")
    comparisons.drop("target", "direction").write_csv(results / "comparisons.csv")

    # Compute-optimal allocation from a cost model fitted to each configuration's runs.
    allocation = []
    for (config, t), model in models.items():
        sub = test.filter((pl.col("config") == config) & (pl.col("T") == t))
        cost = fit_cost_model(package_table(sub), form="power")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a = model.optimal_allocation(np.geomspace(sub["C"].min(), sub["C"].max(), 12).tolist(), cost, targets=[LOSS])
        allocation.append(a.with_columns(config=pl.lit(config), T=pl.lit(t), cost=pl.lit(cost.description)))
    allocation = pl.concat(allocation, how="diagonal_relaxed")
    allocation.write_csv(results / "allocation.csv")

    for t in sorted(cells["T"].unique()):
        out = results / f"T{t}"
        out.mkdir(exist_ok=True)
        ct = cells.filter(pl.col("T") == t)
        configs = [c for c in CONFIGS if c in ct["config"].unique().to_list()]

        # P1, P2: data. P3: fitted law (line) with its 95% interval (band), data as dots.
        for name, x, group in (("P1", "D", "W"), ("P2", "N", "D"), ("P3_D", "D", "W"), ("P3_N", "N", "D")):
            fig, axes = plt.subplots(1, len(configs), figsize=(4.2 * len(configs), 3.6), sharey=True, squeeze=False)
            for ax, config in zip(axes[0], configs):
                cc = ct.filter(pl.col("config") == config)
                if name.startswith("P3"):
                    plot_bands(ax, law_lines(models[config, t], cc, x), x, group,
                               f"{LOSS}__median", f"{LOSS}__q025", f"{LOSS}__q975", dots=cc, dot_value="loss")
                else:
                    plot_lines(ax, cc, x, group)
                ax.set_title(CONFIGS[config], fontsize=10)
            axes[0][0].set_ylabel("test loss")
            axes[0][0].legend(fontsize=7, loc="upper right")
            fig.suptitle(f"T = {t} tasks", fontsize=10)
            fig.tight_layout()
            fig.savefig(out / f"{name}.png", dpi=150)
            plt.close(fig)

        # P4: paired law difference (line, 95% band) and matched cell differences (dots).
        tt = test.filter(pl.col("T") == t)
        ctab = comparisons.filter(pl.col("T") == t)
        base = tt.filter(pl.col("config") == "patient_only").select("W", "D", "draw", base="loss")
        rows4 = [c for c in ("retrieval_learned_query", "retrieval_qwen3_query")
                 if not ctab.filter((pl.col("system") == c) & (pl.col("reference") == "patient_only")).is_empty()]
        if rows4:
            fig, axes = plt.subplots(len(rows4), 2, figsize=(9, 3.4 * len(rows4)), squeeze=False)
            for row, config in zip(axes, rows4):
                law_d = ctab.filter((pl.col("system") == config) & (pl.col("reference") == "patient_only"))
                raw = mean_sem(tt.filter(pl.col("config") == config).join(base, on=["W", "D", "draw"])
                               .with_columns(diff=pl.col("loss") - pl.col("base")), ["W", "N", "D"], "diff")
                for ax, x, group in ((row[0], "D", "W"), (row[1], "N", "D")):
                    plot_bands(ax, law_d, x, group, "difference__median", "difference__q025", "difference__q975",
                               dots=raw, dot_value="loss")
                    ax.axhline(0, color="k", lw=0.8)
                    ax.set_ylabel(f"{SHORT[config]} − #1 test loss")
                    ax.legend(fontsize=7)
            fig.suptitle(f"Retrieval − Patient only (below 0: retrieval better), T = {t}", fontsize=10)
            fig.tight_layout()
            fig.savefig(out / "P4.png", dpi=150)
            plt.close(fig)

        ctrl = ctab.filter(pl.col("reference") == "retrieval_random_docs")
        if not ctrl.is_empty():
            keys = ["W", "N", "D", "draw"]
            r = {c: tt.filter(pl.col("config") == c).select(*keys, loss=pl.col("loss"))
                 for c in ("patient_only", "retrieval_learned_query", "retrieval_random_docs")}
            pairs = (("retrieval_learned_query", "retrieval_random_docs", "#2 − #4 (knowledge)"),
                     ("retrieval_random_docs", "patient_only", "#4 − #1 (architecture)"))
            fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
            for ax, (a, b, label) in zip(axes, pairs):
                law_d = ctab.filter((pl.col("system") == a) & (pl.col("reference") == b))
                raw = mean_sem(r[a].join(r[b].drop("N"), on=["W", "D", "draw"], suffix="_b")
                               .with_columns(diff=pl.col("loss") - pl.col("loss_b")), ["W", "N", "D"], "diff")
                plot_bands(ax, law_d, "D", "W", "difference__median", "difference__q025", "difference__q975",
                           dots=raw, dot_value="loss")
                ax.axhline(0, color="k", lw=0.8)
                ax.set_title(label, fontsize=10)
            axes[0].set_ylabel("test loss difference")
            axes[1].legend(fontsize=7)
            fig.suptitle(f"Control slice, T = {t} (below 0: the first config is better)", fontsize=10)
            fig.tight_layout()
            fig.savefig(out / "P4_control.png", dpi=150)
            plt.close(fig)

        # P5: compute-optimal model size per budget. P6: loss at that optimum, with the runs as dots.
        at = allocation.filter(pl.col("T") == t)
        fig5, ax5 = plt.subplots(figsize=(5.5, 4))
        fig6, ax6 = plt.subplots(figsize=(5.5, 4))
        for col, config in zip(colors(len(configs)), configs):
            a = at.filter(pl.col("config") == config).sort("budget")
            inside = (a["domain"] == "interpolation").to_numpy()
            for ax, stem in ((ax5, N_COL), (ax6, LOSS)):
                y = a[f"{stem}__median"].to_numpy()
                ax.plot(a["budget"], y, color=col, lw=1.2, ls="--")
                ax.plot(a["budget"], np.where(inside, y, np.nan), color=col, lw=1.8, label=CONFIGS[config])
                ax.fill_between(a["budget"], a[f"{stem}__q025"], a[f"{stem}__q975"], color=col, alpha=0.2, lw=0)
            cc = ct.filter(pl.col("config") == config)
            ax6.scatter(cc["C"], cc["loss"], color=col, s=6, alpha=0.4)
        for ax, ylabel, title in ((ax5, "compute-optimal encoder parameters N*", "Compute-optimal model size"),
                                  (ax6, "test loss at N*", "Best loss vs compute (dots: cells)")):
            ax.set_xscale("log")
            ax.set_xlabel("training compute C (FLOPs)")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{title}, T = {t} (dashed: extrapolated)", fontsize=9)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7)
        ax5.set_yscale("log")
        ax6.set_ylim(ct["loss"].min() * 0.9, ct["loss"].max() * 1.1)
        for fig, name in ((fig5, "P5"), (fig6, "P6")):
            fig.tight_layout()
            fig.savefig(out / f"{name}.png", dpi=150)
            plt.close(fig)

        plot_random_doc(runs, t, out)

    print(f"{runs.height} evaluations, {cells.height} cells, {len(models)} fits -> {results}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
