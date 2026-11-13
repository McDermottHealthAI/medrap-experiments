"""Scaling-grid analysis (design.md §10): paired Δ per cell, law fits per arm, comparison, compute.

    python scaling_laws_v2/grid.py [OUTPUTS_DIR]

Arms (all warm-started from the same patient-only checkpoint, so every Δ is paired per draw):
    patient only    patient_only_warm, test on its last checkpoint
    xres            xres_warm_pre (K 4), top-K documents
    xres random     xres_random_docs_warm_pre, trained and tested on random documents

Writes, next to this file:
    results_grid.csv    one row per run: arm, W, N, D, draw, test loss, training compute
    grid_delta.csv      L(patient only) − L(arm) per W × D, mean over draws with a 95% t-interval
    grid_fits.csv       every law form per arm, fit on draw-centred loss (below): parameters, R² over cell means, leave-one-cell-out
                        RMSE, run-to-run SD, warnings, whether it passes the quality gate, and
                        which form is kept: one form for every arm (compare() needs the same law),
                        the one passing the gate for the most arms, then lowest mean
                        leave-one-cell-out RMSE
    grid_compare.csv    simple_scaling_laws.compare of the kept laws against patient only, at the
                        trained cells and beyond them

Each draw has its own 1,000 tasks, so loss levels differ between draws by far more than the arms
differ. Every arm and cell of a draw shares its tasks (each D subsamples the same draw), so the laws
are fit on draw-centred loss: each draw's mean over the cells present in every draw is subtracted
and the overall mean added back. Differences between arms are unchanged; the paired Δ table uses
the raw losses.

N is the trained parameters without the code table, and training compute includes the patient-only
checkpoint each run starts from, both from results_compute.csv (count_compute.py).
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats
from simple_scaling_laws import compare, fit

from ladder import RUN

HERE = Path(__file__).parent
ARMS = {"patient_only_warm": ("patient only", "test_last"), "xres_warm_pre": ("xres", "test_topK"),
        "xres_random_docs_warm_pre": ("xres random", "test_topK")}
COUNTED = {"patient only": "patient_only", "xres": "xres", "xres random": "xres"}
LAWS = ["separable-power", "shifted-separable-power", "kaplan-joint-power", "skaling-joint-power",
        "bounded-rational-power"]
BAD = {"lack_of_fit", "ill_conditioned_fit", "optimizer_nonconvergence"}
EPOCHS = {3000: 7, 10000: 5, 30000: 4, 100000: 3, 153220: 3}  # run.sh
LOSS = "test_loss__bce"


def compute_table() -> pl.DataFrame:
    """Per (counted config, W): FLOPs per patient step (with search) and N, from results_compute.csv."""
    c = pl.read_csv(HERE / "results_compute.csv")
    c = c.with_columns(m=pl.col("run").str.extract_groups(r"^(?P<cfg>.+?)_W(?P<W>[^_]+)_D")).unnest("m")
    c = c.with_columns(warm=pl.col("run").str.contains("_warm"))
    c = c.filter(~((pl.col("cfg") == "patient_only") & pl.col("warm")))
    return c.group_by("cfg", "W").agg(flops=(pl.col("flops_step") + pl.col("flops_faiss")).first(),
                                      N=pl.col("N").first())


def collect(outputs: Path) -> pl.DataFrame:
    cost = compute_table()
    cold = cost.filter(pl.col("cfg") == "patient_only").select("W", cold_flops="flops")
    rows = []
    for run in outputs.iterdir():
        m = RUN.fullmatch(run.name)
        if not m or m["lr"] != "1e-3" or m["dropout"] != "0" or m["K"] not in (None, "4"):
            continue
        arm = ARMS.get(m["config"] + (m["suffix"] or ""))
        if arm is None:
            continue
        for draw in sorted(run.glob("draw*")):
            files = sorted(draw.glob(f"{arm[1]}/loggers/csv/*/metrics.csv"))
            if files:
                loss = pl.read_csv(files[-1]).select("test/loss").drop_nulls().item(-1, 0)
                rows.append({"arm": arm[0], "W": m["W"], "D": int(m["D"]), "draw": draw.name, "loss": loss})
    df = pl.DataFrame(rows).with_columns(cfg=pl.col("arm").replace(COUNTED))
    df = df.join(cost, on=["cfg", "W"], how="left").join(cold, on="W", how="left")
    missing = df.filter(pl.col("N").is_null())["W"].unique().to_list()
    if missing:
        raise SystemExit(f"no compute counts for W {missing}: run count_compute.py on those runs first")
    epochs = pl.col("D").replace_strict(EPOCHS, default=3)
    return df.with_columns(C=(pl.col("flops") + pl.col("cold_flops")) * pl.col("D") * epochs).drop("cfg")


def t_interval(x: list[float]) -> float:
    return float(stats.t.ppf(0.975, len(x) - 1) * np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan")


def deltas(df: pl.DataFrame) -> pl.DataFrame:
    base = df.filter(pl.col("arm") == "patient only").select("W", "D", "draw", base="loss")
    paired = df.filter(pl.col("arm") != "patient only").join(base, on=["W", "D", "draw"])
    rows = []
    for (arm, w, d), g in paired.group_by("arm", "W", "D"):
        x = ((g["base"] - g["loss"]) * 1e4).to_list()
        rows.append({"arm": arm, "W": w, "D": d, "n": len(x), "delta_e4": np.mean(x), "ci_e4": t_interval(x),
                     "delta_pct": float(((g["base"] - g["loss"]) / g["base"]).mean() * 100),
                     "draws_better": sum(v > 0 for v in x)})
    return pl.DataFrame(rows).sort("arm", "W", "D")


def centre(df: pl.DataFrame) -> pl.DataFrame:
    """Subtract each draw's mean loss over the (arm, W, D) cells that every draw has."""
    n_draws = df["draw"].n_unique()
    full = df.group_by("arm", "W", "D").agg(n=pl.col("draw").n_unique()).filter(pl.col("n") == n_draws)
    balanced = df.join(full.select("arm", "W", "D"), on=["arm", "W", "D"])
    offset = balanced.group_by("draw").agg(offset=pl.col("loss").mean() - balanced["loss"].mean())
    return df.join(offset, on="draw").with_columns(loss=pl.col("loss") - pl.col("offset")).drop("offset")


def table(df: pl.DataFrame) -> pl.DataFrame:
    return df.select(training_run_id=pl.concat_str("arm", "W", "D", "draw", separator="|"),
                     test_set_id=pl.concat_str("D", "draw", separator="|"),
                     model_size__n=pl.col("N").cast(pl.Float64), dataset_size__d=pl.col("D").cast(pl.Float64),
                     **{LOSS: pl.col("loss")})


def fit_quiet(frame: pl.DataFrame, law: str, n_draws: int):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fit(frame, law=law, n_draws=n_draws, seed=0)


def cell_means(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.group_by("model_size__n", "dataset_size__d").agg(y=pl.col(LOSS).mean(), sd=pl.col(LOSS).std())


def fits(df: pl.DataFrame) -> tuple[pl.DataFrame, dict]:
    rows = []
    for arm, g in df.group_by("arm"):
        arm = arm[0]
        frame, cells = table(g), cell_means(table(g))
        run_sd = float(np.sqrt((cells["sd"] ** 2).mean()))
        for law in LAWS:
            try:
                model = fit_quiet(frame, law, 200)
            except Exception as e:  # a form that cannot be built here is recorded, not fatal
                rows.append({"arm": arm, "law": law, "error": str(e)[:200]})
                continue
            pred = model.predict(cells, targets=[LOSS])[f"{LOSS}__median"].to_numpy()
            y = cells["y"].to_numpy()
            r2 = float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())
            loo = []
            for n, d in cells.select("model_size__n", "dataset_size__d").iter_rows():
                held = (pl.col("model_size__n") == n) & (pl.col("dataset_size__d") == d)
                try:
                    m = fit_quiet(frame.filter(~held), law, 20)
                    p = m.predict(cells.filter(held), targets=[LOSS])[f"{LOSS}__median"][0]
                    loo.append(float(p) - float(cells.filter(held)["y"][0]))
                except Exception:
                    loo.append(float("nan"))
            loo_rmse = float(np.sqrt(np.nanmean(np.square(loo))))
            codes = sorted({getattr(n, "code", str(n)) for n in model.warnings})
            passes = r2 >= 0.9 and loo_rmse < run_sd and not (BAD & set(codes))
            params = model.params(LOSS) if callable(model.params) else model.params
            rows.append({"arm": arm, "law": law, "r2_cells": r2, "loo_rmse": loo_rmse, "run_sd": run_sd,
                         "passes": passes, "warnings": ";".join(codes), "params": str(params)})
    out = pl.DataFrame(rows, infer_schema_length=None).filter(pl.col("loo_rmse").is_not_null())
    law = (out.group_by("law").agg(n_pass=pl.col("passes").sum(), loo=pl.col("loo_rmse").mean())
           .sort(["n_pass", "loo"], descending=[True, False])["law"][0])
    kept = {arm[0]: fit_quiet(table(g), law, 1000) for arm, g in df.group_by("arm")}
    return out.with_columns(kept=pl.col("law") == law), kept


def comparison(df: pl.DataFrame, kept: dict) -> pl.DataFrame:
    cells = df.select("W", "N", "D").unique()
    beyond = pl.DataFrame({"W": ["256x4x8"] * 2, "D": [500_000, 1_000_000]}).join(
        cells.filter(pl.col("W") == "256x4x8").select("W", "N").unique(), on="W")
    points = []
    for arm in ("xres", "xres random"):
        if arm not in kept or "patient only" not in kept:
            continue
        # The arms have different N at the same W (the reader's parameters), so compare per W at each arm's N.
        for w, n_arm in df.filter(pl.col("arm") == arm).select("W", "N").unique().iter_rows():
            n_base = df.filter((pl.col("arm") == "patient only") & (pl.col("W") == w))["N"][0]
            for d in sorted({*cells["D"].to_list(), *beyond["D"].to_list()}):
                pid = f"{w}|{d}"
                points.append({"arm": arm, "point_id": pid, "W": w, "D": d, "n_arm": n_arm, "n_base": n_base})
    rows = []
    for arm in ("xres", "xres random"):
        pts = pl.DataFrame([p for p in points if p["arm"] == arm])
        if pts.is_empty():
            continue
        # compare() evaluates every system at the same point, so the reference is predicted at the
        # arm's N; the reader adds < 2.2x parameters, inside the law's fitted range.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = compare({"patient only": kept["patient only"], arm: kept[arm]},
                          pts.select("point_id", model_size__n=pl.col("n_arm").cast(pl.Float64),
                                     dataset_size__d=pl.col("D").cast(pl.Float64)),
                          target=LOSS, reference="patient only")
        rows.append(out.filter(pl.col("system") == arm).join(pts.select("point_id", "W", "D"), on="point_id"))
    return pl.concat(rows, how="diagonal_relaxed") if rows else pl.DataFrame()


if __name__ == "__main__":
    outputs = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "outputs"
    df = collect(outputs)
    df.write_csv(HERE / "results_grid.csv")
    pl.Config.set_tbl_rows(100)
    pl.Config.set_tbl_width_chars(200)
    d = deltas(df)
    d.write_csv(HERE / "grid_delta.csv")
    print("Paired Δ = L(patient only) − L(arm), ×1e-4 with 95% t-interval; positive: the arm is better")
    print(d)
    f, kept = fits(centre(df))
    f.write_csv(HERE / "grid_fits.csv")
    with pl.Config(fmt_str_lengths=120):
        print(f.drop("params", strict=False))
    c = comparison(centre(df), kept)
    if not c.is_empty():
        c.write_csv(HERE / "grid_compare.csv")
        print(c)
