"""Stage 0 and Stage 1 results (design.md §7) from outputs/.

    python scaling_laws_v2/ladder.py [OUTPUTS_DIR]

Writes results.csv (one row per test) and prints:
    Stage 0   patient-only test loss per W x lr x dropout (D 153K), best and last checkpoint
    Stage 1   paired difference L(#1) - L(config) per W x D (positive: the config beats patient
              only), mean over draws with a 95% t-interval, for every test of every rung
    gates     tanh(g) at the end of training for the gated configurations
    retrieval share of test patients sharing the most common top document, and distinct top
              documents per patient
"""

import re
import sys
from pathlib import Path

import polars as pl
import torch

RUN = re.compile(r"(?P<config>.+)_W(?P<W>[^_]+)_D(?P<D>\d+)_lr(?P<lr>[^_]+)_do(?P<dropout>[^_/]+)(_K(?P<K>\d+))?(?P<suffix>(_nonull|_z|_warm|_c|_last|_mix)*)")
T_975 = 4.303  # t quantile, 2 degrees of freedom (3 draws)


def collect(outputs: Path) -> pl.DataFrame:
    rows = []
    for f in sorted(outputs.glob("*/draw*/test_*/loggers/csv/*/metrics.csv")):
        run, draw, test = f.parts[-7], f.parts[-6], f.parts[-5]
        m = RUN.fullmatch(run).groupdict()
        last = pl.read_csv(f).tail(1).to_dicts()[0]
        rows.append({
            "config": m["config"] + m["suffix"],
            **{k: m[k] for k in ("W", "lr", "dropout")},
            "D": int(m["D"]),
            "draw": int(draw.removeprefix("draw")),
            "test": test.removeprefix("test_"),
            "loss": last["test/loss"],
            "auroc": last["test/auroc/mean"],
            "top1_mode_frac": last.get("retrieval/test/top1_mode_frac") if m["K"] else None,
            "top1_unique_ratio": last.get("retrieval/test/top1_unique_ratio") if m["K"] else None,
        })
    return pl.DataFrame(rows, infer_schema_length=None)


def gates(outputs: Path) -> pl.DataFrame:
    rows = []
    for ckpt in sorted(outputs.glob("*/draw*/train/checkpoints/last.ckpt")):
        state = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
        gate = next((v for k, v in state.items() if k.endswith("fusion.gate")), None)
        if gate is not None:
            m = RUN.fullmatch(ckpt.parts[-5]).groupdict()
            rows.append({"config": m["config"] + m["suffix"], "W": m["W"], "D": int(m["D"]), "gate": float(torch.tanh(gate))})
    return pl.DataFrame(rows, infer_schema_length=None)


def main(outputs: Path) -> None:
    pl.Config.set_tbl_rows(100)
    pl.Config.set_tbl_cols(20)
    pl.Config.set_tbl_hide_dataframe_shape(True)
    pl.Config.set_tbl_width_chars(200)
    runs = collect(outputs)
    runs.write_csv(outputs.parent / "results.csv")

    stage0 = runs.filter((pl.col("config") == "patient_only") & (pl.col("D") == 153220))
    print("Stage 0: patient-only test loss, D 153K (mean over draws)")
    print(
        stage0.group_by("W", "lr", "dropout", "test")
        .agg(pl.col("loss").mean().round(5), pl.col("auroc").mean().round(3), pl.len().alias("n"))
        .pivot(on="test", index=["W", "lr", "dropout"], values=["loss", "auroc"])
        .sort("W", "lr", "dropout")
    )

    stage1 = runs.filter((pl.col("lr") == "1e-3") & (pl.col("dropout") == "0"))
    base = stage1.filter((pl.col("config") == "patient_only") & (pl.col("test") == "last")).select(
        "W", "D", "draw", pl.col("loss").alias("base")
    )
    paired = (
        stage1.filter(pl.col("config") != "patient_only")
        .join(base, on=["W", "D", "draw"])
        .with_columns(delta=pl.col("base") - pl.col("loss"), rel=(pl.col("base") - pl.col("loss")) / pl.col("base") * 100)
        .group_by("config", "test", "W", "D")
        .agg(
            pl.len().alias("n"),
            (pl.col("delta").mean() * 1e4).round(2).alias("Δ×1e4"),
            (T_975 * pl.col("delta").std() / pl.len().sqrt() * 1e4).round(2).alias("±95%"),
            pl.col("rel").mean().round(2).alias("Δ %"),
            (pl.col("delta") > 0).sum().alias("draws better"),
            pl.col("auroc").mean().round(3),
        )
        .sort("W", "D", "config", "test")
    )
    print("\nStage 1: L(patient only) - L(config), positive = config better; lr 1e-3, dropout 0, K 4")
    print(paired)

    g = gates(outputs)
    if g.height:
        print("\nGate tanh(g) at the end of training (mean over draws)")
        print(g.group_by("config", "W", "D").agg(pl.col("gate").mean().round(3), pl.col("gate").min().round(3).alias("min")).sort("config", "W", "D"))

    print("\nRetrieval diversity at test (top K test, mean over draws)")
    print(
        stage1.filter(pl.col("test") == "topK", ~pl.col("config").str.contains("random"))
        .group_by("config", "W", "D")
        .agg(pl.col("top1_mode_frac").mean().round(3), pl.col("top1_unique_ratio").mean().round(3))
        .sort("config", "W", "D")
    )


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / "outputs"))
