"""Validation loss during training, by document encoding (advisor: do richer encodings reach the same
performance earlier in training?).

    python scaling_laws_v2/curves.py

Reads val/loss (15 evaluations per run) from every draw of the arms below at D 10K and 153K, and
writes curves.csv (mean over draws per evaluation) and curves.png (one panel per W and D). Warm runs
start from the patient-only checkpoint, so their first evaluation is already low; patient only (cold)
is the run they start from.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

OUT = Path("outputs")
ARMS = {  # label: run-name suffix after _W{W}_D{D}_lr1e-3_do0
    "patient only": ("patient_only", ""),
    "patient only, warm": ("patient_only", "_warm"),
    "retrieval only, word pieces (cold)": ("retrieval_only", "_K4"),
    "retrieval only, last layer": ("retrieval_only", "_K4_warm_pre"),
    "retrieval only, all layers": ("retrieval_only", "_K4_warm_attend"),
    "xres, last layer": ("xres", "_K4_warm_pre"),
    "xres, all layers": ("xres", "_K4_warm_attend"),
}
STYLE = {
    "patient only": ("#555555", "-"),
    "patient only, warm": ("#555555", "--"),
    "retrieval only, word pieces (cold)": ("#c44e52", ":"),
    "retrieval only, last layer": ("#c44e52", "--"),
    "retrieval only, all layers": ("#c44e52", "-"),
    "xres, last layer": ("#4c72b0", "--"),
    "xres, all layers": ("#4c72b0", "-"),
}


def load() -> pl.DataFrame:
    frames = []
    for w in ("64x2x4", "256x4x8"):
        for d in (10000, 153220):
            for label, (config, suffix) in ARMS.items():
                run = OUT / f"{config}_W{w}_D{d}_lr1e-3_do0{suffix}"
                for f in sorted(run.glob("draw[1-3]/train/loggers/csv/*/metrics.csv")):
                    m = pl.read_csv(f, infer_schema_length=0).select("step", "val/loss")
                    m = m.filter(pl.col("val/loss").is_not_null() & (pl.col("val/loss") != "")).cast(pl.Float64)
                    frames.append(
                        m.with_columns(
                            progress=pl.col("step") / pl.col("step").max(),
                            arm=pl.lit(label),
                            W=pl.lit(w),
                            D=pl.lit(d),
                            draw=pl.lit(f.parts[-6]),
                        )
                    )
    return pl.concat(frames)


if __name__ == "__main__":
    df = load()
    curves = (
        df.with_columns(eval=pl.col("step").rank("dense").over("arm", "W", "D", "draw"))
        .group_by("arm", "W", "D", "eval")
        .agg(progress=pl.col("progress").mean(), val_loss=pl.col("val/loss").mean(), n=pl.len())
        .sort("W", "D", "arm", "eval")
    )
    curves.write_csv("curves.csv")

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    handles = {}
    for ax, (w, d) in zip(axes.flat, [(w, d) for w in ("64x2x4", "256x4x8") for d in (10000, 153220)]):
        part = curves.filter((pl.col("W") == w) & (pl.col("D") == d))
        for label in ARMS:
            arm = part.filter(pl.col("arm") == label)
            if arm.height:
                color, ls = STYLE[label]
                (handles[label],) = ax.plot(arm["progress"], arm["val_loss"], color=color, ls=ls, lw=2)
        po = part.filter(pl.col("arm") == "patient only")["val_loss"]
        ax.set_ylim(po.min() * 0.97, po.min() * 1.25)
        ax.set_title(f"W {w}, D {d:,}")
        ax.set_xlabel("fraction of training")
        ax.set_ylabel("validation loss")
        ax.grid(alpha=0.3)
    order = [a for a in ARMS if a in handles]
    fig.legend([handles[a] for a in order], order, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig("curves.png", dpi=130)
    # Loss at a quarter, half and all of training, as % above patient only's final validation loss.
    final_po = curves.filter(pl.col("arm") == "patient only").group_by("W", "D").agg(po=pl.col("val_loss").last())
    table = (
        curves.join(final_po, on=["W", "D"])
        .with_columns(gap=(pl.col("val_loss") / pl.col("po") - 1) * 100)
        .group_by("arm", "W", "D")
        .agg(
            at_25=pl.col("gap").filter(pl.col("progress") >= 0.25).first().round(2),
            at_50=pl.col("gap").filter(pl.col("progress") >= 0.5).first().round(2),
            at_100=pl.col("gap").last().round(2),
            n=pl.col("n").max(),
        )
        .sort("W", "D", "arm")
    )
    pl.Config.set_tbl_rows(100)
    pl.Config.set_fmt_str_lengths(40)
    print(table)
    table.write_csv("curves_table.csv")
