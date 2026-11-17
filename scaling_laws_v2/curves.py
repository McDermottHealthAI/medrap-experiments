"""Validation loss against training examples seen, by document encoding (do richer encodings reach
the same performance with less training?).

    python scaling_laws_v2/curves.py

Reads val/loss (15 evaluations per run) from every draw of the arms below at D 10K and 153K. The
x-axis is patients seen in training, optimizer steps × 32 (the effective batch). A warm run starts
from its patient-only checkpoint, so it is shifted right by everything that run saw: warm and cold
arms are compared at the same total training. Writes curves.csv (mean over draws per evaluation),
curves.png (one panel per W and D) and curves_table.csv.
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
    "retrieval only, learned query": ("retrieval_only_learned", "_K4_warm_pre"),
    "xres, last layer": ("xres", "_K4_warm_pre"),
    "xres, all layers": ("xres", "_K4_warm_attend"),
}
STYLE = {
    "patient only": ("#555555", "-"),
    "patient only, warm": ("#555555", "--"),
    "retrieval only, word pieces (cold)": ("#c44e52", ":"),
    "retrieval only, last layer": ("#c44e52", "--"),
    "retrieval only, all layers": ("#c44e52", "-"),
    "retrieval only, learned query": ("#c44e52", "-."),
    "xres, last layer": ("#4c72b0", "--"),
    "xres, all layers": ("#4c72b0", "-"),
}


BATCH = 32


def final_step(run_dir: Path) -> int:
    f = next(run_dir.glob("train/loggers/csv/*/metrics.csv"))
    return int(pl.read_csv(f, infer_schema_length=0).select(pl.col("step").cast(pl.Int64).max()).item())


def load() -> pl.DataFrame:
    frames = []
    for w in ("64x2x4", "256x4x8"):
        for d in (10000, 153220):
            for label, (config, suffix) in ARMS.items():
                run = OUT / f"{config}_W{w}_D{d}_lr1e-3_do0{suffix}"
                for f in sorted(run.glob("draw[1-3]/train/loggers/csv/*/metrics.csv")):
                    m = pl.read_csv(f, infer_schema_length=0).select("step", "val/loss")
                    m = m.filter(pl.col("val/loss").is_not_null() & (pl.col("val/loss") != "")).cast(pl.Float64)
                    seen = pl.col("step") * BATCH
                    if "warm" in suffix:  # add what the patient-only checkpoint was trained on
                        cold = run.parent / f"patient_only_W{w}_D{d}_lr1e-3_do0" / f.parts[-6]
                        seen = seen + final_step(cold) * BATCH
                    frames.append(
                        m.with_columns(
                            progress=seen,
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

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
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
        ax.set_xlabel("patients seen in training (incl. the warm-start checkpoint)")
        ax.xaxis.set_major_formatter(lambda x, _: f"{x / 1e3:.0f}K")
        ax.set_ylabel("validation loss")
        ax.grid(alpha=0.3)
    order = [a for a in ARMS if a in handles]
    fig.legend([handles[a] for a in order], order, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig("curves.png", dpi=130)
    # Final validation loss above patient only's final loss, and patients seen until an arm first
    # reaches patient only's final loss (empty if it never does).
    final_po = curves.filter(pl.col("arm") == "patient only").group_by("W", "D").agg(po=pl.col("val_loss").last())
    table = (
        curves.join(final_po, on=["W", "D"])
        .group_by("arm", "W", "D")
        .agg(
            seen_at_end_K=(pl.col("progress").last() / 1e3).round(0),
            final_gap_pct=((pl.col("val_loss").last() / pl.col("po").first() - 1) * 100).round(2),
            seen_to_reach_po_K=(pl.col("progress").filter(pl.col("val_loss") <= pl.col("po")).first() / 1e3).round(0),
            n=pl.col("n").max(),
        )
        .sort("W", "D", "seen_at_end_K", "arm")
    )
    pl.Config.set_tbl_rows(100)
    pl.Config.set_fmt_str_lengths(40)
    print(table)
    table.write_csv("curves_table.csv")
