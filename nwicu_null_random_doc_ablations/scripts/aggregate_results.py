"""Aggregate the step-matched (42-epoch) NWICU results for
results/random_doc_null_doc_ablations_nwicu/.

Computes, from this experiment's outputs/ (test split, last.ckpt == best-val-loss
checkpoint in this Lightning config):
  1. central-table cells per variant: patient_only / real docs / random-doc
     ablation (mean of 3 reps) / null-doc ablation, mean +/- population sd, 5 draws
  2. deltas vs patient_only and vs real docs
  3. train-time random_docs control (learned-linear only)
  4. gate diagnostics at end of training (eff_k, score_std) per retrieval variant

The qwen3_text sections are skipped gracefully until its runs exist.

Usage:  cd nwicu_null_random_doc_ablations && .venv/bin/python scripts/aggregate_results.py
"""

import csv
import glob
import statistics as st

ARMS = {
    "learned_linear": "marginalized_binary_learned_linear_capacity_starved_n25_30d",
    "qwen3_text": "marginalized_binary_qwen3_text_capacity_starved_n25_30d",
}
PO = "patient_only_capacity_starved_n25_30d"


def metric(run_dir: str, key: str) -> float | None:
    files = sorted(glob.glob(f"{run_dir}/loggers/csv/*/metrics.csv"))
    if not files:
        return None
    rows = [r for r in csv.DictReader(open(files[-1])) if r.get(key)]
    return float(rows[-1][key]) if rows else None


def draws(pattern: str, key: str = "test/auroc/mean"):
    vals = [metric(pattern.format(d=d), key) for d in range(1, 6)]
    return vals if all(v is not None for v in vals) else None


def fmt(vals):
    return f"{st.mean(vals):.4f} ± {st.pstdev(vals):.4f}"


def main() -> None:
    po = draws(f"outputs/{PO}_test_eval/draw{{d}}")
    print("=== NWICU step-matched (42 epochs ≈ 14.4k steps), test split, n=5 draws ===")
    print(f"patient_only (no retrieval)   {fmt(po)}")

    for arm, d in ARMS.items():
        real = draws(f"outputs/{d}_test_eval/draw{{d}}")
        if real is None:
            print(f"\n{arm}: runs not present yet — skipped")
            continue
        rand = [
            st.mean(
                m for r in (1, 2, 3)
                if (m := metric(f"outputs/{d}_test_eval_random_docs_rep{r}/draw{i}", "test/auroc/mean")) is not None
            )
            for i in range(1, 6)
        ]
        null = draws(f"outputs/{d}_test_eval_null_docs/draw{{d}}")
        print(f"\n{arm}:")
        print(f"  real docs      {fmt(real)}   (Δ vs patient_only {st.mean(real)-st.mean(po):+.4f})")
        print(f"  random abl     {fmt(rand)}   (Δ vs real {st.mean(rand)-st.mean(real):+.4f})")
        print(f"  null abl       {fmt(null)}   (Δ vs real {st.mean(null)-st.mean(real):+.4f})")
        print(f"  per-draw real: {[f'{x:.4f}' for x in real]}")

        # gate diagnostics at end of training
        ek, ss = [], []
        for i in range(1, 6):
            f = sorted(glob.glob(f"outputs/{d}/draw{i}/loggers/csv/*/metrics.csv"))
            if not f:
                continue
            rows = list(csv.DictReader(open(f[-1])))
            for key, acc in (("retrieval/train/differentiable/effective_k_mean", ek),
                             ("retrieval/train/differentiable/score_std", ss)):
                vals = [float(r[key]) for r in rows if r.get(key)]
                if vals:
                    acc.append(vals[-1])
        if ek:
            print(f"  gate at end of training: eff_k {st.mean(ek):.2f}, score_std {st.mean(ss):.2f}")

    trand = draws(f"outputs/{ARMS['learned_linear']}_train_random_docs_test_eval/draw{{d}}")
    if trand:
        print(f"\ntrain-time random (ll)        {fmt(trand)}")


if __name__ == "__main__":
    main()
