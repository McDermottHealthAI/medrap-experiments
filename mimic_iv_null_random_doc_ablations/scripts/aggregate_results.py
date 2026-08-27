"""Aggregate every number reported in results/random_doc_null_doc_ablations_mimic_iv/.

Recomputes, from this experiment's outputs/ (and, for the peak-to-peak check, the
original training logs referenced read-only):
  1. the reproduction scorecard vs the published capacity_starved_retrieval tables
  2. the frozen random-doc / null-doc ablation tables (test split)
  3. the central-table cells (mean +/- population sd over the 5 draws)
  4. the log-based peak-to-peak robustness check

Usage:  cd mimic_iv_null_random_doc_ablations && .venv/bin/python scripts/aggregate_results.py
"""

import csv
import glob
import statistics as st

ORIG = "/groups/mm6677_gp/hs3627/medrap-experiments/mimic_iv_sweep/outputs"
ARMS = {
    "patient_only": "patient_only_capacity_starved_n25_30d",
    "learned_linear": "marginalized_binary_learned_linear_capacity_starved_n25_30d",
    "qwen3_text": "marginalized_binary_qwen3_text_capacity_starved_n25_30d",
}
# Published tables from results/capacity_starved_retrieval/README.md
PUB_TEST = {
    "patient_only": [0.8635, 0.8098, 0.8503, 0.8715, 0.7950],
    "learned_linear": [0.8581, 0.8276, 0.8596, 0.8881, 0.8050],
    "qwen3_text": [0.8723, 0.8335, 0.8653, 0.8934, 0.8224],
}
PUB_VAL = {
    "patient_only": [0.8565, 0.8202, 0.8534, 0.8805, 0.7904],
    "learned_linear": [0.8583, 0.8400, 0.8567, 0.9066, 0.8023],
    "qwen3_text": [0.8725, 0.8525, 0.8638, 0.9204, 0.8160],
}


def metric(run_dir: str, key: str) -> float:
    files = sorted(glob.glob(f"{run_dir}/loggers/csv/*/metrics.csv"))
    rows = [r for r in csv.DictReader(open(files[-1])) if r.get(key)]
    return float(rows[-1][key])


def fmt(vals):
    return f"{st.mean(vals):.4f} ± {st.pstdev(vals):.4f}"


def draws(pattern: str, key: str = "test/auroc/mean"):
    return [metric(pattern.format(d=d), key) for d in range(1, 6)]


def main() -> None:
    print("=== 1. Reproduction scorecard ===")
    for split, pub, suffix, key in (
        ("test", PUB_TEST, "test_repro", "test/auroc/mean"),
        ("val (200-batch protocol)", PUB_VAL, "val_repro200", "val/auroc/mean"),
    ):
        n = ok = 0
        for arm, vals in pub.items():
            rep = draws(f"outputs/{ARMS[arm]}_{suffix}/draw{{d}}", key)
            for r, p in zip(rep, vals):
                n += 1
                if abs(r - p) < 0.0005:
                    ok += 1
                else:
                    print(f"    off: {split} {arm}: {r:.4f} vs published {p:.4f}")
        print(f"  {split}: {ok}/{n} exact (4-decimal)")

    print("\n=== 2+3. Ablation tables / central-table cells (test split) ===")
    po = draws(f"outputs/{ARMS['patient_only']}_test_repro/draw{{d}}")
    print(f"  patient_only          {fmt(po)}")
    for arm in ("learned_linear", "qwen3_text"):
        real = draws(f"outputs/{ARMS[arm]}_test_repro/draw{{d}}")
        rand = [
            st.mean(
                metric(f"outputs/{ARMS[arm]}_test_eval_random_docs_rep{r}/draw{d}", "test/auroc/mean")
                for r in (1, 2, 3)
            )
            for d in range(1, 6)
        ]
        null = draws(f"outputs/{ARMS[arm]}_test_eval_null_docs/draw{{d}}")
        print(f"  {arm}: real {fmt(real)} | random {fmt(rand)} (Δ {st.mean(rand)-st.mean(real):+.4f})"
              f" | null {fmt(null)} (Δ {st.mean(null)-st.mean(real):+.4f})")

    print("\n=== 4. Peak-to-peak robustness check (val split, original training logs) ===")
    peak, last = {}, {}
    for arm, d in ARMS.items():
        peak[arm], last[arm] = [], []
        for i in range(1, 6):
            f = sorted(glob.glob(f"{ORIG}/{d}/draw{i}/loggers/csv/*/metrics.csv"))[-1]
            va = [float(r["val/auroc/mean"]) for r in csv.DictReader(open(f)) if r.get("val/auroc/mean")]
            peak[arm].append(max(va))
            last[arm].append(va[-1])
    for arm in ("learned_linear", "qwen3_text"):
        dl = [last[arm][i] - last["patient_only"][i] for i in range(5)]
        dp = [peak[arm][i] - peak["patient_only"][i] for i in range(5)]
        print(f"  Δ {arm}: last-to-last {st.mean(dl):+.4f} ({sum(x > 0 for x in dl)}/5)"
              f" | peak-to-peak {st.mean(dp):+.4f} ({sum(x > 0 for x in dp)}/5)")


if __name__ == "__main__":
    main()
