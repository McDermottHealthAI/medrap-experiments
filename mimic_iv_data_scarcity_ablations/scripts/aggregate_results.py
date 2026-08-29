"""Aggregate every number for results/random_doc_null_doc_ablations_data_scarcity/.

Recomputes, from this experiment's outputs/:
  1. the reproduction scorecard vs. the published data_scarcity_retrieval test/val tables
  2. the frozen random-doc / null-doc ablation tables (test split), per fraction

Usage:  cd mimic_iv_data_scarcity_ablations && .venv/bin/python scripts/aggregate_results.py
"""

import csv
import glob
import statistics as st

FRACTIONS = (50, 20, 10, 5)

# Published val/auroc/mean numbers from results/data_scarcity_retrieval/README.md
PUB_VAL = {
    50: {
        "patient_only": [0.9040, 0.8888, 0.8909, 0.9240, 0.8648],
        "learned_linear": [0.8881, 0.8811, 0.8888, 0.9273, 0.8604],
    },
    20: {
        "patient_only": [0.8543, 0.8197, 0.8648, 0.8911, 0.8314],
        "learned_linear": [0.8910, 0.8428, 0.8676, 0.9022, 0.8316],
    },
    10: {
        "patient_only": [0.8547, 0.8285, 0.8366, 0.8364, 0.7945],
        "learned_linear": [0.8677, 0.8326, 0.8478, 0.8651, 0.7942],
    },
    5: {
        "patient_only": [0.8392, 0.7843, 0.8143, 0.8254, 0.7593],
        "learned_linear": [0.8640, 0.8057, 0.8321, 0.8318, 0.7692],
    },
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
    print("=== 1. Reproduction scorecard (test split vs. published val, sanity direction only) ===")
    for frac in FRACTIONS:
        po = draws(f"outputs/patient_only_data_scarcity_n25_30d_train{frac}pct_test_repro/draw{{d}}")
        ll = draws(
            f"outputs/marginalized_binary_learned_linear_data_scarcity_n25_30d_train{frac}pct_test_repro/draw{{d}}"
        )
        print(f"  train{frac}pct: patient_only test {fmt(po)} | learned_linear test {fmt(ll)}")

    print("\n=== 2. Ablation tables (test split), per fraction ===")
    for frac in FRACTIONS:
        po = draws(f"outputs/patient_only_data_scarcity_n25_30d_train{frac}pct_test_repro/draw{{d}}")
        real = draws(
            f"outputs/marginalized_binary_learned_linear_data_scarcity_n25_30d_train{frac}pct_test_repro/draw{{d}}"
        )
        rand = [
            st.mean(
                metric(
                    f"outputs/marginalized_binary_learned_linear_data_scarcity_n25_30d_train{frac}pct_test_eval_random_docs_rep{r}/draw{d}",
                    "test/auroc/mean",
                )
                for r in (1, 2, 3)
            )
            for d in range(1, 6)
        ]
        null = draws(
            f"outputs/marginalized_binary_learned_linear_data_scarcity_n25_30d_train{frac}pct_test_eval_null_docs/draw{{d}}"
        )
        print(f"  train{frac}pct:")
        print(f"    patient_only  {fmt(po)}")
        print(f"    real docs     {fmt(real)}")
        print(f"    random docs   {fmt(rand)} (Δ vs real {st.mean(rand) - st.mean(real):+.4f})")
        print(f"    null docs     {fmt(null)} (Δ vs real {st.mean(null) - st.mean(real):+.4f})")


if __name__ == "__main__":
    main()
