"""Summarize saved per-run metrics for the 27-run experiment.

python scripts/aggregate.py --runs /authorized/runs
"""

import argparse
import json
import statistics
from pathlib import Path

TASK_SEEDS = [101, 202, 303]
TRAINING_SEEDS = [42, 43, 44]
ARMS = ["patient_only", "real", "random"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--split", choices=["tuning", "held_out"], default="held_out")
    args = parser.parse_args()
    results = {}
    for task in TASK_SEEDS:
        for seed in TRAINING_SEEDS:
            for arm in ARMS:
                run = args.runs / f"task{task}_seed{seed}_{arm}"
                complete = json.loads((run / "complete.json").read_text())
                training = json.loads((run / "training.json").read_text())
                assert training["epochs"] == 5 and training["examples"] == 766100
                assert training["steps"] == 23945
                metrics = json.loads((run / f"{args.split}_metrics.json").read_text())
                assert len(metrics["auroc"]) == 25 and all(
                    x is not None for x in metrics["auroc"]
                )
                assert metrics == complete["results"][args.split]
                results[task, seed, arm] = metrics

    print(f"# {args.split} results\n")
    print(
        "Macro AUROC: mean +/- sample SD across three training seeds within each task panel.\n"
    )
    print("| Task seed | Patient-only | Real retrieval | Random retrieval |")
    print("| --- | ---: | ---: | ---: |")
    for task in TASK_SEEDS:
        cells = []
        for arm in ARMS:
            values = [
                results[task, seed, arm]["macro_auroc"] for seed in TRAINING_SEEDS
            ]
            cells.append(
                f"{statistics.mean(values):.5f} +/- {statistics.stdev(values):.5f}"
            )
        print(f"| {task} | " + " | ".join(cells) + " |")

    print(
        "\nMean +/- sample SD over the nine task/training seed combinations per arm.\n"
    )
    print("| Arm | Macro AUROC | Predictive BCE |")
    print("| --- | ---: | ---: |")
    for arm in ARMS:
        rows = [
            results[task, seed, arm] for task in TASK_SEEDS for seed in TRAINING_SEEDS
        ]
        auc = statistics.mean(r["macro_auroc"] for r in rows)
        bce = statistics.mean(r["bce"] for r in rows)
        auc_sd = statistics.stdev(r["macro_auroc"] for r in rows)
        bce_sd = statistics.stdev(r["bce"] for r in rows)
        print(f"| {arm} | {auc:.6f} +/- {auc_sd:.6f} | {bce:.6f} +/- {bce_sd:.6f} |")
    print()
    for control in ["patient_only", "random"]:
        differences = [
            results[task, seed, "real"]["macro_auroc"]
            - results[task, seed, control]["macro_auroc"]
            for task in TASK_SEEDS
            for seed in TRAINING_SEEDS
        ]
        print(
            f"Real-minus-{control}: {statistics.mean(differences):+.6f}; "
            f"real higher in {sum(d > 0 for d in differences)}/9 paired runs."
        )
    print(
        "\nPer-panel SDs describe training-seed variation; overall SDs combine task-panel and training-seed variation."
    )
    print("These are descriptive SDs, not standard errors or confidence intervals.")


if __name__ == "__main__":
    main()
