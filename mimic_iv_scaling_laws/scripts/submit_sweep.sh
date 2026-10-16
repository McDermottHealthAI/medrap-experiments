#!/bin/bash
# Submit one train.sh array (5 draws) per cell: the main grid (every arm x dim x n_patients at
# MAIN_NUM_TASKS) plus the task-count ablation (every arm x n_patients at ABLATION_DIM for each
# ABLATION_TASK_COUNTS).
#
# Usage:
#   cd mimic_iv_scaling_laws && scripts/submit_sweep.sh
#   # or queued behind the task jobs, using the line submit_tasks.sh prints last:
#   SBATCH_DEPENDENCY=afterok:<ids> scripts/submit_sweep.sh

set -euo pipefail
source scripts/grid.sh

submit() {  # ARM DIM N T
    sbatch --job-name="sl-$1-dim$2-trainN$3-tasks$4" scripts/train.sh "$1" "$2" "$3" "$4"
}

for ARM in "${ARMS[@]}"; do
    for N in "${N_PATIENTS[@]}"; do
        for DIM in "${DIMS[@]}"; do
            submit "${ARM}" "${DIM}" "${N}" "${MAIN_NUM_TASKS}"
        done
        for T in "${ABLATION_TASK_COUNTS[@]}"; do
            submit "${ARM}" "${ABLATION_DIM}" "${N}" "${T}"
        done
    done
done
