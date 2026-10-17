#!/bin/bash
# Submit one train.sh array (5 draws) per cell: the main grid (every arm x dim x n_patients at
# MAIN_NUM_TASKS) plus the task-count ablation (every arm x n_patients at ABLATION_DIM for each
# ABLATION_TASK_COUNTS).
#
# Usage:
#   cd mimic_iv_scaling_laws && scripts/submit_sweep.sh
#   # or queued behind the task jobs, using the line submit_tasks.sh prints last:
#   scripts/submit_sweep.sh afterok:<ids>

set -euo pipefail
source scripts/grid.sh

DEPENDENCY=()
if [ -n "${1:-}" ]; then
    DEPENDENCY=(--dependency="$1")
fi

submit() {  # ARM DIM N T -- only the draws that don't have test metrics yet, so reruns are safe
    local missing=()
    for IDX in 0 1 2 3 4; do
        compgen -G "outputs/$1_dim$2_trainN$3_tasks$4_test/draw$((IDX + 1))/loggers/csv/*/metrics.csv" >/dev/null \
            || missing+=("${IDX}")
    done
    [ "${#missing[@]}" -eq 0 ] && return
    sbatch "${DEPENDENCY[@]}" --array="$(IFS=,; echo "${missing[*]}")" \
        --job-name="sl-$1-dim$2-trainN$3-tasks$4" scripts/train.sh "$1" "$2" "$3" "$4"
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
