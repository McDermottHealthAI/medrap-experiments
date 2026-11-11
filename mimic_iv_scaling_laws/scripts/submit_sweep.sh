#!/bin/bash
# Submit one train.sh array (5 draws) per cell in grid.sh: the main grid (ARMS x DIMS x
# N_PATIENTS at each MAIN_TASK_COUNTS), the task-count ablation (ARMS x N_PATIENTS at
# ABLATION_DIM for each ABLATION_TASK_COUNTS) and the knowledge-vs-capacity control
# (CONTROL_ARMS x CONTROL_DIMS x CONTROL_N_PATIENTS at CONTROL_NUM_TASKS). Draws that already
# have test metrics are skipped, so rerunning only fills in what is missing.
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

submit() {  # ARM DIM N T -- only the draws that don't have test metrics yet
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
        for T in "${MAIN_TASK_COUNTS[@]}"; do
            for DIM in "${DIMS[@]}"; do
                submit "${ARM}" "${DIM}" "${N}" "${T}"
            done
        done
        for T in "${ABLATION_TASK_COUNTS[@]}"; do
            submit "${ARM}" "${ABLATION_DIM}" "${N}" "${T}"
        done
    done
done

for ARM in "${CONTROL_ARMS[@]}"; do
    for DIM in "${CONTROL_DIMS[@]}"; do
        for N in "${CONTROL_N_PATIENTS[@]}"; do
            submit "${ARM}" "${DIM}" "${N}" "${CONTROL_NUM_TASKS}"
        done
    done
done
