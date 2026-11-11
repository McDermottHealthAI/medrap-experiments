#!/bin/bash
# Stage-2 grid: V2_ARMS x V2_CAPACITIES x V2_N_PATIENTS at V2_NUM_TASKS (grid.sh), 5 draws each,
# all with the head bias initialized from the training base rates. Draws that already have
# test metrics in outputs_v2/ (e.g. from the pilot) are skipped.
#
# Usage:  cd mimic_iv_scaling_laws && scripts/submit_v2_grid.sh          # main grid, V2_NUM_TASKS
#         cd mimic_iv_scaling_laws && scripts/submit_v2_grid.sh tasks    # task-count axis

set -euo pipefail
source scripts/grid.sh
export SL_VENV="${PWD}/v2env/.venv" SL_OUTPUT_ROOT=outputs_v2 SL_BIAS_INIT=1
if [ "${1:-}" = tasks ]; then
    TASK_COUNTS=("${V2_TASK_COUNTS[@]}") CAPACITIES=("${V2_TASK_CAPACITIES[@]}")
else
    TASK_COUNTS=("${V2_NUM_TASKS}") CAPACITIES=("${V2_CAPACITIES[@]}")
fi
for T in "${TASK_COUNTS[@]}"; do
for ARM in "${V2_ARMS[@]}"; do
    for CAP in "${CAPACITIES[@]}"; do
        for N in "${V2_N_PATIENTS[@]}"; do
            missing=()
            for IDX in 0 1 2 3 4; do
                compgen -G "${SL_OUTPUT_ROOT}/${ARM}_dim${CAP}_trainN${N}_tasks${T}_test/draw$((IDX + 1))/loggers/csv/*/metrics.csv" \
                    >/dev/null || missing+=("${IDX}")
            done
            [ "${#missing[@]}" -eq 0 ] && continue
            extra=()
            case "${ARM}" in
                retrieval_qwen3) extra=(--time=24:00:00 --mem=64G) ;;  # frozen Qwen3 forward per patient
                retrieval*) extra=(--time=12:00:00) ;;
            esac
            sbatch "${extra[@]}" --array="$(IFS=,; echo "${missing[*]}")" \
                --job-name="sl-v2-${ARM}-dim${CAP}-trainN${N}-tasks${T}" scripts/train.sh "${ARM}" "${CAP}" "${N}" "${T}"
        done
    done
done
done
