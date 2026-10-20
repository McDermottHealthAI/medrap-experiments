#!/bin/bash
# Patient-specific retrieval pilot: the retrieval_qwen3 arm on the knowledge-control slice
# (CONTROL_DIMS x CONTROL_N_PATIENTS at CONTROL_NUM_TASKS), so it pairs with the existing
# patient_only / retrieval / random-docs / null-docs runs there. Same pinned MedRAP and
# training setup as those runs. Skips draws that already have test metrics.
#
# Usage:  cd mimic_iv_scaling_laws && scripts/submit_qwen3_pilot.sh

set -euo pipefail
source scripts/grid.sh
ARM=retrieval_qwen3
T=${CONTROL_NUM_TASKS}
for DIM in "${CONTROL_DIMS[@]}"; do
    for N in "${CONTROL_N_PATIENTS[@]}"; do
        missing=()
        for IDX in 0 1 2 3 4; do
            compgen -G "outputs/${ARM}_dim${DIM}_trainN${N}_tasks${T}_test/draw$((IDX + 1))/loggers/csv/*/metrics.csv" >/dev/null \
                || missing+=("${IDX}")
        done
        [ "${#missing[@]}" -eq 0 ] && continue
        # A frozen Qwen3 forward pass per patient makes these much slower than learned-query runs.
        sbatch --array="$(IFS=,; echo "${missing[*]}")" --time=24:00:00 --mem=64G \
            --job-name="sl-${ARM}-dim${DIM}-trainN${N}-tasks${T}" scripts/train.sh "${ARM}" "${DIM}" "${N}" "${T}"
    done
done
