#!/bin/bash
# Submit one train.sh array (5 draws) per (arm, dim, n_patients, num_tasks) cell in grid.sh.
#
# Usage:
#   cd mimic_iv_scaling_laws && scripts/submit_sweep.sh

set -euo pipefail
source scripts/grid.sh

for T in "${TASK_COUNTS[@]}"; do
    for ARM in "${ARMS[@]}"; do
        for DIM in "${DIMS[@]}"; do
            for N in "${N_PATIENTS[@]}"; do
                sbatch --job-name="sl-${ARM}-dim${DIM}-trainN${N}-tasks${T}" \
                    scripts/train.sh "${ARM}" "${DIM}" "${N}" "${T}"
            done
        done
    done
done
