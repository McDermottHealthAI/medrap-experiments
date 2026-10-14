#!/bin/bash
# Submit one train.sh array (5 draws) per (arm, dim, n_patients) cell in grid.sh.
#
# Usage:
#   cd mimic_iv_scaling_laws && scripts/submit_sweep.sh

set -euo pipefail
source scripts/grid.sh

for ARM in "${ARMS[@]}"; do
    for DIM in "${DIMS[@]}"; do
        for N in "${N_PATIENTS[@]}"; do
            sbatch --job-name="sl-${ARM}-dim${DIM}-trainN${N}" scripts/train.sh "${ARM}" "${DIM}" "${N}"
        done
    done
done
