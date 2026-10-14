#!/bin/bash
# Submit one train.sh array (5 draws) per (arm, dim, pct) cell in grid.sh.
#
# Usage:
#   cd mimic_iv_scaling_laws && scripts/submit_sweep.sh

set -euo pipefail
source scripts/grid.sh

for ARM in "${ARMS[@]}"; do
    for DIM in "${DIMS[@]}"; do
        for PCT in "${PCTS[@]}"; do
            sbatch --job-name="sl-${ARM}-dim${DIM}-train${PCT}pct" scripts/train.sh "${ARM}" "${DIM}" "${PCT}"
        done
    done
done
