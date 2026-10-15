#!/bin/bash
# Build the cohort, then (once it succeeds) generate every TASK_COUNTS level in grid.sh.
#
# Usage:
#   cd mimic_iv_scaling_laws && scripts/submit_tasks.sh
#   scripts/submit_tasks.sh --skip-cohort    # cohort already built; only (re)generate tasks

set -euo pipefail
source scripts/grid.sh
mkdir -p logs

DEPENDENCY=()
if [ "${1:-}" != "--skip-cohort" ]; then
    COHORT_JOB=$(sbatch --parsable scripts/build_cohort.sh)
    echo "build_cohort: ${COHORT_JOB}"
    DEPENDENCY=(--dependency="afterok:${COHORT_JOB}")
fi

for T in "${TASK_COUNTS[@]}"; do
    sbatch "${DEPENDENCY[@]}" --job-name="sl-tasks-n${T}" scripts/generate_tasks.sh "${T}"
done
