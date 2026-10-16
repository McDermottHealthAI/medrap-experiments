#!/bin/bash
# Build the cohort, then (once it succeeds) generate every task count in grid.sh.
# The last line printed is the afterok dependency for submit_sweep.sh.
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

TASK_JOBS=()
for T in "${TASK_COUNTS[@]}"; do
    JOB=$(sbatch --parsable "${DEPENDENCY[@]}" --job-name="sl-tasks-n${T}" scripts/generate_tasks.sh "${T}")
    echo "generate_tasks n${T}: ${JOB}"
    TASK_JOBS+=("${JOB}")
done

echo "SBATCH_DEPENDENCY=afterok:$(IFS=:; echo "${TASK_JOBS[*]}")"
