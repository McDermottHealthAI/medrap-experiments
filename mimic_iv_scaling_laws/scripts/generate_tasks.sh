#!/bin/bash
# ============================================================
# SLURM array: generate NUM_TASKS random 30d code-occurrence tasks for each of
# the 5 draws from this experiment's own cohort (scripts/build_cohort.sh), then
# write the train-subsampled label dirs for every N_PATIENTS in grid.sh.
#
#   data/tasks_n<T>_30d/draw<d>/tasks                 full-size labels
#   data/tasks_n<T>_30d_trainN<N>/draw<d>/tasks       train.parquet subsampled to N patients,
#                                                     tuning/held_out copied unchanged
#
# Prediction times depend only on the draw seed, not on NUM_TASKS, so every task
# count in a draw labels the same patients at the same times, and the subsample
# seed (1000000 * draw + N) picks the same N training patients for every T.
#
# Args:
#   NUM_TASKS  number of random task codes per draw
#
# Usage:
#   cd mimic_iv_scaling_laws
#   sbatch --job-name=sl-tasks-n25 scripts/generate_tasks.sh 25
#   (or scripts/submit_tasks.sh for every TASK_COUNTS level in grid.sh)
# ============================================================

#SBATCH --array=0-4
#SBATCH --partition=cpu
#SBATCH --account=mm6677_gp
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

NUM_TASKS=$1

REPO_DIR="${SLURM_SUBMIT_DIR}"
cd "${REPO_DIR}"
source scripts/grid.sh

IDX=${SLURM_ARRAY_TASK_ID}
DRAW=$((IDX + 1))
SEED=${DRAW_SEEDS[$IDX]}

VENV="${REPO_DIR}/.venv/bin/activate"  # created by: cd mimic_iv_scaling_laws && uv sync
COHORT_DIR="${REPO_DIR}/data/cohort"
OUTPUT_DIR="${REPO_DIR}/data/tasks_n${NUM_TASKS}_30d/draw${DRAW}"

# shellcheck source=/dev/null
source "${VENV}"
mkdir -p logs "${OUTPUT_DIR}"

if [ ! -d "${COHORT_DIR}/tensorized" ]; then
    echo "ERROR: ${COHORT_DIR}/tensorized not found. Run scripts/build_cohort.sh first." >&2
    exit 1
fi

echo "=== Job info ==="
echo "  Array job : ${SLURM_ARRAY_JOB_ID:-local}[${IDX}]"
echo "  Node      : ${SLURMD_NODENAME:-$(hostname)}"
echo "  NUM_TASKS : ${NUM_TASKS}"
echo "  Draw      : ${DRAW}/5 (seed=${SEED})"
echo "  Output    : ${OUTPUT_DIR}"
echo "  Started   : $(date)"
echo ""

medrap-preprocess \
    "meds_data_dir=${COHORT_DIR}/intermediate" \
    "tensorized_dir=${COHORT_DIR}/tensorized" \
    "output_dir=${OUTPUT_DIR}" \
    "num_tasks=${NUM_TASKS}" \
    horizon_days=30.0 \
    min_history_days=1.0 \
    "seed=${SEED}" \
    code_selection=random \
    anchor_strategy=uniform_event \
    do_overwrite=true

for N in "${N_PATIENTS[@]}"; do
    python scripts/subsample_train_labels.py \
        "${OUTPUT_DIR}/tasks" \
        "${REPO_DIR}/data/tasks_n${NUM_TASKS}_30d_trainN${N}/draw${DRAW}/tasks" \
        --n-rows "${N}" \
        --seed "$((1000000 * DRAW + N))"
done

echo ""
echo "=== Done: $(date) ==="
