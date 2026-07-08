#!/bin/bash
# ============================================================
# SLURM array: Generate multi-task binary code-occurrence labels
#              for each task-count N in the sweep.
# ------------------------------------------------------------
# Submits one job per N value {10, 25, 50, 100, 250, 500}.
# Each job calls `medrap-preprocess` with the already-tensorized
# lab cohort (tensorized_dir), so only the task-generation stage
# runs (MEDS-transforms and MTD_preprocess are skipped).
#
# Outputs for index i land in:
#   data/tasks/n<N>/tasks/{train,tuning,held_out}.parquet
#
# Usage:
#   cd mimic_iv_sweep
#   sbatch scripts/generate_labels.sh
# ============================================================

#SBATCH --job-name=sweep-gen-labels
#SBATCH --array=0-5
#SBATCH --partition=cpu
#SBATCH --account=mm6677_gp
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

IDX=${SLURM_ARRAY_TASK_ID}

NS=(10 25 50 100 250 500)
N=${NS[$IDX]}

REPO_DIR="${SLURM_SUBMIT_DIR}"
VENV="${REPO_DIR}/.venv/bin/activate"  # created by: cd mimic_iv_sweep && uv sync

MEDS_DATA_DIR="/groups/mm6677_gp/data/MIMIC_MEDS/MEDS_cohort"
TENSORIZED_DIR="/groups/mm6677_gp/data/MIMIC_MEDS/MEDS_cohort/processed"
OUTPUT_DIR="${REPO_DIR}/data/tasks/n${N}"

cd "${REPO_DIR}"
# shellcheck source=/dev/null
source "${VENV}"
mkdir -p logs "${OUTPUT_DIR}"

echo "=== Job info ==="
echo "  Array job : ${SLURM_ARRAY_JOB_ID:-local}[${IDX}]"
echo "  Node      : ${SLURMD_NODENAME:-$(hostname)}"
echo "  N (tasks) : ${N}"
echo "  Output    : ${OUTPUT_DIR}"
echo "  Started   : $(date)"
echo ""

medrap-preprocess \
    "meds_data_dir=${MEDS_DATA_DIR}" \
    "tensorized_dir=${TENSORIZED_DIR}" \
    "output_dir=${OUTPUT_DIR}" \
    "num_tasks=${N}" \
    horizon_days=7.0 \
    min_history_days=1.0 \
    seed=42 \
    do_overwrite=true \
    "$@"

echo ""
echo "=== Done: $(date) ==="
echo "Labels saved to ${OUTPUT_DIR}/tasks"
