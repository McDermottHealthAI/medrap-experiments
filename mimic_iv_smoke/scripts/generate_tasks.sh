#!/bin/bash
# ============================================================
# SLURM: Generate multi-task binary code-occurrence labels
#        from the raw MEDS cohort via `medrap-preprocess`.
# ------------------------------------------------------------
# Smoke test of the new `medrap.preprocess.task_generation` stage
# (mimic_iv/ predates this — it derived labels via the standalone
# scripts/prepare_multi_task_labels.py instead). Points `medrap-preprocess`
# at the already-tensorized lab-shared cohort (tensorized_dir) so it skips
# MEDS-transforms + MTD_preprocess and only runs task-label generation.
#
# Outputs to ${OUTPUT_DIR}/tasks/{split}.parquet, code_index.json,
# metadata.json — consumed by scripts/train.sh as MT_LABELS_DIR.
#
# Usage:
#   sbatch scripts/generate_tasks.sh
#   sbatch scripts/generate_tasks.sh num_tasks=50
# ============================================================

#SBATCH --job-name=medrap-smoke-gen-tasks
#SBATCH --partition=cpu
#SBATCH --account=mm6677_gp
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR}"
VENV="${REPO_DIR}/.venv/bin/activate"  # created by: cd mimic_iv_smoke && uv sync

MEDS_DATA_DIR="/groups/mm6677_gp/data/MIMIC_MEDS/MEDS_cohort"
TENSORIZED_DIR="/groups/mm6677_gp/data/MIMIC_MEDS/MEDS_cohort/processed"
OUTPUT_DIR="${REPO_DIR}/data/tasks_gen"

cd "${REPO_DIR}"
# shellcheck source=/dev/null
source "${VENV}"
mkdir -p logs

echo "=== Job info ==="
echo "  Job ID   : ${SLURM_JOB_ID:-local}"
echo "  Node     : ${SLURMD_NODENAME:-$(hostname)}"
echo "  Started  : $(date)"
echo ""

medrap-preprocess \
    "meds_data_dir=${MEDS_DATA_DIR}" \
    "tensorized_dir=${TENSORIZED_DIR}" \
    "output_dir=${OUTPUT_DIR}" \
    num_tasks=25 \
    horizon_days=7.0 \
    min_history_days=1.0 \
    seed=42 \
    do_overwrite=true \
    "$@"

echo ""
echo "=== Done: $(date) ==="
echo "Task labels saved to ${OUTPUT_DIR}/tasks"
