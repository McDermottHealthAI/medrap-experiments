#!/bin/bash
# ============================================================
# SLURM array: tuning + held_out predicted probabilities from a trained cell's last.ckpt
# (one array task per draw), for analyses that need per-patient predictions -- e.g.
# scripts/bias_check.py. Rebuilds the model from scripts/model_args.sh, exactly as train.sh.
#
# Args: ARM DIM N_TRAIN NUM_TASKS   (same as train.sh)
# Output: outputs/predictions/<arm>_dim<D>_trainN<N>_tasks<T>/draw<d>/probabilities.parquet
#
# Usage:
#   cd mimic_iv_scaling_laws
#   sbatch --job-name=sl-pred-patient_only-dim4-trainN1000-tasks25 scripts/predict_probabilities.sh patient_only 4 1000 25
# ============================================================

#SBATCH --array=0-4
#SBATCH --partition=gpu
#SBATCH --account=mm6677_gp
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

ARM=$1
DIM=$2
N_TRAIN=$3
NUM_TASKS=$4

IDX=${SLURM_ARRAY_TASK_ID}
DRAW=$((IDX + 1))

REPO_DIR="${SLURM_SUBMIT_DIR}"
RETRIEVAL_DB="${REPO_DIR}/data/retrieval_db"
TENSORIZED_DIR="${REPO_DIR}/data/cohort/tensorized"
LABELS_DIR="${REPO_DIR}/data/tasks_n${NUM_TASKS}_30d_trainN${N_TRAIN}/draw${DRAW}/tasks"
RUN_NAME="${ARM}_dim${DIM}_trainN${N_TRAIN}_tasks${NUM_TASKS}"
CHECKPOINT="${REPO_DIR}/outputs/${RUN_NAME}/draw${DRAW}/checkpoints/last.ckpt"
OUTPUT_DIR="${REPO_DIR}/outputs/predictions/${RUN_NAME}/draw${DRAW}"

cd "${REPO_DIR}"
# shellcheck source=/dev/null
source "${REPO_DIR}/.venv/bin/activate"
mkdir -p logs "${OUTPUT_DIR}"
[ -f "${CHECKPOINT}" ] || { echo "ERROR: ${CHECKPOINT} not found." >&2; exit 1; }

# shellcheck source=scripts/model_args.sh
source "${REPO_DIR}/scripts/model_args.sh"

medrap-predict-probabilities \
    "${MODEL_ARGS[@]}" \
    training.trainer.accelerator=gpu \
    training.trainer.devices=1 \
    "checkpoint_path=${CHECKPOINT}" \
    "index_dataframe_dir=${LABELS_DIR}" \
    "splits=[tuning,held_out]" \
    "output_dir=${OUTPUT_DIR}" \
    do_overwrite=true
