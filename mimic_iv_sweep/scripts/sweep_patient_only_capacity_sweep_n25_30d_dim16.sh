#!/bin/bash
# ============================================================
# SLURM array: patient_only training, CLEAN capacity-only cut
# (embedding_dim=16, ff_dim=32, num_heads=1, num_layers=1) with
# max_seq_len HELD FIXED at 256 (full, unlike the old capacity_starved /
# extreme_capacity_starved lines which also cut context length). Full
# training data. Part of a 5-point N-only sweep (embedding_dim
# 4/8/16/32/64), reusing the existing full-capacity (128) baseline as the
# top anchor.
# ------------------------------------------------------------
# Uses the restored tensorized cohort (original deleted ~2026-08-25).
#
# 5 array tasks = 5 draws.
#
# Usage:
#   cd mimic_iv_sweep
#   sbatch scripts/sweep_patient_only_capacity_sweep_n25_30d_dim16.sh
# ============================================================

#SBATCH --job-name=sweep-patient-only-capacity-sweep-dim16
#SBATCH --array=0-4
#SBATCH --partition=gpu
#SBATCH --account=mm6677_gp
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

IDX=${SLURM_ARRAY_TASK_ID}
DRAW=$((IDX + 1))
N=25

REPO_DIR="${SLURM_SUBMIT_DIR}"
VENV="${REPO_DIR}/.venv/bin/activate"  # created by: cd mimic_iv_sweep && uv sync

TENSORIZED_DIR="/groups/mm6677_gp/zzw2102/data/MIMIC_MEDS_restored/processed"
LABELS_DIR="${REPO_DIR}/data/tasks_zach_uniform_event_n25_30d/draw${DRAW}/tasks"
OUTPUT_DIR="${REPO_DIR}/outputs/patient_only_capacity_sweep_n25_30d_dim16/draw${DRAW}"

cd "${REPO_DIR}"
# shellcheck source=/dev/null
source "${VENV}"
mkdir -p logs "${OUTPUT_DIR}"

if [ ! -d "${LABELS_DIR}" ]; then
    echo "ERROR: ${LABELS_DIR} not found." >&2
    exit 1
fi

echo "=== Job info ==="
echo "  Draw          : ${DRAW}/5"
echo "  Architecture  : patient_only, embedding_dim=16, max_seq_len=256 (fixed)"
echo "  Started       : $(date)"
echo ""

medrap-train \
    encoder=rope \
    encoder.embedding_dim=16 \
    encoder.num_heads=1 \
    encoder.num_layers=1 \
    encoder.ff_dim=32 \
    pooling=masked_mean \
    head=linear \
    "head.out_dim=${N}" \
    training/task=multitask_binary \
    "training.task.num_tasks=${N}" \
    training/datamodule=meds_multitask \
    "training.datamodule.config.tensorized_cohort_dir=${TENSORIZED_DIR}" \
    training.datamodule.config.max_seq_len=256 \
    "training.datamodule.config.task_labels_dir=${LABELS_DIR}" \
    "training.datamodule.mt_labels_dir=${LABELS_DIR}" \
    "training.datamodule.num_tasks=${N}" \
    training.datamodule.batch_size=32 \
    training.datamodule.config.seq_sampling_strategy=to_end \
    training/trainer=lightning_wandb \
    training.trainer.max_epochs=3 \
    training.trainer.accelerator=gpu \
    training.trainer.devices=1 \
    training.trainer.gradient_clip_val=1.0 \
    training.trainer.log_every_n_steps=10 \
    training.module.lr=1e-3 \
    training.module.warmup_steps=200 \
    training.module.validation_auroc_log_per_task=true \
    fusion=passthrough \
    query_projector.in_dim=16 \
    head.in_dim=16 \
    training/loss=multitask_binary_bce \
    "output_dir=${OUTPUT_DIR}" \
    do_overwrite=true \
    "wandb_run_name=patient-only-capacity-sweep-dim16-d30-draw${DRAW}-${SLURM_ARRAY_JOB_ID:-local}_${IDX}" \
    "$@"

echo ""
echo "=== Done: $(date) ==="
