#!/bin/bash
# ============================================================
# Task-count ablation: patient_only training, EXTREME capacity cut
# (embedding_dim=4, ff_dim=8, max_seq_len=8), N=1 task codes.
# ------------------------------------------------------------
# METHODOLOGY CAVEAT: reuses the OLD label set at data/tasks/n1/tasks/
# (pre-anchor-fix, horizon_days=7.0, single draw, NOT the zach_uniform_event
# 30d methodology used everywhere else in this project) -- the intermediate/
# MEDS directory needed to regenerate labels at the current methodology was
# deleted along with the original tensorized cohort and is unrecoverable
# from any available snapshot. See results/task_count_ablation/README.md
# for the full caveat.
#
# Single job, no array (only 1 draw available for these old label sets).
#
# Usage:
#   cd mimic_iv_sweep
#   sbatch scripts/sweep_patient_only_task_count_ablation_n1.sh
# ============================================================

#SBATCH --job-name=sweep-patient-only-task-count-n1
#SBATCH --partition=gpu
#SBATCH --account=mm6677_gp
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

N=1

REPO_DIR="${SLURM_SUBMIT_DIR}"
VENV="${REPO_DIR}/.venv/bin/activate"  # created by: cd mimic_iv_sweep && uv sync

TENSORIZED_DIR="/groups/mm6677_gp/zzw2102/data/MIMIC_MEDS_restored/processed"
LABELS_DIR="${REPO_DIR}/data/tasks/n1/tasks"
OUTPUT_DIR="${REPO_DIR}/outputs/patient_only_task_count_ablation_n1"

cd "${REPO_DIR}"
# shellcheck source=/dev/null
source "${VENV}"
mkdir -p logs "${OUTPUT_DIR}"

if [ ! -d "${LABELS_DIR}" ]; then
    echo "ERROR: ${LABELS_DIR} not found." >&2
    exit 1
fi

echo "=== Job info ==="
echo "  Architecture  : patient_only, EXTREME capacity cut, N=1 tasks"
echo "  Started       : $(date)"
echo ""

medrap-train \
    encoder=rope \
    encoder.embedding_dim=4 \
    encoder.num_heads=1 \
    encoder.num_layers=1 \
    encoder.ff_dim=8 \
    pooling=masked_mean \
    head=linear \
    "head.out_dim=${N}" \
    training/task=multitask_binary \
    "training.task.num_tasks=${N}" \
    training/datamodule=meds_multitask \
    "training.datamodule.config.tensorized_cohort_dir=${TENSORIZED_DIR}" \
    training.datamodule.config.max_seq_len=8 \
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
    query_projector.in_dim=4 \
    head.in_dim=4 \
    training/loss=multitask_binary_bce \
    "output_dir=${OUTPUT_DIR}" \
    do_overwrite=true \
    "wandb_run_name=patient-only-task-count-n1-extreme-starved-${SLURM_JOB_ID:-local}" \
    "$@"

echo ""
echo "=== Done: $(date) ==="
