#!/bin/bash
# ============================================================
# Task-count ablation: held-out TEST split AUROC/loss for patient_only,
# N=128 tasks.
# ============================================================

#SBATCH --job-name=eval-test-patient-only-task-count-n128
#SBATCH --partition=gpu
#SBATCH --account=mm6677_gp
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

N=128

REPO_DIR="${SLURM_SUBMIT_DIR}"
VENV="${REPO_DIR}/.venv/bin/activate"  # created by: cd mimic_iv_sweep && uv sync

TENSORIZED_DIR="/groups/mm6677_gp/zzw2102/data/MIMIC_MEDS_restored/processed"
LABELS_DIR="${REPO_DIR}/data/tasks/n128/tasks"
TRAIN_OUTPUT_DIR="${REPO_DIR}/outputs/patient_only_task_count_ablation_n128"
CHECKPOINT_PATH="${TRAIN_OUTPUT_DIR}/checkpoints/last.ckpt"
EVAL_OUTPUT_DIR="${REPO_DIR}/outputs/patient_only_task_count_ablation_n128_test_eval"

cd "${REPO_DIR}"
# shellcheck source=/dev/null
source "${VENV}"
mkdir -p logs "${EVAL_OUTPUT_DIR}"

if [ ! -f "${CHECKPOINT_PATH}" ]; then
    echo "ERROR: ${CHECKPOINT_PATH} not found." >&2
    exit 1
fi

echo "=== Job info ==="
echo "  eval_mode     : test (MEDS held_out split), N=128 tasks"
echo "  Started       : $(date)"
echo ""

medrap-eval \
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
    training.trainer.accelerator=gpu \
    training.trainer.devices=1 \
    training.module.validation_auroc_log_per_task=true \
    fusion=passthrough \
    query_projector.in_dim=4 \
    head.in_dim=4 \
    training/loss=multitask_binary_bce \
    "checkpoint_path=${CHECKPOINT_PATH}" \
    eval_mode=test \
    "output_dir=${EVAL_OUTPUT_DIR}" \
    do_overwrite=true \
    "wandb_run_name=eval-test-patient-only-task-count-n128-extreme-starved-${SLURM_JOB_ID:-local}" \
    "$@"

echo ""
echo "=== Done: $(date) ==="
