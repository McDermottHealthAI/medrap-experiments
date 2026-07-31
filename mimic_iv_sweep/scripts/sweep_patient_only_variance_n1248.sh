#!/bin/bash
# ============================================================
# SLURM array: patient_only training for the variance study -- N in
#              {1, 2, 4, 8, 16, 32, 64, 128}, EACH REPEATED FOR 5 INDEPENDENT
#              RANDOM DRAWS of the N task codes (see
#              generate_labels_variance_n1248.sh). Pairs with
#              sweep_marginalized_binary_variance_n1248.sh -- both train on
#              the exact same per-draw label file, so their AUROC
#              difference is a paired comparison within each draw.
# ------------------------------------------------------------
# 40 array tasks = 8 N values x 5 draws, flattened as
# IDX = n_idx * 5 + draw_idx (n_idx in [0,7], draw_idx in [0,4]) -- same
# flattening as generate_labels_variance_n1248.sh.
#
# Architecture/hyperparameters match sweep_patient_only_n1248.sh exactly;
# only LABELS_DIR/OUTPUT_DIR/wandb_run_name vary per (N, draw).
#
# Validation protocol changed here (rationale:
# ../mimic_iv_sweep_frequent/NULL_RESULT_DIAGNOSIS.md):
# training.trainer.limit_val_batches=1.0 scores the FULL tuning split on
# every validation pass instead of the 200-batch (6,400-row) prefix
# conf/training/trainer/lightning_wandb.yaml defaults to, and
# training.trainer.val_check_interval moves 0.2 -> 0.5 to offset the cost
# (two validation passes per epoch instead of five). That prefix held so few
# positives per task that a single positive changing rank could move the
# task's AUROC by up to ~0.5 -- larger than the across-draw spread this
# study is trying to measure. The change is applied to this patient_only
# baseline as well as the marginalized arm precisely so the paired
# comparison still uses one validation protocol; numbers from this script
# are NOT comparable to results published before this change.
#
# Array index -> (N, draw):
#   0-4   -> N=1,   draws 1-5
#   5-9   -> N=2,   draws 1-5
#   10-14 -> N=4,   draws 1-5
#   15-19 -> N=8,   draws 1-5
#   20-24 -> N=16,  draws 1-5
#   25-29 -> N=32,  draws 1-5
#   30-34 -> N=64,  draws 1-5
#   35-39 -> N=128, draws 1-5
#
# Prerequisites:
#   sbatch scripts/generate_labels_variance_n1248.sh
#
# Usage:
#   cd mimic_iv_sweep
#   sbatch scripts/sweep_patient_only_variance_n1248.sh
#   sbatch --array=0-4 scripts/sweep_patient_only_variance_n1248.sh   # only N=1, all 5 draws
# ============================================================

#SBATCH --job-name=sweep-patient-only-variance-n1248
#SBATCH --array=0-39
#SBATCH --partition=gpu
#SBATCH --account=mm6677_gp
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

IDX=${SLURM_ARRAY_TASK_ID}

NS=(1 2 4 8 16 32 64 128)
N_IDX=$((IDX / 5))
DRAW_IDX=$((IDX % 5))
N=${NS[$N_IDX]}
DRAW=$((DRAW_IDX + 1))

REPO_DIR="${SLURM_SUBMIT_DIR}"
VENV="${REPO_DIR}/.venv/bin/activate"  # created by: cd mimic_iv_sweep && uv sync

TENSORIZED_DIR="/groups/mm6677_gp/data/MIMIC_MEDS/MEDS_cohort/processed"
LABELS_DIR="${REPO_DIR}/data/tasks_variance/draw${DRAW}/n${N}/tasks"
OUTPUT_DIR="${REPO_DIR}/outputs/patient_only_variance_n1248/draw${DRAW}/n${N}"

cd "${REPO_DIR}"
# shellcheck source=/dev/null
source "${VENV}"
mkdir -p logs "${OUTPUT_DIR}"

if [ ! -d "${LABELS_DIR}" ]; then
    echo "ERROR: ${LABELS_DIR} not found. Run sbatch scripts/generate_labels_variance_n1248.sh for N=${N} draw=${DRAW} first." >&2
    exit 1
fi

echo "=== Job info ==="
echo "  Array job     : ${SLURM_ARRAY_JOB_ID:-local}[${IDX}]"
echo "  Node          : ${SLURMD_NODENAME:-$(hostname)}"
echo "  GPU(s)        : ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "  N (tasks)     : ${N}"
echo "  Draw          : ${DRAW}/5"
echo "  Architecture  : patient_only (no retrieval)"
echo "  Started       : $(date)"
echo ""

# Same combo as sweep_architecture.sh's `patient_only` arm -- see that
# script for the full rationale (in particular why query_projector.in_dim=128
# is still required even though fusion=passthrough discards the
# query_projector output: model.forward() calls query_projector(encoder_out)
# unconditionally).
medrap-train \
    encoder=rope \
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
    training.trainer.limit_val_batches=1.0 \
    training.trainer.val_check_interval=0.5 \
    training.module.lr=1e-3 \
    training.module.warmup_steps=200 \
    training.module.validation_auroc_log_per_task=true \
    fusion=passthrough \
    query_projector.in_dim=128 \
    head.in_dim=128 \
    training/loss=multitask_binary_bce \
    "output_dir=${OUTPUT_DIR}" \
    do_overwrite=true \
    "wandb_run_name=patient-only-variance-draw${DRAW}-n${N}-${SLURM_ARRAY_JOB_ID:-local}_${IDX}" \
    "$@"

echo ""
echo "=== Done: $(date) ==="
