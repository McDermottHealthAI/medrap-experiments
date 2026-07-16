#!/bin/bash
# ============================================================
# SLURM array: Task-count sweep
# ------------------------------------------------------------
# Trains the same architecture (RoPE encoder + cross-attention
# medium fusion, k=4) across N ∈ {1, 2, 4, 8, 16, 32}
# multi-task targets to measure how performance scales with the
# number of jointly-learned prediction targets.
#
# N is kept small because task codes are sampled from a long-tailed clinical
# vocabulary: only a handful of codes (e.g. Blood Pressure, Weight, BMI) have
# enough in-window positive volume for a stable per-task AUROC given the
# current anchor-sampling scheme (see check_task_balance.py) -- requesting a
# large N mostly adds low-count, noise-dominated tasks rather than signal.
#
# Array index → N:
#   0 → 1     1 → 2     2 → 4
#   3 → 8     4 → 16    5 → 32
#
# Prerequisites (run once before this array):
#   sbatch scripts/prepare_retrieval.sh
#   sbatch scripts/generate_labels.sh       # all 6 N values
#
# Usage:
#   cd mimic_iv_sweep
#   sbatch scripts/sweep_task_count.sh
#   # run a single index, e.g. N=8:
#   sbatch --array=3 scripts/sweep_task_count.sh
# ============================================================

#SBATCH --job-name=sweep-task-count
#SBATCH --array=0-5
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

NS=(1 2 4 8 16 32)
N=${NS[$IDX]}

REPO_DIR="${SLURM_SUBMIT_DIR}"
VENV="${REPO_DIR}/.venv/bin/activate"  # created by: cd mimic_iv_sweep && uv sync

RETRIEVAL_DB="${REPO_DIR}/data/retrieval_db"
TENSORIZED_DIR="/groups/mm6677_gp/data/MIMIC_MEDS/MEDS_cohort/processed"
LABELS_DIR="${REPO_DIR}/data/tasks/n${N}/tasks"
OUTPUT_DIR="${REPO_DIR}/outputs/task_count/n${N}"

cd "${REPO_DIR}"
# shellcheck source=/dev/null
source "${VENV}"
mkdir -p logs "${OUTPUT_DIR}"

echo "=== Job info ==="
echo "  Array job : ${SLURM_ARRAY_JOB_ID:-local}[${IDX}]"
echo "  Node      : ${SLURMD_NODENAME:-$(hostname)}"
echo "  GPU(s)    : ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "  N (tasks) : ${N}"
echo "  Started   : $(date)"
echo ""

medrap-train \
    encoder=rope \
    query_projector=sequence_mean_1024 \
    query_projector.in_dim=128 \
    retriever=hf_dataset \
    "retriever.dataset_path=${RETRIEVAL_DB}" \
    retriever.doc_ids_column=null \
    retriever.k=4 \
    retrieval_encoder=token_feature \
    retrieval_encoder.vocab_size=151936 \
    retrieval_encoder.embedding_dim=64 \
    fusion=cross_attention_medium \
    pooling=masked_mean \
    head=linear \
    head.in_dim=256 \
    "head.out_dim=${N}" \
    training/task=multitask_binary \
    "training.task.num_tasks=${N}" \
    training/loss=multitask_binary_bce \
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
    "wandb_run_name=sweep-task-count-n${N}-${SLURM_ARRAY_JOB_ID:-local}_${IDX}" \
    "output_dir=${OUTPUT_DIR}" \
    do_overwrite=true \
    "$@"

echo ""
echo "=== Done: $(date) ==="
