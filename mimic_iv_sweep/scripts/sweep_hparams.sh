#!/bin/bash
# ============================================================
# SLURM array: Hyperparameter sweep
# ------------------------------------------------------------
# Explores k (retrieval neighbours) × lr (learning rate) at
# the best mid-scale configuration (N=25, RoPE + cross-attention
# medium, 3 epochs).
#
# Grid: k ∈ {4, 8, 16, 32} × lr ∈ {1e-4, 1e-3, 3e-3} → 12 jobs
#
# Array index → (k, lr):
#   0  → k=4,  lr=1e-4     1  → k=4,  lr=1e-3     2  → k=4,  lr=3e-3
#   3  → k=8,  lr=1e-4     4  → k=8,  lr=1e-3     5  → k=8,  lr=3e-3
#   6  → k=16, lr=1e-4     7  → k=16, lr=1e-3     8  → k=16, lr=3e-3
#   9  → k=32, lr=1e-4    10  → k=32, lr=1e-3    11  → k=32, lr=3e-3
#
# Prerequisites:
#   sbatch scripts/prepare_retrieval.sh
#   sbatch --array=1 scripts/generate_labels.sh   # N=25 (index 1)
#
# Usage:
#   cd mimic_iv_sweep
#   sbatch scripts/sweep_hparams.sh
#   # single cell, e.g. k=8, lr=1e-3:
#   sbatch --array=4 scripts/sweep_hparams.sh
# ============================================================

#SBATCH --job-name=sweep-hparams
#SBATCH --array=0-11
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

KS=(4 4 4 8 8 8 16 16 16 32 32 32)
LRS=(1e-4 1e-3 3e-3 1e-4 1e-3 3e-3 1e-4 1e-3 3e-3 1e-4 1e-3 3e-3)

K=${KS[$IDX]}
LR=${LRS[$IDX]}
N=25

REPO_DIR="${SLURM_SUBMIT_DIR}"
VENV="${REPO_DIR}/.venv/bin/activate"  # created by: cd mimic_iv_sweep && uv sync

RETRIEVAL_DB="${REPO_DIR}/data/retrieval_db"
TENSORIZED_DIR="/groups/mm6677_gp/data/MIMIC_MEDS/MEDS_cohort/processed"
LABELS_DIR="${REPO_DIR}/data/tasks/n${N}/tasks"
OUTPUT_DIR="${REPO_DIR}/outputs/hparams/k${K}_lr${LR}"

cd "${REPO_DIR}"
# shellcheck source=/dev/null
source "${VENV}"
mkdir -p logs "${OUTPUT_DIR}"

echo "=== Job info ==="
echo "  Array job : ${SLURM_ARRAY_JOB_ID:-local}[${IDX}]"
echo "  Node      : ${SLURMD_NODENAME:-$(hostname)}"
echo "  GPU(s)    : ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "  N (tasks) : ${N}"
echo "  k         : ${K}"
echo "  lr        : ${LR}"
echo "  Started   : $(date)"
echo ""

medrap-train \
    encoder=rope \
    query_projector=sequence_mean_1024 \
    query_projector.in_dim=128 \
    retriever=hf_dataset \
    "retriever.dataset_path=${RETRIEVAL_DB}" \
    retriever.doc_ids_column=null \
    "retriever.k=${K}" \
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
    "training.module.lr=${LR}" \
    training.module.warmup_steps=200 \
    "wandb_run_name=sweep-hparams-k${K}-lr${LR}-${SLURM_ARRAY_JOB_ID:-local}_${IDX}" \
    "output_dir=${OUTPUT_DIR}" \
    do_overwrite=true \
    "$@"

echo ""
echo "=== Done: $(date) ==="
