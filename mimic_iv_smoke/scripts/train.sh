#!/usr/bin/env bash
# ============================================================
# SLURM: Multi-task binary code prediction
#        TimeDeltaRoPE encoder + CrossAttentionFusion (medium).
# ------------------------------------------------------------
# Smoke test of mimic_iv/scripts/train_multitask_rope_cross_attention_slurm.sh
# against the post-refactor MedRAP CLI. Same architecture and task family,
# but:
#   - `medrap-train` (flat CLI), not `medrap train`.
#   - reuses the existing lab-shared tensorized cohort at the same path as
#     mimic_iv/ — no need to re-run medrap-preprocess or MTD_preprocess.
#   - task labels are reused as-is from ../mimic_iv (label prep is
#     untouched by the MedRAP refactor, no need to regenerate).
#   - no GCS upload tail (not carrying that forward without confirming
#     bucket access is still current).
#
# Predicts N=25 binary code-occurrence tasks simultaneously
# ("will code X appear within 7 days?") using the full RAP
# pipeline with retrieval active throughout.
#
# Architecture:
#   encoder         : rope  (D=128)
#   retrieval_enc   : token_feature  (D_mem=64)
#   fusion          : cross_attention_medium
#                       d_model=256, num_heads=8, ff_dim=512, layers=2
#   pooling         : masked_mean
#   head            : linear 256 -> N (one logit per task)
#   task/loss       : multitask_binary / multitask_binary_bce
#
# Prerequisites (run once before this script):
#   sbatch scripts/prepare_retrieval.sh
#   sbatch ../mimic_iv/scripts/prepare_multi_task_labels_slurm.sh   (if not already run)
#
# Usage:
#   sbatch scripts/train.sh
#   sbatch scripts/train.sh training.trainer.max_epochs=20
#
# Extra arguments are forwarded to `medrap-train` as Hydra overrides.
# ============================================================

#SBATCH --job-name=medrap-smoke-train
#SBATCH --partition=gpu
#SBATCH --account=mm6677_gp
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR}"
VENV="${REPO_DIR}/.venv/bin/activate"

RETRIEVAL_DB="${REPO_DIR}/data/retrieval_db"
TENSORIZED_DIR="/groups/mm6677_gp/data/MIMIC_MEDS/MEDS_cohort/processed"
MT_LABELS_DIR="${MT_LABELS_DIR:-${REPO_DIR}/../mimic_iv/data/mt_labels/top25_7d}"
NUM_TASKS=25

OUTPUT_DIR="${REPO_DIR}/outputs/mt_rope_cross_attention"

echo "=== Job info ==="
echo "  Job ID   : ${SLURM_JOB_ID:-local}"
echo "  Node     : ${SLURMD_NODENAME:-$(hostname)}"
echo "  GPU(s)   : ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "  Started  : $(date)"
echo ""

cd "${REPO_DIR}"
# shellcheck source=/dev/null
source "${VENV}"

mkdir -p logs "${OUTPUT_DIR}"

echo "=== Starting medrap-train ==="

medrap-train \
    encoder=rope \
    query_projector=sequence_mean_1024 \
    query_projector.in_dim=128 \
    retriever=hf_dataset \
    "retriever.dataset_path=${RETRIEVAL_DB}" \
    retriever.doc_ids_column=null \
    retriever.k=8 \
    retrieval_encoder=token_feature \
    retrieval_encoder.vocab_size=151936 \
    retrieval_encoder.embedding_dim=64 \
    fusion=cross_attention_medium \
    pooling=masked_mean \
    head=linear \
    head.in_dim=256 \
    "head.out_dim=${NUM_TASKS}" \
    training/task=multitask_binary \
    "training.task.num_tasks=${NUM_TASKS}" \
    training/loss=multitask_binary_bce \
    training/datamodule=meds_multitask \
    "training.datamodule.config.tensorized_cohort_dir=${TENSORIZED_DIR}" \
    training.datamodule.config.max_seq_len=256 \
    "training.datamodule.config.task_labels_dir=${MT_LABELS_DIR}" \
    "training.datamodule.mt_labels_dir=${MT_LABELS_DIR}" \
    "training.datamodule.num_tasks=${NUM_TASKS}" \
    training.datamodule.batch_size=32 \
    training.datamodule.config.seq_sampling_strategy=to_end \
    training/trainer=lightning_wandb \
    training.trainer.max_epochs=1 \
    training.trainer.accelerator=gpu \
    training.trainer.devices=1 \
    training.trainer.gradient_clip_val=1.0 \
    training.trainer.log_every_n_steps=10 \
    training.module.lr=1e-4 \
    training.module.warmup_steps=200 \
    "wandb_run_name=mimic-iv-smoke-mt25-rope-cross-attn-${SLURM_JOB_ID:-local}" \
    "output_dir=${OUTPUT_DIR}" \
    do_overwrite=true \
    "$@"

echo ""
echo "=== Done: $(date) ==="
