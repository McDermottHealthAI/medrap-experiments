#!/bin/bash
# ============================================================
# SLURM array: cross-attention marginalized retrieval
#              (marginalized_output_mode=binary) for the duration x
#              variance study -- fixed N=25, 5 independent random draws, at
#              7-day and 30-day fixed occurrence-window durations (see
#              generate_labels_duration_variance_n25.sh). Pairs with
#              sweep_patient_only_duration_variance_n25.sh -- both train on
#              the exact same per-(duration, draw) label file.
# ------------------------------------------------------------
# 10 array tasks = 2 durations x 5 draws, flattened as
# IDX = duration_idx * 5 + draw_idx -- same flattening as
# generate_labels_duration_variance_n25.sh /
# sweep_patient_only_duration_variance_n25.sh.
#
# Architecture/hyperparameters match sweep_marginalized_binary_n1248.sh
# exactly (k=4, marginalized_output_mode=binary per MedRAP#93; N fixed at
# 25 here instead of swept); only LABELS_DIR/OUTPUT_DIR/wandb_run_name vary
# per (duration, draw).
#
# Prerequisites:
#   sbatch scripts/prepare_retrieval.sh
#   sbatch scripts/generate_labels_duration_variance_n25.sh
#
# Usage:
#   cd mimic_iv_sweep
#   sbatch scripts/sweep_marginalized_binary_duration_variance_n25.sh
#   sbatch --dependency=aftercorr:<label_gen_jobid> scripts/sweep_marginalized_binary_duration_variance_n25.sh
# ============================================================

#SBATCH --job-name=sweep-marginalized-binary-duration-variance-n25
#SBATCH --array=0-9
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

DURATIONS=(7 30)
DURATION_IDX=$((IDX / 5))
DRAW_IDX=$((IDX % 5))
DURATION_DAYS=${DURATIONS[$DURATION_IDX]}
DRAW=$((DRAW_IDX + 1))
N=25

REPO_DIR="${SLURM_SUBMIT_DIR}"
VENV="${REPO_DIR}/.venv/bin/activate"  # created by: cd mimic_iv_sweep && uv sync

RETRIEVAL_DB="${REPO_DIR}/data/retrieval_db"
TENSORIZED_DIR="/groups/mm6677_gp/data/MIMIC_MEDS/MEDS_cohort/processed"
LABELS_DIR="${REPO_DIR}/data/tasks_duration_variance/d${DURATION_DAYS}/draw${DRAW}/tasks"
OUTPUT_DIR="${REPO_DIR}/outputs/marginalized_binary_duration_variance_n25/d${DURATION_DAYS}/draw${DRAW}"

cd "${REPO_DIR}"
# shellcheck source=/dev/null
source "${VENV}"
mkdir -p logs "${OUTPUT_DIR}"

if [ ! -d "${RETRIEVAL_DB}" ]; then
    echo "ERROR: ${RETRIEVAL_DB} not found. Run sbatch scripts/prepare_retrieval.sh first." >&2
    exit 1
fi

if [ ! -d "${LABELS_DIR}" ]; then
    echo "ERROR: ${LABELS_DIR} not found. Run sbatch scripts/generate_labels_duration_variance_n25.sh for duration=${DURATION_DAYS}d draw=${DRAW} first." >&2
    exit 1
fi

echo "=== Job info ==="
echo "  Array job              : ${SLURM_ARRAY_JOB_ID:-local}[${IDX}]"
echo "  Node                   : ${SLURMD_NODENAME:-$(hostname)}"
echo "  GPU(s)                 : ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "  N (tasks)              : ${N}"
echo "  Duration                : ${DURATION_DAYS}d"
echo "  Draw                    : ${DRAW}/5"
echo "  marginalized_output_mode: binary (MedRAP#93)"
echo "  Started                 : $(date)"
echo ""

# Same combo as sweep_marginalized_binary_n1248.sh -- only LABELS_DIR/
# OUTPUT_DIR/wandb_run_name differ here.
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
    training.module.lr=1e-3 \
    training.module.warmup_steps=200 \
    training.module.validation_auroc_log_per_task=true \
    query_projector=sequence_mean_1024 \
    query_projector.in_dim=128 \
    retriever=hf_dataset \
    "retriever.dataset_path=${RETRIEVAL_DB}" \
    retriever.doc_ids_column=null \
    retriever.k=4 \
    retrieval_encoder=token_feature \
    retrieval_encoder.vocab_size=151936 \
    retrieval_encoder.embedding_dim=64 \
    fusion=cross_attention_perdoc_medium \
    marginalized_retrieval=true \
    marginalized_output_mode=binary \
    head.in_dim=256 \
    training/loss=multitask_binary_bce_marginalized \
    "training.loss.num_tasks=${N}" \
    "output_dir=${OUTPUT_DIR}" \
    do_overwrite=true \
    "wandb_run_name=marginalized-binary-duration-variance-d${DURATION_DAYS}-draw${DRAW}-${SLURM_ARRAY_JOB_ID:-local}_${IDX}" \
    "$@"

echo ""
echo "=== Done: $(date) ==="
