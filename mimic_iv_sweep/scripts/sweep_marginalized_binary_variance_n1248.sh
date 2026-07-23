#!/bin/bash
# ============================================================
# SLURM array: cross-attention marginalized retrieval
#              (marginalized_output_mode=binary) for the variance study --
#              N in {1, 2, 4, 8, 16, 32, 64, 128}, EACH REPEATED FOR 5
#              INDEPENDENT RANDOM DRAWS of the N task codes (see
#              generate_labels_variance_n1248.sh). Pairs with
#              sweep_patient_only_variance_n1248.sh -- both train on the
#              exact same per-draw label file, so their AUROC difference is
#              a paired comparison within each draw.
# ------------------------------------------------------------
# 40 array tasks = 8 N values x 5 draws, flattened as
# IDX = n_idx * 5 + draw_idx (n_idx in [0,7], draw_idx in [0,4]) -- same
# flattening as generate_labels_variance_n1248.sh /
# sweep_patient_only_variance_n1248.sh.
#
# Architecture/hyperparameters match sweep_marginalized_binary_n1248.sh
# exactly (k=4, marginalized_output_mode=binary per MedRAP#93); only
# LABELS_DIR/OUTPUT_DIR/wandb_run_name vary per (N, draw).
#
# Array index -> (N, draw): same table as sweep_patient_only_variance_n1248.sh
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
#   sbatch scripts/prepare_retrieval.sh
#   sbatch scripts/generate_labels_variance_n1248.sh
#
# Usage:
#   cd mimic_iv_sweep
#   sbatch scripts/sweep_marginalized_binary_variance_n1248.sh
#   sbatch --array=0-4 scripts/sweep_marginalized_binary_variance_n1248.sh   # only N=1, all 5 draws
# ============================================================

#SBATCH --job-name=sweep-marginalized-binary-variance-n1248
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

RETRIEVAL_DB="${REPO_DIR}/data/retrieval_db"
TENSORIZED_DIR="/groups/mm6677_gp/data/MIMIC_MEDS/MEDS_cohort/processed"
LABELS_DIR="${REPO_DIR}/data/tasks_variance/draw${DRAW}/n${N}/tasks"
OUTPUT_DIR="${REPO_DIR}/outputs/marginalized_binary_variance_n1248/draw${DRAW}/n${N}"

cd "${REPO_DIR}"
# shellcheck source=/dev/null
source "${VENV}"
mkdir -p logs "${OUTPUT_DIR}"

if [ ! -d "${RETRIEVAL_DB}" ]; then
    echo "ERROR: ${RETRIEVAL_DB} not found. Run sbatch scripts/prepare_retrieval.sh first." >&2
    exit 1
fi

if [ ! -d "${LABELS_DIR}" ]; then
    echo "ERROR: ${LABELS_DIR} not found. Run sbatch scripts/generate_labels_variance_n1248.sh for N=${N} draw=${DRAW} first." >&2
    exit 1
fi

echo "=== Job info ==="
echo "  Array job              : ${SLURM_ARRAY_JOB_ID:-local}[${IDX}]"
echo "  Node                   : ${SLURMD_NODENAME:-$(hostname)}"
echo "  GPU(s)                 : ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "  N (tasks)              : ${N}"
echo "  Draw                   : ${DRAW}/5"
echo "  marginalized_output_mode: binary (MedRAP#93)"
echo "  Started                : $(date)"
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
    "wandb_run_name=marginalized-binary-variance-draw${DRAW}-n${N}-${SLURM_ARRAY_JOB_ID:-local}_${IDX}" \
    "$@"

echo ""
echo "=== Done: $(date) ==="
