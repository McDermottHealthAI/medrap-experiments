#!/bin/bash
# ============================================================
# SLURM array: cross-attention marginalized retrieval with
#              marginalized_output_mode=binary, trained on RANDOM PER-TASK
#              DURATION labels (generate_labels_duration_n1248_frequent.sh)
#              instead of the fixed 7-day-horizon labels every other script
#              in this directory uses, N in {1, 2, 4, 8, 16, 32, 64, 128}.
# ------------------------------------------------------------
# Identical architecture/hyperparameters to sweep_marginalized_binary_n1248.sh
# -- only the task_labels_dir differs (data/tasks_duration/n<N>/tasks instead
# of data/tasks/n<N>/tasks), so this isolates the effect of random per-task
# occurrence-window durations (log-uniform, [1, 90] days -- see
# generate_labels_duration_n1248_frequent.sh) against the same-else-identical
# fixed-7-day-window baseline already in marginalized-binary-frequent-n{N}-*.
#
# Validation protocol changed here (rationale: NULL_RESULT_DIAGNOSIS.md):
# training.trainer.limit_val_batches=1.0 scores the FULL tuning split on
# every validation pass instead of the 200-batch (6,400-row) prefix
# conf/training/trainer/lightning_wandb.yaml defaults to, and
# training.trainer.val_check_interval moves 0.2 -> 0.5 to offset the cost
# (two validation passes per epoch instead of five). That prefix held so few
# positives per task that a single positive changing rank could move the
# task's AUROC by up to ~0.5, which is the same size as every difference
# this sweep has reported. Numbers from this script are therefore NOT
# comparable to results published before this change.
#
# The marginalized document score is now cosine rather than dot
# (marginalized_score_similarity=cosine), same rationale. The raw dot
# product left ||q|| acting as an implicit inverse temperature, so the
# softmax over the K retrieved documents saturated -- effective_k_mean sat
# at 1.00-1.15 across all 32 marginalized runs, making the marginalization
# arithmetically a no-op and driving the gradient into the retrieval scores
# to exactly zero. cosine bounds the scores to [-1, 1] and removes that
# degree of freedom.
#
# Array index -> N:
#   0 -> N=1
#   1 -> N=2
#   2 -> N=4
#   3 -> N=8
#   4 -> N=16
#   5 -> N=32
#   6 -> N=64
#   7 -> N=128
#
# Prerequisites:
#   sbatch ../mimic_iv_sweep/scripts/prepare_retrieval.sh   (if not already run)
#   sbatch scripts/generate_labels_duration_n1248_frequent.sh
#
# Usage:
#   cd mimic_iv_sweep_frequent
#   sbatch scripts/sweep_marginalized_binary_duration_n1248.sh
#   sbatch --array=0 scripts/sweep_marginalized_binary_duration_n1248.sh   # N=1 only
# ============================================================

#SBATCH --job-name=sweep-marginalized-binary-duration-n1248-frequent
#SBATCH --array=0-7
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
N=${NS[$IDX]}

REPO_DIR="${SLURM_SUBMIT_DIR}"
VENV="${REPO_DIR}/.venv/bin/activate"  # created by: cd mimic_iv_sweep_frequent && uv sync

# Shared with mimic_iv_sweep/ -- not rebuilt here.
RETRIEVAL_DB="${REPO_DIR}/../mimic_iv_sweep/data/retrieval_db"
TENSORIZED_DIR="/groups/mm6677_gp/data/MIMIC_MEDS/MEDS_cohort/processed"
LABELS_DIR="${REPO_DIR}/data/tasks_duration/n${N}/tasks"
OUTPUT_DIR="${REPO_DIR}/outputs/marginalized_binary_duration_n1248/n${N}"

cd "${REPO_DIR}"
# shellcheck source=/dev/null
source "${VENV}"
mkdir -p logs "${OUTPUT_DIR}"

if [ ! -d "${RETRIEVAL_DB}" ]; then
    echo "ERROR: ${RETRIEVAL_DB} not found. Run sbatch ../mimic_iv_sweep/scripts/prepare_retrieval.sh first." >&2
    exit 1
fi

if [ ! -d "${LABELS_DIR}" ]; then
    echo "ERROR: ${LABELS_DIR} not found. Run sbatch scripts/generate_labels_duration_n1248_frequent.sh for N=${N} first." >&2
    exit 1
fi

echo "=== Job info ==="
echo "  Array job              : ${SLURM_ARRAY_JOB_ID:-local}[${IDX}]"
echo "  Node                   : ${SLURMD_NODENAME:-$(hostname)}"
echo "  GPU(s)                 : ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "  N (tasks)              : ${N}"
echo "  Code selection         : most_frequent"
echo "  Task durations         : random, log-uniform [1, 90] days (per task)"
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
    training.trainer.limit_val_batches=1.0 \
    training.trainer.val_check_interval=0.5 \
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
    marginalized_score_similarity=cosine \
    head.in_dim=256 \
    training/loss=multitask_binary_bce_marginalized \
    "training.loss.num_tasks=${N}" \
    "output_dir=${OUTPUT_DIR}" \
    do_overwrite=true \
    "wandb_run_name=marginalized-binary-duration-frequent-n${N}-${SLURM_ARRAY_JOB_ID:-local}_${IDX}" \
    "$@"

echo ""
echo "=== Done: $(date) ==="
