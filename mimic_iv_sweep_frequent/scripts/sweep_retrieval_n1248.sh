#!/bin/bash
# ============================================================
# SLURM array: standard (non-marginalized) cross-attention retrieval,
#              N in {1, 2, 4, 8} -- frequent-code task labels.
# ------------------------------------------------------------
# Same task labels, retrieval index, and base hyperparameters as
# sweep_marginalized_n1248.sh, but the `retrieval` architecture from
# ../mimic_iv_sweep/scripts/sweep_architecture.sh: a single pooled
# cross-attention fusion (not per-document) feeding one LinearHead(B,N),
# trained with plain MultiTaskBCELoss (training/loss=multitask_binary_bce).
#
# Motivation: `logits` here is the head's direct per-task output -- the same
# quantity the loss, accuracy, and AUROC all consistently read. It never goes
# through RetrievalAugmentedModel's per-document softmax-over-tasks path
# (_marginal_class_probabilities), which only runs when
# marginalized_retrieval=true. That path is suspected of producing invalid
# AUROC for independent multi-task outputs (see mimic_iv_sweep_frequent's
# sweep_marginalized_n1248.sh results and README) -- this run gives a
# same-labels, same-retrieval-index reference point unaffected by that
# question, comparable to the older mt25-rope-cross-attn runs (which used the
# same non-marginalized `retrieval` architecture, just on pre-MedRAP#92 task
# labels with positive-rate/count filtering).
#
# Array index -> N:
#   0 -> N=1
#   1 -> N=2
#   2 -> N=4
#   3 -> N=8
#
# Prerequisites:
#   sbatch ../mimic_iv_sweep/scripts/prepare_retrieval.sh   (if not already run)
#   sbatch scripts/generate_labels_n1248_frequent.sh
#
# Usage:
#   cd mimic_iv_sweep_frequent
#   sbatch scripts/sweep_retrieval_n1248.sh
#   sbatch --array=0 scripts/sweep_retrieval_n1248.sh   # N=1 only
# ============================================================

#SBATCH --job-name=sweep-retrieval-n1248-frequent
#SBATCH --array=0-3
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

NS=(1 2 4 8)
N=${NS[$IDX]}

REPO_DIR="${SLURM_SUBMIT_DIR}"
VENV="${REPO_DIR}/.venv/bin/activate"  # created by: cd mimic_iv_sweep_frequent && uv sync

# Shared with mimic_iv_sweep/ -- not rebuilt here, see
# sweep_marginalized_n1248.sh's header note.
RETRIEVAL_DB="${REPO_DIR}/../mimic_iv_sweep/data/retrieval_db"
TENSORIZED_DIR="/groups/mm6677_gp/data/MIMIC_MEDS/MEDS_cohort/processed"
LABELS_DIR="${REPO_DIR}/data/tasks/n${N}/tasks"
OUTPUT_DIR="${REPO_DIR}/outputs/retrieval_n1248/n${N}"

cd "${REPO_DIR}"
# shellcheck source=/dev/null
source "${VENV}"
mkdir -p logs "${OUTPUT_DIR}"

if [ ! -d "${RETRIEVAL_DB}" ]; then
    echo "ERROR: ${RETRIEVAL_DB} not found. Run sbatch ../mimic_iv_sweep/scripts/prepare_retrieval.sh first." >&2
    exit 1
fi

echo "=== Job info ==="
echo "  Array job     : ${SLURM_ARRAY_JOB_ID:-local}[${IDX}]"
echo "  Node          : ${SLURMD_NODENAME:-$(hostname)}"
echo "  GPU(s)        : ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "  N (tasks)     : ${N}"
echo "  Architecture  : retrieval (non-marginalized)"
echo "  Started       : $(date)"
echo ""

# Same combo as ../mimic_iv_sweep/scripts/sweep_architecture.sh's `retrieval`
# arm -- see that script for the full architecture breakdown; only
# LABELS_DIR/OUTPUT_DIR/wandb_run_name differ here.
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
    fusion=cross_attention_medium \
    head.in_dim=256 \
    training/loss=multitask_binary_bce \
    "output_dir=${OUTPUT_DIR}" \
    do_overwrite=true \
    "wandb_run_name=retrieval-frequent-n${N}-${SLURM_ARRAY_JOB_ID:-local}_${IDX}" \
    "$@"

echo ""
echo "=== Done: $(date) ==="
