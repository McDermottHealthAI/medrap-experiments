#!/bin/bash
# ============================================================
# SLURM array: standard (non-marginalized) cross-attention retrieval,
#              N in {1, 2, 4, 8, 16, 32, 64, 128} -- random-task labels.
# ------------------------------------------------------------
# Random-task-label counterpart of
# ../mimic_iv_sweep_frequent/scripts/sweep_retrieval_n1248.sh: a single
# pooled cross-attention fusion (not per-document) feeding one
# LinearHead(B,N), trained with plain MultiTaskBCELoss
# (training/loss=multitask_binary_bce).
#
# Motivation: `logits` here is the head's direct per-task output -- the same
# quantity the loss, accuracy, and AUROC all consistently read. It never goes
# through RetrievalAugmentedModel's per-document softmax-over-tasks path
# (_marginal_class_probabilities), which only runs when
# marginalized_retrieval=true. Fills in the same-labels four-way comparison
# (patient_only / retrieval / marginalized-categorical / marginalized-binary)
# for the original random-task labels, alongside sweep_patient_only_n1248.sh,
# sweep_marginalized_n1248.sh, and sweep_marginalized_binary_n1248.sh.
#
# Validation protocol changed here (rationale:
# ../mimic_iv_sweep_frequent/NULL_RESULT_DIAGNOSIS.md):
# training.trainer.limit_val_batches=1.0 scores the FULL tuning split on
# every validation pass instead of the 200-batch (6,400-row) prefix
# conf/training/trainer/lightning_wandb.yaml defaults to, and
# training.trainer.val_check_interval moves 0.2 -> 0.5 to offset the cost
# (two validation passes per epoch instead of five). That prefix held so few
# positives per task that a single positive changing rank could move the
# task's AUROC by up to ~0.5, which is the same size as every arm-vs-arm
# difference the four-way comparison has reported.
#
# This arm is edited DELIBERATELY, as an explicit in-scope exception to the
# "leave the retrieval sweeps alone" guidance: the arms of this comparison
# train on the same labels and are read against each other cell-by-cell, so
# the validation protocol is a shared property of the comparison, not a
# per-arm knob. Applying the fix to only some arms would leave this one
# scored on a 6,400-row prefix while the others are scored on the full
# tuning split, i.e. it would convert the confound this change removes into
# a between-arm confound -- strictly worse than leaving all arms alone.
# Numbers from this script are NOT comparable to results published before
# this change.
#
# Only the validation protocol changes here. No architecture, loss, data or
# seed argument is touched, so this arm's trained model is unchanged.
# marginalized_score_similarity in particular is deliberately NOT set: this
# arm never enters the marginalized document-scoring path at all.
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
#   sbatch scripts/prepare_retrieval.sh   (if not already run)
#   sbatch scripts/generate_labels_n1248.sh
#
# Usage:
#   cd mimic_iv_sweep
#   sbatch scripts/sweep_retrieval_n1248.sh
#   sbatch --array=0 scripts/sweep_retrieval_n1248.sh   # N=1 only
# ============================================================

#SBATCH --job-name=sweep-retrieval-n1248
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
VENV="${REPO_DIR}/.venv/bin/activate"  # created by: cd mimic_iv_sweep && uv sync

RETRIEVAL_DB="${REPO_DIR}/data/retrieval_db"
TENSORIZED_DIR="/groups/mm6677_gp/data/MIMIC_MEDS/MEDS_cohort/processed"
LABELS_DIR="${REPO_DIR}/data/tasks/n${N}/tasks"
OUTPUT_DIR="${REPO_DIR}/outputs/retrieval_n1248/n${N}"

cd "${REPO_DIR}"
# shellcheck source=/dev/null
source "${VENV}"
mkdir -p logs "${OUTPUT_DIR}"

if [ ! -d "${RETRIEVAL_DB}" ]; then
    echo "ERROR: ${RETRIEVAL_DB} not found. Run sbatch scripts/prepare_retrieval.sh first." >&2
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

# Same combo as sweep_architecture.sh's `retrieval` arm -- see that script
# for the full architecture breakdown; only LABELS_DIR/OUTPUT_DIR/
# wandb_run_name differ here.
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
    fusion=cross_attention_medium \
    head.in_dim=256 \
    training/loss=multitask_binary_bce \
    "output_dir=${OUTPUT_DIR}" \
    do_overwrite=true \
    "wandb_run_name=retrieval-n${N}-${SLURM_ARRAY_JOB_ID:-local}_${IDX}" \
    "$@"

echo ""
echo "=== Done: $(date) ==="
