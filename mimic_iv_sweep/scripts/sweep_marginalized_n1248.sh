#!/bin/bash
# ============================================================
# SLURM array: cross-attention marginalized retrieval, N in {1, 2, 4, 8}
# ------------------------------------------------------------
# Runs only the `marginalized` architecture from sweep_architecture.sh
# (RoPE + per-document cross-attention + marginalized_retrieval=true +
# multitask_binary_bce_marginalized loss) across the task-count sweep
# N in {1, 2, 4, 8}, on task labels generated with the new random,
# unfiltered task sampler (no positive-rate/count band -- see
# McDermottHealthAI/MedRAP#92).
#
# Per-task AUROC is logged every validation pass
# (training.module.validation_auroc_log_per_task=true), not just the mean,
# so results can be inspected per sampled task after the fact.
#
# Array index -> N:
#   0 -> N=1
#   1 -> N=2
#   2 -> N=4
#   3 -> N=8
#
# Prerequisites:
#   sbatch scripts/prepare_retrieval.sh
#   sbatch scripts/generate_labels_n1248.sh
#
# Usage:
#   cd mimic_iv_sweep
#   sbatch scripts/sweep_marginalized_n1248.sh
#   sbatch --array=0 scripts/sweep_marginalized_n1248.sh   # N=1 only
# ============================================================

#SBATCH --job-name=sweep-marginalized-n1248
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
VENV="${REPO_DIR}/.venv/bin/activate"  # created by: cd mimic_iv_sweep && uv sync

RETRIEVAL_DB="${REPO_DIR}/data/retrieval_db"
TENSORIZED_DIR="/groups/mm6677_gp/data/MIMIC_MEDS/MEDS_cohort/processed"
LABELS_DIR="${REPO_DIR}/data/tasks/n${N}/tasks"
OUTPUT_DIR="${REPO_DIR}/outputs/marginalized_n1248/n${N}"

cd "${REPO_DIR}"
# shellcheck source=/dev/null
source "${VENV}"
mkdir -p logs "${OUTPUT_DIR}"

echo "=== Job info ==="
echo "  Array job  : ${SLURM_ARRAY_JOB_ID:-local}[${IDX}]"
echo "  Node       : ${SLURMD_NODENAME:-$(hostname)}"
echo "  GPU(s)     : ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "  N (tasks)  : ${N}"
echo "  Started    : $(date)"
echo ""

# Marginalized retrieval: requires marginalized_retrieval=true and a fusion
# module that produces per-document state (cross_attention_perdoc_medium,
# not cross_attention_medium) so the model emits per-doc logits for
# MultiTaskBCEMarginalizedLoss to marginalize over. Same combo as
# sweep_architecture.sh's `marginalized` arm.
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
    head.in_dim=256 \
    training/loss=multitask_binary_bce_marginalized \
    "training.loss.num_tasks=${N}" \
    "output_dir=${OUTPUT_DIR}" \
    do_overwrite=true \
    "wandb_run_name=marginalized-n${N}-${SLURM_ARRAY_JOB_ID:-local}_${IDX}" \
    "$@"

echo ""
echo "=== Done: $(date) ==="
