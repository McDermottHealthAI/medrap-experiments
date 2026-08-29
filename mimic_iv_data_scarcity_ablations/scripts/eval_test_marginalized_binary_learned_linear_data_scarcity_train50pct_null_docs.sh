#!/bin/bash
# ============================================================
# MIMIC data-scarcity ablation line: evaluates HAYK's ORIGINAL
# data-scarcity checkpoints (hs3627 working copy, read-only) against the
# snapshot-restored original tensorized cohort. Mechanical clone of
# zzw2102's mimic_iv_null_random_doc_ablations/ eval scripts (same ablation
# infra, same restored cohort, same null corpus), pointed at the
# data-scarcity checkpoints (results/data_scarcity_retrieval/) instead of
# the capacity-starved ones -- that line already has its own random/null
# ablations done and is NOT re-run here.
# ============================================================
# ============================================================
# SLURM array: held-out TEST split AUROC for the data-scarcity
#              marginalized(binary, learned-linear query projector)
#              checkpoint at train50pct -- frozen null-doc ablation
#              (sweep_marginalized_binary_learned_linear_data_scarcity_n25_30d_train50pct.sh).
# ------------------------------------------------------------
# Mechanical clone of zzw2102's capacity-starved random/null-doc ablation
# scripts, pointed at the data-scarcity checkpoints instead. Uses
# retriever.k=4, matching training exactly (marginalized over K, not
# top-1-only inference).
#
# Prerequisites:
#   sweep_marginalized_binary_learned_linear_data_scarcity_n25_30d_train50pct.sh
#   must have already finished.
#
# Usage:
#   cd mimic_iv_data_scarcity_ablations
#   sbatch scripts/eval_test_marginalized_binary_learned_linear_data_scarcity_train50pct_null_docs.sh
# ============================================================

#SBATCH --job-name=mimic-ds-abl-marginalized-null-docs-train50pct
#SBATCH --array=0-4
#SBATCH --partition=gpu
#SBATCH --account=mm6677_gp
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

IDX=${SLURM_ARRAY_TASK_ID}
DRAW=$((IDX + 1))
N=25

REPO_DIR="${SLURM_SUBMIT_DIR}"
VENV="${REPO_DIR}/.venv/bin/activate"  # created by: cd mimic_iv_data_scarcity_ablations && uv sync

RETRIEVAL_DB="${REPO_DIR}/data/retrieval_db_null"
TENSORIZED_DIR="/groups/mm6677_gp/zzw2102/data/MIMIC_MEDS_restored/processed"
LABELS_DIR="/groups/mm6677_gp/hs3627/medrap-experiments/mimic_iv_sweep/data/tasks_zach_uniform_event_n25_30d_train50pct/draw${DRAW}/tasks"
TRAIN_OUTPUT_DIR="/groups/mm6677_gp/hs3627/medrap-experiments/mimic_iv_sweep/outputs/marginalized_binary_data_scarcity_n25_30d_train50pct/draw${DRAW}"
CHECKPOINT_PATH="${TRAIN_OUTPUT_DIR}/checkpoints/last.ckpt"
EVAL_OUTPUT_DIR="${REPO_DIR}/outputs/marginalized_binary_learned_linear_data_scarcity_n25_30d_train50pct_test_eval_null_docs/draw${DRAW}"

cd "${REPO_DIR}"
# shellcheck source=/dev/null
source "${VENV}"
mkdir -p logs "${EVAL_OUTPUT_DIR}"

if [ ! -d "${RETRIEVAL_DB}" ]; then
    echo "ERROR: ${RETRIEVAL_DB} not found." >&2
    exit 1
fi

if [ ! -f "${CHECKPOINT_PATH}" ]; then
    echo "ERROR: ${CHECKPOINT_PATH} not found." >&2
    exit 1
fi

echo "=== Job info ==="
echo "  Array job     : ${SLURM_ARRAY_JOB_ID:-local}[${IDX}]"
echo "  Node          : ${SLURMD_NODENAME:-$(hostname)}"
echo "  N (tasks)     : ${N}"
echo "  Duration      : 30d"
echo "  Draw          : ${DRAW}/5"
echo "  query_projector: sequence_mean_1024 (learned-linear), FULL capacity, train50pct"
echo "  eval_mode     : test (MEDS held_out split)"
echo "  Checkpoint    : ${CHECKPOINT_PATH}"
echo "  Started       : $(date)"
echo ""

medrap-eval \
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
    training.trainer.accelerator=gpu \
    training.trainer.devices=1 \
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
    "checkpoint_path=${CHECKPOINT_PATH}" \
    eval_mode=test \
    "output_dir=${EVAL_OUTPUT_DIR}" \
    do_overwrite=true \
    "wandb_run_name=mimic-ds-abl-marginalized-null-docs-train50pct-d30-draw${DRAW}-${SLURM_ARRAY_JOB_ID:-local}_${IDX}" \
    "$@"

echo ""
echo "=== Done: $(date) ==="
