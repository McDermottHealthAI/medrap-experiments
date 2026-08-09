#!/bin/bash
# ============================================================
# SLURM array: cross-attention marginalized retrieval
#              (marginalized_output_mode=binary, k=4) with
#              query_projector=qwen3_text_adapter (MedRAP experiment/
#              qwen3-text-query-projector, ResidualAdapterQueryProjector)
#              -- same Zach anchor_strategy=uniform_event 7d labels as
#              sweep_marginalized_binary_qwen3_text_query_n25_7d.sh, only
#              the query projector changes.
# ------------------------------------------------------------
# sweep_marginalized_binary_qwen3_text_query_n25_7d.sh (MedRAP#101) fixed
# retrieval-space *alignment* (queries and docs share a space by
# construction, via a frozen Qwen3 encoder on both sides) and roughly
# halved the AUROC gap to patient_only, but both sides of retrieval are
# now fully frozen -- the model has zero ability to learn which similarity
# structure actually matters for this task. ResidualAdapterQueryProjector
# wraps the frozen Qwen3TextQueryProjector with a small trainable low-rank
# residual (query = base + up(relu(down(base)))), zero-initialized so
# training starts exactly at the validated Qwen3 alignment and only departs
# from it as gradients justify it.
#
# patient_only doesn't use the query projector at all (fusion=passthrough),
# so its results are unaffected by this change -- reuse
# sweep_patient_only_zach_uniform_event_n25_7d.sh's existing results
# (job 9194141) rather than rerunning it.
#
# 5 array tasks = 5 draws, matching generate_labels_zach_uniform_event_n25_7d.sh.
#
# Prerequisites:
#   sbatch scripts/prepare_retrieval.sh
#   sbatch scripts/generate_labels_zach_uniform_event_n25_7d.sh
#
# Usage:
#   cd mimic_iv_sweep
#   sbatch scripts/sweep_marginalized_binary_qwen3_adapter_query_n25_7d.sh
# ============================================================

#SBATCH --job-name=sweep-marginalized-binary-qwen3-adapter-query-n25-7d
#SBATCH --array=0-4
#SBATCH --partition=gpu
#SBATCH --account=mm6677_gp
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

IDX=${SLURM_ARRAY_TASK_ID}
DRAW=$((IDX + 1))
N=25

REPO_DIR="${SLURM_SUBMIT_DIR}"
VENV="${REPO_DIR}/.venv/bin/activate"  # created by: cd mimic_iv_sweep && uv sync

RETRIEVAL_DB="${REPO_DIR}/data/retrieval_db"
TENSORIZED_DIR="/groups/mm6677_gp/data/MIMIC_MEDS/MEDS_cohort/processed"
CODE_METADATA_PATH="${TENSORIZED_DIR}/metadata/codes.parquet"
LABELS_DIR="${REPO_DIR}/data/tasks_zach_uniform_event_n25_7d/draw${DRAW}/tasks"
OUTPUT_DIR="${REPO_DIR}/outputs/marginalized_binary_qwen3_adapter_query_n25_7d/draw${DRAW}"

cd "${REPO_DIR}"
# shellcheck source=/dev/null
source "${VENV}"
mkdir -p logs "${OUTPUT_DIR}"

if [ ! -d "${RETRIEVAL_DB}" ]; then
    echo "ERROR: ${RETRIEVAL_DB} not found. Run sbatch scripts/prepare_retrieval.sh first." >&2
    exit 1
fi

if [ ! -d "${LABELS_DIR}" ]; then
    echo "ERROR: ${LABELS_DIR} not found. Run sbatch scripts/generate_labels_zach_uniform_event_n25_7d.sh for draw=${DRAW} first." >&2
    exit 1
fi

if [ ! -f "${CODE_METADATA_PATH}" ]; then
    echo "ERROR: ${CODE_METADATA_PATH} not found." >&2
    exit 1
fi

echo "=== Job info ==="
echo "  Array job              : ${SLURM_ARRAY_JOB_ID:-local}[${IDX}]"
echo "  Node                   : ${SLURMD_NODENAME:-$(hostname)}"
echo "  GPU(s)                 : ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "  N (tasks)              : ${N}"
echo "  Duration               : 7d"
echo "  Draw                   : ${DRAW}/5"
echo "  marginalized_output_mode: binary (MedRAP#93)"
echo "  query_projector        : qwen3_text_adapter (trainable residual adapter)"
echo "  Started                : $(date)"
echo ""

# Same combo as sweep_marginalized_binary_qwen3_text_query_n25_7d.sh --
# only query_projector (and OUTPUT_DIR/wandb_run_name) differ here.
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
    query_projector=qwen3_text_adapter \
    query_projector.base.model_name_or_path=Qwen/Qwen3-Embedding-0.6B \
    "query_projector.base.code_metadata_path=${CODE_METADATA_PATH}" \
    query_projector.base.max_codes=32 \
    query_projector.base.device=cuda \
    query_projector.dim=1024 \
    query_projector.rank=64 \
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
    "wandb_run_name=marginalized-binary-qwen3-adapter-query-d7-draw${DRAW}-${SLURM_ARRAY_JOB_ID:-local}_${IDX}" \
    "$@"

echo ""
echo "=== Done: $(date) ==="
