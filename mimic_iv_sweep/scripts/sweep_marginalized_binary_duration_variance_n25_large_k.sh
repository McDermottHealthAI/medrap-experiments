#!/bin/bash
# ============================================================
# SLURM array: cross-attention marginalized retrieval
#              (marginalized_output_mode=binary) at LARGER retriever k
#              (128, 256, vs. the usual k=4) for the duration x variance
#              study -- fixed N=25, 5 independent random draws, at 7-day
#              and 30-day fixed occurrence-window durations. Reuses the
#              exact same labels as
#              sweep_marginalized_binary_duration_variance_n25.sh (no
#              relabeling needed -- k doesn't affect task-code selection or
#              anchor sampling); only compares against that script's k=4
#              results and the (retrieval-independent) patient_only
#              baseline from sweep_patient_only_duration_variance_n25.sh.
# ------------------------------------------------------------
# 20 array tasks = 2 k values x 2 durations x 5 draws, flattened as
# IDX = k_idx * 10 + duration_idx * 5 + draw_idx (k_idx in [0,1],
# duration_idx in [0,1], draw_idx in [0,4]).
#
# PerDocCrossAttentionFusion (fusion=cross_attention_perdoc_medium) expands
# the batch to B*K internally -- it runs cross-attention once per retrieved
# document, so memory scales ~linearly with k. batch_size is intentionally
# left at 32 (same as every other sweep in this directory, effective
# B*K=4096/8192 at k=128/256) rather than scaled down -- if a job OOMs,
# that's a real, useful data point (tells us the largest k this
# architecture/GPU can actually support), not something to avoid by
# guessing a smaller batch size upfront.
#
# Array index -> (k, duration, draw):
#   0-4   -> k=128, 7d,  draws 1-5
#   5-9   -> k=128, 30d, draws 1-5
#   10-14 -> k=256, 7d,  draws 1-5
#   15-19 -> k=256, 30d, draws 1-5
#
# Prerequisites:
#   sbatch scripts/prepare_retrieval.sh
#   sbatch scripts/generate_labels_duration_variance_n25.sh   (already run;
#     labels already exist at data/tasks_duration_variance/d{7,30}/draw{1-5}/tasks)
#
# Usage:
#   cd mimic_iv_sweep
#   sbatch scripts/sweep_marginalized_binary_duration_variance_n25_large_k.sh
#   sbatch --array=0-4 scripts/sweep_marginalized_binary_duration_variance_n25_large_k.sh   # k=128, 7d only, all draws
# ============================================================

#SBATCH --job-name=sweep-marginalized-binary-duration-variance-n25-large-k
#SBATCH --array=0-19
#SBATCH --partition=gpu
#SBATCH --account=mm6677_gp
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=16:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

IDX=${SLURM_ARRAY_TASK_ID}

K_VALUES=(128 256)
DURATIONS=(7 30)

K_IDX=$((IDX / 10))
REM=$((IDX % 10))
DURATION_IDX=$((REM / 5))
DRAW_IDX=$((REM % 5))

K=${K_VALUES[$K_IDX]}
DURATION_DAYS=${DURATIONS[$DURATION_IDX]}
DRAW=$((DRAW_IDX + 1))
N=25
BATCH_SIZE=32

REPO_DIR="${SLURM_SUBMIT_DIR}"
VENV="${REPO_DIR}/.venv/bin/activate"  # created by: cd mimic_iv_sweep && uv sync

RETRIEVAL_DB="${REPO_DIR}/data/retrieval_db"
TENSORIZED_DIR="/groups/mm6677_gp/data/MIMIC_MEDS/MEDS_cohort/processed"
LABELS_DIR="${REPO_DIR}/data/tasks_duration_variance/d${DURATION_DAYS}/draw${DRAW}/tasks"
OUTPUT_DIR="${REPO_DIR}/outputs/marginalized_binary_duration_variance_n25_k${K}/d${DURATION_DAYS}/draw${DRAW}"

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
echo "  retriever.k             : ${K}"
echo "  batch_size              : ${BATCH_SIZE} (same as k=4 sweeps, not scaled down)"
echo "  marginalized_output_mode: binary (MedRAP#93)"
echo "  Started                 : $(date)"
echo ""

# Same combo as sweep_marginalized_binary_duration_variance_n25.sh -- only
# retriever.k, training.datamodule.batch_size, OUTPUT_DIR, and
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
    "training.datamodule.batch_size=${BATCH_SIZE}" \
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
    "retriever.k=${K}" \
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
    "wandb_run_name=marginalized-binary-duration-variance-k${K}-d${DURATION_DAYS}-draw${DRAW}-${SLURM_ARRAY_JOB_ID:-local}_${IDX}" \
    "$@"

echo ""
echo "=== Done: $(date) ==="
