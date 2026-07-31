#!/bin/bash
# ============================================================
# SLURM array: evaluate the large-k duration x variance study's
#              marginalized(binary) checkpoints
#              (sweep_marginalized_binary_duration_variance_n25_large_k.sh,
#              N=25, k trained in {32,64,128}, 5 draws, 7d/30d) held-out
#              test AUROC in two "inference style" single-document modes:
#              - top1:    retriever.k=1, ablation_mode=none.
#              - random1: retriever.k=1, ablation_mode=random_docs.
#              Eval always uses retriever.k=1 regardless of the checkpoint's
#              TRAINED k -- see eval_inference_style_n1248.sh for why this
#              is valid (K-agnostic marginalization). TRAIN_K only selects
#              which checkpoint directory to load.
# ------------------------------------------------------------
# 60 array tasks = 2 modes x 3 trained-k values x 2 durations x 5 draws,
# flattened as IDX = mode_idx * 30 + k_idx * 10 + duration_idx * 5 + draw_idx
# (mode_idx in [0,1]: 0=top1, 1=random1; k_idx in [0,2]: 32,64,128;
# duration_idx in [0,1]; draw_idx in [0,4]) -- same k/duration/draw
# flattening as sweep_marginalized_binary_duration_variance_n25_large_k.sh.
#
# Array index -> (mode, trained-k, duration, draw):
#   0-9    -> top1,    k=32,  7d/30d x draws 1-5
#   10-19  -> top1,    k=64,  7d/30d x draws 1-5
#   20-29  -> top1,    k=128, 7d/30d x draws 1-5
#   30-39  -> random1, k=32,  7d/30d x draws 1-5
#   40-49  -> random1, k=64,  7d/30d x draws 1-5
#   50-59  -> random1, k=128, 7d/30d x draws 1-5
# (within each 10-block: 0-4 = 7d draws 1-5, 5-9 = 30d draws 1-5)
#
# Prerequisites:
#   sweep_marginalized_binary_duration_variance_n25_large_k.sh must have
#   already finished (checkpoints at
#   outputs/marginalized_binary_duration_variance_n25_k<K>/d<D>/draw<d>/checkpoints/last.ckpt).
#   k=256 has no checkpoints (OOM'd during training) and is not covered here.
#
# Usage:
#   cd mimic_iv_sweep
#   sbatch scripts/eval_inference_style_duration_variance_n25_large_k.sh
# ============================================================

#SBATCH --job-name=eval-inference-style-duration-variance-n25-large-k
#SBATCH --array=0-59
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

TRAIN_KS=(32 64 128)
DURATIONS=(7 30)
MODES=(top1 random1)
N=25

MODE_IDX=$((IDX / 30))
REM1=$((IDX % 30))
K_IDX=$((REM1 / 10))
REM2=$((REM1 % 10))
DURATION_IDX=$((REM2 / 5))
DRAW_IDX=$((REM2 % 5))

TRAIN_K=${TRAIN_KS[$K_IDX]}
DURATION_DAYS=${DURATIONS[$DURATION_IDX]}
DRAW=$((DRAW_IDX + 1))
MODE=${MODES[$MODE_IDX]}

if [ "$MODE" = "top1" ]; then
    ABLATION_MODE=none
else
    ABLATION_MODE=random_docs
fi

REPO_DIR="${SLURM_SUBMIT_DIR}"
VENV="${REPO_DIR}/.venv/bin/activate"  # created by: cd mimic_iv_sweep && uv sync

RETRIEVAL_DB="${REPO_DIR}/data/retrieval_db"
TENSORIZED_DIR="/groups/mm6677_gp/data/MIMIC_MEDS/MEDS_cohort/processed"
LABELS_DIR="${REPO_DIR}/data/tasks_duration_variance/d${DURATION_DAYS}/draw${DRAW}/tasks"
TRAIN_OUTPUT_DIR="${REPO_DIR}/outputs/marginalized_binary_duration_variance_n25_k${TRAIN_K}/d${DURATION_DAYS}/draw${DRAW}"
CHECKPOINT_PATH="${TRAIN_OUTPUT_DIR}/checkpoints/last.ckpt"
EVAL_OUTPUT_DIR="${REPO_DIR}/outputs/marginalized_binary_duration_variance_n25_k${TRAIN_K}_${MODE}_eval/d${DURATION_DAYS}/draw${DRAW}"

cd "${REPO_DIR}"
# shellcheck source=/dev/null
source "${VENV}"
mkdir -p logs "${EVAL_OUTPUT_DIR}"

if [ ! -f "${CHECKPOINT_PATH}" ]; then
    echo "ERROR: ${CHECKPOINT_PATH} not found. Run sweep_marginalized_binary_duration_variance_n25_large_k.sh for k=${TRAIN_K} duration=${DURATION_DAYS}d draw=${DRAW} first." >&2
    exit 1
fi

echo "=== Job info ==="
echo "  Array job     : ${SLURM_ARRAY_JOB_ID:-local}[${IDX}]"
echo "  Node          : ${SLURMD_NODENAME:-$(hostname)}"
echo "  GPU(s)        : ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "  N (tasks)     : ${N}"
echo "  Trained k     : ${TRAIN_K}"
echo "  Duration      : ${DURATION_DAYS}d"
echo "  Draw          : ${DRAW}/5"
echo "  Mode          : ${MODE} (ablation_mode=${ABLATION_MODE}, eval retriever.k=1)"
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
    retriever.k=1 \
    "retriever.ablation_mode=${ABLATION_MODE}" \
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
    "wandb_run_name=marginalized-binary-${MODE}-test-duration-variance-k${TRAIN_K}-d${DURATION_DAYS}-draw${DRAW}-${SLURM_ARRAY_JOB_ID:-local}_${IDX}" \
    "$@"

echo ""
echo "=== Done: $(date) ==="
