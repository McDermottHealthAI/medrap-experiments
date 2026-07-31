#!/bin/bash
# ============================================================
# SLURM array: evaluate the base random-task marginalized(binary)
#              checkpoints (sweep_marginalized_binary_n1248.sh, N=1..128,
#              k=4 trained, random-task-code labels) held-out test AUROC in
#              two "inference style" single-document modes:
#              - top1:    retriever.k=1, ablation_mode=none -- the single
#                         best-retrieved (nearest-neighbor) document.
#              - random1: retriever.k=1, ablation_mode=random_docs -- a
#                         uniform-random corpus document, breaking
#                         patient-document alignment (MedRAP's built-in
#                         retrieval ablation, medrap/model/retrievers.py).
#              Compares whether the trained model's marginalization is
#              actually using retrieval quality, or whether any single
#              document (even an irrelevant random one) performs
#              comparably at inference time.
# ------------------------------------------------------------
# 16 array tasks = 8 N values x 2 modes, flattened as
# IDX = mode_idx * 8 + n_idx (mode_idx in [0,1]: 0=top1, 1=random1;
# n_idx in [0,7]).
#
# Same rationale as mimic_iv_sweep_frequent/scripts/eval_marginalized_binary_top1_n1248.sh
# for why k=1 is valid on a k=4-trained checkpoint: PerDocCrossAttentionFusion
# and marginalized_output_mode=binary marginalization are both K-agnostic
# (no weight/positional embedding sized by K), so this checkpoint's weights
# are valid at any K.
#
# Requires the same medrap pin as the training scripts (McDermottHealthAI/MedRAP
# main + PR#93/94/95/96/97/98, pinned in pyproject.toml) -- eval_mode=test
# AUROC logging needs #94; marginalized_*/wandb_* on _eval.yaml needs #95;
# do_overwrite on medrap-eval needs #97.
#
# Array index -> (mode, N):
#   0-7   -> top1,    N=1,2,4,8,16,32,64,128
#   8-15  -> random1, N=1,2,4,8,16,32,64,128
#
# Prerequisites:
#   sweep_marginalized_binary_n1248.sh must have already finished (checkpoints
#   at outputs/marginalized_binary_n1248/n<N>/checkpoints/last.ckpt).
#
# Usage:
#   cd mimic_iv_sweep
#   sbatch scripts/eval_inference_style_n1248.sh
#   sbatch --array=0-7 scripts/eval_inference_style_n1248.sh    # top1 only
#   sbatch --array=8-15 scripts/eval_inference_style_n1248.sh   # random1 only
# ============================================================

#SBATCH --job-name=eval-inference-style-n1248
#SBATCH --array=0-15
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

NS=(1 2 4 8 16 32 64 128)
MODES=(top1 random1)

MODE_IDX=$((IDX / 8))
N_IDX=$((IDX % 8))
N=${NS[$N_IDX]}
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
LABELS_DIR="${REPO_DIR}/data/tasks/n${N}/tasks"
TRAIN_OUTPUT_DIR="${REPO_DIR}/outputs/marginalized_binary_n1248/n${N}"
CHECKPOINT_PATH="${TRAIN_OUTPUT_DIR}/checkpoints/last.ckpt"
EVAL_OUTPUT_DIR="${REPO_DIR}/outputs/marginalized_binary_n1248_${MODE}_eval/n${N}"

cd "${REPO_DIR}"
# shellcheck source=/dev/null
source "${VENV}"
mkdir -p logs "${EVAL_OUTPUT_DIR}"

if [ ! -f "${CHECKPOINT_PATH}" ]; then
    echo "ERROR: ${CHECKPOINT_PATH} not found. Run sweep_marginalized_binary_n1248.sh for N=${N} first." >&2
    exit 1
fi

echo "=== Job info ==="
echo "  Array job     : ${SLURM_ARRAY_JOB_ID:-local}[${IDX}]"
echo "  Node          : ${SLURMD_NODENAME:-$(hostname)}"
echo "  GPU(s)        : ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "  N (tasks)     : ${N}"
echo "  Mode          : ${MODE} (ablation_mode=${ABLATION_MODE})"
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
    "wandb_run_name=marginalized-binary-${MODE}-test-n${N}-${SLURM_ARRAY_JOB_ID:-local}_${IDX}" \
    "$@"

echo ""
echo "=== Done: $(date) ==="
