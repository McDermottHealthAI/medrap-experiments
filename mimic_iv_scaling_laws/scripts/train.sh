#!/bin/bash
# ============================================================
# SLURM array: one (arm, capacity, data fraction) cell of the scaling-law
# grid -- trains on the subsampled labels, then evaluates the last
# checkpoint on the held_out (test) split. 5 array tasks = 5 draws.
#
# Args:
#   ARM   patient_only | retrieval
#   DIM   encoder.embedding_dim (num_heads=1, num_layers=1, ff_dim=2*DIM,
#         max_seq_len fixed at 256 -- same capacity cut as
#         mimic_iv_sweep's capacity_sweep line)
#   PCT   train-set percentage (labels from scripts/make_labels.sh)
#
# Usage:
#   cd mimic_iv_scaling_laws
#   sbatch --job-name=sl-retrieval-dim16-train10pct scripts/train.sh retrieval 16 10
#   (or scripts/submit_sweep.sh for the whole grid)
# ============================================================

#SBATCH --array=0-4
#SBATCH --partition=gpu
#SBATCH --account=mm6677_gp
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

ARM=$1
DIM=$2
PCT=$3

IDX=${SLURM_ARRAY_TASK_ID}
DRAW=$((IDX + 1))
N=25

REPO_DIR="${SLURM_SUBMIT_DIR}"
VENV="${REPO_DIR}/.venv/bin/activate"  # created by: cd mimic_iv_scaling_laws && uv sync

RETRIEVAL_DB="${REPO_DIR}/data/retrieval_db"
TENSORIZED_DIR="/groups/mm6677_gp/zzw2102/data/MIMIC_MEDS_restored/processed"
LABELS_DIR="${REPO_DIR}/data/tasks_n25_30d_train${PCT}pct/draw${DRAW}/tasks"
RUN_NAME="${ARM}_dim${DIM}_train${PCT}pct"
TRAIN_OUTPUT_DIR="${REPO_DIR}/outputs/${RUN_NAME}/draw${DRAW}"
EVAL_OUTPUT_DIR="${REPO_DIR}/outputs/${RUN_NAME}_test/draw${DRAW}"

cd "${REPO_DIR}"
# shellcheck source=/dev/null
source "${VENV}"
mkdir -p logs "${TRAIN_OUTPUT_DIR}" "${EVAL_OUTPUT_DIR}"

if [ ! -d "${LABELS_DIR}" ]; then
    echo "ERROR: ${LABELS_DIR} not found. Run scripts/make_labels.sh first." >&2
    exit 1
fi

MODEL_ARGS=(
    encoder=rope
    "encoder.embedding_dim=${DIM}"
    encoder.num_heads=1
    encoder.num_layers=1
    "encoder.ff_dim=$((2 * DIM))"
    pooling=masked_mean
    head=linear
    "head.out_dim=${N}"
    training/task=multitask_binary
    "training.task.num_tasks=${N}"
    training/datamodule=meds_multitask
    "training.datamodule.config.tensorized_cohort_dir=${TENSORIZED_DIR}"
    training.datamodule.config.max_seq_len=256
    "training.datamodule.config.task_labels_dir=${LABELS_DIR}"
    "training.datamodule.mt_labels_dir=${LABELS_DIR}"
    "training.datamodule.num_tasks=${N}"
    training.datamodule.batch_size=32
    training.datamodule.config.seq_sampling_strategy=to_end
    training/trainer=lightning_wandb
    training.trainer.accelerator=gpu
    training.trainer.devices=1
    training.module.validation_auroc_log_per_task=true
    "query_projector.in_dim=${DIM}"
)

case "${ARM}" in
    patient_only)
        MODEL_ARGS+=(
            fusion=passthrough
            "head.in_dim=${DIM}"
            training/loss=multitask_binary_bce
        )
        ;;
    retrieval)
        if [ ! -d "${RETRIEVAL_DB}" ]; then
            echo "ERROR: ${RETRIEVAL_DB} not found. See README.md for the symlink." >&2
            exit 1
        fi
        MODEL_ARGS+=(
            query_projector=sequence_mean_1024
            retriever=hf_dataset
            "retriever.dataset_path=${RETRIEVAL_DB}"
            retriever.doc_ids_column=null
            retriever.k=4
            retrieval_encoder=token_feature
            retrieval_encoder.vocab_size=151936
            retrieval_encoder.embedding_dim=64
            fusion=cross_attention_perdoc_medium
            marginalized_retrieval=true
            marginalized_output_mode=binary
            head.in_dim=256
            training/loss=multitask_binary_bce_marginalized
            "training.loss.num_tasks=${N}"
        )
        ;;
    *)
        echo "ERROR: unknown ARM '${ARM}' (expected patient_only or retrieval)." >&2
        exit 1
        ;;
esac

echo "=== Job info ==="
echo "  Array job     : ${SLURM_ARRAY_JOB_ID:-local}[${IDX}]"
echo "  Node          : ${SLURMD_NODENAME:-$(hostname)}"
echo "  Arm           : ${ARM}"
echo "  embedding_dim : ${DIM}"
echo "  Train subset  : ${PCT}%"
echo "  Draw          : ${DRAW}/5"
echo "  Started       : $(date)"
echo ""

medrap-train \
    "${MODEL_ARGS[@]}" \
    training.trainer.max_epochs=3 \
    training.trainer.gradient_clip_val=1.0 \
    training.trainer.log_every_n_steps=10 \
    training.module.lr=1e-3 \
    training.module.warmup_steps=200 \
    "seed=${DRAW}" \
    "output_dir=${TRAIN_OUTPUT_DIR}" \
    do_overwrite=true \
    "wandb_run_name=sl-${RUN_NAME}-draw${DRAW}-${SLURM_ARRAY_JOB_ID:-local}_${IDX}"

medrap-eval \
    "${MODEL_ARGS[@]}" \
    "checkpoint_path=${TRAIN_OUTPUT_DIR}/checkpoints/last.ckpt" \
    eval_mode=test \
    "output_dir=${EVAL_OUTPUT_DIR}" \
    do_overwrite=true \
    "wandb_run_name=sl-${RUN_NAME}-test-draw${DRAW}-${SLURM_ARRAY_JOB_ID:-local}_${IDX}"

echo ""
echo "=== Done: $(date) ==="
