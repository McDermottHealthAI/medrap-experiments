#!/bin/bash
# ============================================================
# SLURM array: one (arm, capacity, training patients, task count) cell of the
# scaling-law grid -- trains on the subsampled labels, then evaluates the last
# checkpoint on the held_out (test) split. 5 array tasks = 5 draws.
#
# Args:
#   ARM   patient_only | retrieval | retrieval_random_docs | retrieval_null_docs
#   DIM   encoder.embedding_dim (num_heads=1, num_layers=1, ff_dim=2*DIM,
#         max_seq_len fixed at 256 -- same capacity cut as
#         mimic_iv_sweep's capacity_sweep line)
#   N_TRAIN  number of training patients
#   NUM_TASKS  number of tasks (labels from scripts/generate_tasks.sh)
#
# Epochs/warmup scale with N_TRAIN (mimic_iv_sweep's extreme_starved trainN
# schedule) so small-data cells get enough optimizer steps.
#
# Usage:
#   cd mimic_iv_scaling_laws
#   sbatch --job-name=sl-retrieval-dim16-trainN10000-tasks25 scripts/train.sh retrieval 16 10000 25
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
N_TRAIN=$3
NUM_TASKS=$4

IDX=${SLURM_ARRAY_TASK_ID}
DRAW=$((IDX + 1))

REPO_DIR="${SLURM_SUBMIT_DIR}"
VENV="${REPO_DIR}/.venv/bin/activate"  # created by: cd mimic_iv_scaling_laws && uv sync

RETRIEVAL_DB="${REPO_DIR}/data/retrieval_db"
TENSORIZED_DIR="${REPO_DIR}/data/cohort/tensorized"  # built by scripts/build_cohort.sh
LABELS_DIR="${REPO_DIR}/data/tasks_n${NUM_TASKS}_30d_trainN${N_TRAIN}/draw${DRAW}/tasks"
RUN_NAME="${ARM}_dim${DIM}_trainN${N_TRAIN}_tasks${NUM_TASKS}"
# Train and eval write to the node's local NVMe scratch (Slurm creates and deletes
# /localscratch/$USER/$SLURM_JOB_ID); only the small CSV logs and configs are copied back to
# outputs/ at the end. The shared /groups quota (snapshots count against it) can't absorb
# checkpoints from dozens of concurrent jobs.
FINAL_TRAIN_DIR="${REPO_DIR}/outputs/${RUN_NAME}/draw${DRAW}"
FINAL_EVAL_DIR="${REPO_DIR}/outputs/${RUN_NAME}_test/draw${DRAW}"
SCRATCH_DIR="/localscratch/${USER}/${SLURM_JOB_ID:-local}/${RUN_NAME}"
TRAIN_OUTPUT_DIR="${SCRATCH_DIR}/draw${DRAW}"
EVAL_OUTPUT_DIR="${SCRATCH_DIR}/draw${DRAW}_test"

cd "${REPO_DIR}"
# shellcheck source=/dev/null
source "${VENV}"
mkdir -p logs "${TRAIN_OUTPUT_DIR}" "${EVAL_OUTPUT_DIR}"

if [ ! -d "${LABELS_DIR}" ]; then
    echo "ERROR: ${LABELS_DIR} not found. Run scripts/submit_tasks.sh first." >&2
    exit 1
fi

# shellcheck source=scripts/model_args.sh
source "${REPO_DIR}/scripts/model_args.sh"
MODEL_ARGS+=(
    training/trainer=lightning_wandb
    training.trainer.accelerator=gpu
    training.trainer.devices=1
    # Only last.ckpt (the one the test eval uses): no best-val or progress snapshots, which the
    # shared /groups quota can't absorb across ~45 concurrent retrieval jobs. Training is unchanged.
    +training.trainer.callbacks.0.save_top_k=0
    "training.trainer.callbacks.1.fractions=[]"
)

case "${N_TRAIN}" in
    1000) MAX_EPOCHS=10 WARMUP_STEPS=60 ;;
    3000) MAX_EPOCHS=7 WARMUP_STEPS=100 ;;
    10000) MAX_EPOCHS=5 WARMUP_STEPS=200 ;;
    30000) MAX_EPOCHS=4 WARMUP_STEPS=200 ;;
    100000) MAX_EPOCHS=3 WARMUP_STEPS=200 ;;
    *)
        echo "ERROR: no epoch/warmup schedule for N_TRAIN=${N_TRAIN}." >&2
        exit 1
        ;;
esac

echo "=== Job info ==="
echo "  Array job     : ${SLURM_ARRAY_JOB_ID:-local}[${IDX}]"
echo "  Node          : ${SLURMD_NODENAME:-$(hostname)}"
echo "  Arm           : ${ARM}"
echo "  embedding_dim : ${DIM}"
echo "  Tasks         : ${NUM_TASKS}"
echo "  Train patients: ${N_TRAIN} (${MAX_EPOCHS} epochs, ${WARMUP_STEPS} warmup steps)"
echo "  Draw          : ${DRAW}/5"
echo "  Started       : $(date)"
echo ""

medrap-train \
    "${MODEL_ARGS[@]}" \
    "training.trainer.max_epochs=${MAX_EPOCHS}" \
    training.trainer.gradient_clip_val=1.0 \
    training.trainer.log_every_n_steps=10 \
    training.module.lr=1e-3 \
    "training.module.warmup_steps=${WARMUP_STEPS}" \
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

# Copy back what the analysis reads (CSV logs: test metrics, val curves) plus configs. The final
# dirs are replaced, not merged, so a crashed earlier attempt can't leave a stale csv version.
for pair in "${TRAIN_OUTPUT_DIR}:${FINAL_TRAIN_DIR}" "${EVAL_OUTPUT_DIR}:${FINAL_EVAL_DIR}"; do
    src="${pair%%:*}"
    dst="${pair##*:}"
    rm -rf "${dst}"
    mkdir -p "${dst}/loggers"
    cp "${src}"/*.yaml "${dst}/" 2>/dev/null || true
    cp -r "${src}/loggers/csv" "${dst}/loggers/"
done

echo ""
echo "=== Done: $(date) ==="
