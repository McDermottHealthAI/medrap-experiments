#!/bin/bash
# ============================================================
# NWICU STEP-MATCHED (E1): identical to the nwicu_sweep script of the same
# name except training.trainer.max_epochs 3 -> 42, matching MIMIC's ~14.4k
# optimizer steps (343 steps/epoch x 42 = 14,406 vs MIMIC 14,364). Tests
# whether NWICU's machinery-not-content ablation verdict was caused by
# undertraining. See results/mimic_iv_explore/.
# ============================================================
# ============================================================
# NWICU REPLICATION of mimic_iv_sweep/scripts/eval_test_patient_only_capacity_starved_n25_30d.sh
# Generated as a mechanical clone: identical Hydra overrides and SBATCH
# resources; only the dataset paths (MIMIC_MEDS -> NWICU), job names, and
# wandb run names differ. See results/capacity_starved_retrieval_nwicu/.
# ============================================================
# ============================================================
# SLURM array: held-out TEST split AUROC for the capacity-starved
#              patient_only checkpoints
#              (sweep_patient_only_capacity_starved_n25_30d.sh).
# ------------------------------------------------------------
# Every AUROC number reported for the capacity-starved experiment so far
# (results/capacity_starved_retrieval/README.md) is val/auroc/mean --
# computed once at the end of fit over the MEDS tuning/validation split
# (see EndOfFitValAUROCCallback), the exact model state saved as
# checkpoints/last.ckpt. Since the capacity-starved result (retrieval
# consistently beating patient_only) was found by comparing variants on
# that same validation split, this confirms the effect on genuinely
# unseen data (eval_mode=test -> MEDS held_out split, never touched by
# any training or model-selection decision in this whole line of
# experiments).
#
# Same starved encoder config as training (embedding_dim=16, num_heads=1,
# num_layers=1, ff_dim=32, max_seq_len=32) -- must match exactly to load
# the checkpoint's state dict.
#
# 5 array tasks = 5 draws, matching sweep_patient_only_capacity_starved_n25_30d.sh.
#
# Prerequisites:
#   sweep_patient_only_capacity_starved_n25_30d.sh must have already
#   finished (checkpoints at
#   outputs/patient_only_capacity_starved_n25_30d/draw<d>/checkpoints/last.ckpt).
#
# Usage:
#   cd nwicu_null_random_doc_ablations
#   sbatch scripts/eval_test_patient_only_capacity_starved_n25_30d.sh
# ============================================================

#SBATCH --job-name=nwicu42-eval-test-patient-only-capacity-starved-n25-30d
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
VENV="${REPO_DIR}/.venv/bin/activate"  # created by: cd nwicu_null_random_doc_ablations && uv sync

TENSORIZED_DIR="/groups/mm6677_gp/data/NWICU/processed"
LABELS_DIR="${REPO_DIR}/data/tasks_zach_uniform_event_n25_30d/draw${DRAW}/tasks"
TRAIN_OUTPUT_DIR="${REPO_DIR}/outputs/patient_only_capacity_starved_n25_30d/draw${DRAW}"
CHECKPOINT_PATH="${TRAIN_OUTPUT_DIR}/checkpoints/last.ckpt"
EVAL_OUTPUT_DIR="${REPO_DIR}/outputs/patient_only_capacity_starved_n25_30d_test_eval/draw${DRAW}"

cd "${REPO_DIR}"
# shellcheck source=/dev/null
source "${VENV}"
mkdir -p logs "${EVAL_OUTPUT_DIR}"

if [ ! -f "${CHECKPOINT_PATH}" ]; then
    echo "ERROR: ${CHECKPOINT_PATH} not found. Run sweep_patient_only_capacity_starved_n25_30d.sh for draw=${DRAW} first." >&2
    exit 1
fi

echo "=== Job info ==="
echo "  Array job     : ${SLURM_ARRAY_JOB_ID:-local}[${IDX}]"
echo "  Node          : ${SLURMD_NODENAME:-$(hostname)}"
echo "  GPU(s)        : ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "  N (tasks)     : ${N}"
echo "  Duration      : 30d"
echo "  Draw          : ${DRAW}/5"
echo "  Architecture  : patient_only (no retrieval), CAPACITY-STARVED"
echo "  eval_mode     : test (MEDS held_out split)"
echo "  Checkpoint    : ${CHECKPOINT_PATH}"
echo "  Started       : $(date)"
echo ""

medrap-eval \
    encoder=rope \
    encoder.embedding_dim=16 \
    encoder.num_heads=1 \
    encoder.num_layers=1 \
    encoder.ff_dim=32 \
    pooling=masked_mean \
    head=linear \
    "head.out_dim=${N}" \
    training/task=multitask_binary \
    "training.task.num_tasks=${N}" \
    training/datamodule=meds_multitask \
    "training.datamodule.config.tensorized_cohort_dir=${TENSORIZED_DIR}" \
    training.datamodule.config.max_seq_len=32 \
    "training.datamodule.config.task_labels_dir=${LABELS_DIR}" \
    "training.datamodule.mt_labels_dir=${LABELS_DIR}" \
    "training.datamodule.num_tasks=${N}" \
    training.datamodule.batch_size=32 \
    training.datamodule.config.seq_sampling_strategy=to_end \
    training/trainer=lightning_wandb \
    training.trainer.accelerator=gpu \
    training.trainer.devices=1 \
    training.module.validation_auroc_log_per_task=true \
    fusion=passthrough \
    query_projector.in_dim=16 \
    head.in_dim=16 \
    training/loss=multitask_binary_bce \
    "checkpoint_path=${CHECKPOINT_PATH}" \
    eval_mode=test \
    "output_dir=${EVAL_OUTPUT_DIR}" \
    do_overwrite=true \
    "wandb_run_name=nwicu42-patient-only-capacity-starved-test-d30-draw${DRAW}-${SLURM_ARRAY_JOB_ID:-local}_${IDX}" \
    "$@"

echo ""
echo "=== Done: $(date) ==="
