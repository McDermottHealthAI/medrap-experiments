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
# SLURM array: held-out TEST split AUROC for the data-scarcity patient_only
#              checkpoint at train5pct
#              (sweep_patient_only_data_scarcity_n25_30d_train5pct.sh).
# ------------------------------------------------------------
# Reproduction gate: confirms the published val numbers in
# results/data_scarcity_retrieval/README.md still hold on the restored
# tensorized cohort before any ablation is interpreted, and establishes the
# test-split patient_only floor these ablations compare against.
#
# 5 array tasks = 5 draws.
#
# Prerequisites:
#   sweep_patient_only_data_scarcity_n25_30d_train5pct.sh must have
#   already finished (checkpoints at
#   outputs/patient_only_data_scarcity_n25_30d_train5pct/draw<d>/checkpoints/last.ckpt).
#
# Usage:
#   cd mimic_iv_data_scarcity_ablations
#   sbatch scripts/eval_test_patient_only_data_scarcity_train5pct_repro.sh
# ============================================================

#SBATCH --job-name=mimic-ds-abl-patient-only-repro-train5pct
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

TENSORIZED_DIR="/groups/mm6677_gp/zzw2102/data/MIMIC_MEDS_restored/processed"
LABELS_DIR="/groups/mm6677_gp/hs3627/medrap-experiments/mimic_iv_sweep/data/tasks_zach_uniform_event_n25_30d_train5pct/draw${DRAW}/tasks"
TRAIN_OUTPUT_DIR="/groups/mm6677_gp/hs3627/medrap-experiments/mimic_iv_sweep/outputs/patient_only_data_scarcity_n25_30d_train5pct/draw${DRAW}"
CHECKPOINT_PATH="${TRAIN_OUTPUT_DIR}/checkpoints/last.ckpt"
EVAL_OUTPUT_DIR="${REPO_DIR}/outputs/patient_only_data_scarcity_n25_30d_train5pct_test_repro/draw${DRAW}"

cd "${REPO_DIR}"
# shellcheck source=/dev/null
source "${VENV}"
mkdir -p logs "${EVAL_OUTPUT_DIR}"

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
echo "  Architecture  : patient_only (no retrieval), FULL capacity, train5pct"
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
    fusion=passthrough \
    query_projector.in_dim=128 \
    head.in_dim=128 \
    training/loss=multitask_binary_bce \
    "checkpoint_path=${CHECKPOINT_PATH}" \
    eval_mode=test \
    "output_dir=${EVAL_OUTPUT_DIR}" \
    do_overwrite=true \
    "wandb_run_name=mimic-ds-abl-patient-only-repro-train5pct-d30-draw${DRAW}-${SLURM_ARRAY_JOB_ID:-local}_${IDX}" \
    "$@"

echo ""
echo "=== Done: $(date) ==="
