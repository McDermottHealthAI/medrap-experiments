#!/bin/bash
# ============================================================
# SLURM array: patient-only (no retrieval), N in {1, 2, 4, 8, 16, 32, 64, 128} --
#              frequent-code task labels.
# ------------------------------------------------------------
# Same task labels and base hyperparameters as sweep_retrieval_n1248.sh /
# sweep_marginalized_n1248.sh, but the `patient_only` architecture from
# ../mimic_iv_sweep/scripts/sweep_architecture.sh: RoPE encoder ->
# masked-mean pooling -> LinearHead(B,N) directly, no retriever/fusion at
# all (fusion=passthrough). Doesn't touch data/retrieval_db -- no
# prepare_retrieval.sh prerequisite.
#
# Motivation: a third leg of the same-labels comparison alongside
# sweep_retrieval_n1248.sh and sweep_marginalized_n1248.sh -- isolates
# whether retrieval (in either form) adds anything over the patient's own
# EHR sequence for these tasks, on labels with real (if still low)
# positive rates.
#
# Array index -> N:
#   0 -> N=1
#   1 -> N=2
#   2 -> N=4
#   3 -> N=8
#   4 -> N=16
#   5 -> N=32
#   6 -> N=64
#   7 -> N=128
#
# Prerequisites:
#   sbatch scripts/generate_labels_n1248_frequent.sh
#
# Usage:
#   cd mimic_iv_sweep_frequent
#   sbatch scripts/sweep_patient_only_n1248.sh
#   sbatch --array=0 scripts/sweep_patient_only_n1248.sh   # N=1 only
# ============================================================

#SBATCH --job-name=sweep-patient-only-n1248-frequent
#SBATCH --array=0-7
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

NS=(1 2 4 8 16 32 64 128)
N=${NS[$IDX]}

REPO_DIR="${SLURM_SUBMIT_DIR}"
VENV="${REPO_DIR}/.venv/bin/activate"  # created by: cd mimic_iv_sweep_frequent && uv sync

TENSORIZED_DIR="/groups/mm6677_gp/data/MIMIC_MEDS/MEDS_cohort/processed"
LABELS_DIR="${REPO_DIR}/data/tasks/n${N}/tasks"
OUTPUT_DIR="${REPO_DIR}/outputs/patient_only_n1248/n${N}"

cd "${REPO_DIR}"
# shellcheck source=/dev/null
source "${VENV}"
mkdir -p logs "${OUTPUT_DIR}"

echo "=== Job info ==="
echo "  Array job     : ${SLURM_ARRAY_JOB_ID:-local}[${IDX}]"
echo "  Node          : ${SLURMD_NODENAME:-$(hostname)}"
echo "  GPU(s)        : ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "  N (tasks)     : ${N}"
echo "  Architecture  : patient_only (no retrieval)"
echo "  Started       : $(date)"
echo ""

# Same combo as ../mimic_iv_sweep/scripts/sweep_architecture.sh's
# `patient_only` arm -- see that script for the full rationale (in
# particular why query_projector.in_dim=128 is still required even though
# fusion=passthrough discards the query_projector output: model.forward()
# calls query_projector(encoder_out) unconditionally).
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
    fusion=passthrough \
    query_projector.in_dim=128 \
    head.in_dim=128 \
    training/loss=multitask_binary_bce \
    "output_dir=${OUTPUT_DIR}" \
    do_overwrite=true \
    "wandb_run_name=patient-only-frequent-n${N}-${SLURM_ARRAY_JOB_ID:-local}_${IDX}" \
    "$@"

echo ""
echo "=== Done: $(date) ==="
