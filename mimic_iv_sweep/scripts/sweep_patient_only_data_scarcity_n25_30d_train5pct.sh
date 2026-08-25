#!/bin/bash
# ============================================================
# SLURM array: patient_only training on the Zach anchor_strategy=uniform_event
#              30d labels, with the TRAINING SET subsampled to 5% of
#              subjects (tuning/held_out splits are the full, unchanged
#              size -- only the training signal shrinks). Model is at FULL
#              capacity (identical to sweep_patient_only_zach_uniform_event_n25_30d.sh) --
#              this is a second, orthogonal scarcity axis to
#              results/capacity_starved_retrieval/ (which shrank the model
#              instead of the data). Pairs with
#              sweep_marginalized_binary_learned_linear_data_scarcity_n25_30d_train5pct.sh,
#              which trains on the exact same subsampled label file.
# ------------------------------------------------------------
# Motivation: results/capacity_starved_retrieval/ found that retrieval
# consistently beats patient_only once the MODEL is too small to represent
# the task on its own (~8x parameter cut). This tests whether the same
# effect shows up when the model is full-capacity but DATA is scarce
# instead -- does retrieval help once there aren't enough training examples
# to learn the task, independent of parameter count? At 5% subsampling
# (see scripts/subsample_train_labels.py), some rare tasks may have very
# few or zero positive training examples by chance -- this is expected and
# not a bug; both patient_only and marginalized face the same handicap on
# a given (draw, task), so the retrieval-vs-no-retrieval comparison stays
# fair even when individual tasks are near-degenerate in-sample.
#
# 5 array tasks = 5 draws, matching generate_labels_zach_uniform_event_n25_30d.sh
# (same 5 draws' train split is subsampled, tuning/held_out untouched).
#
# Prerequisites:
#   sbatch scripts/generate_labels_zach_uniform_event_n25_30d.sh
#   python3 scripts/subsample_train_labels.py <source> <dest> --fraction 0.05 --seed <seed>
#     (run once per draw to produce data/tasks_zach_uniform_event_n25_30d_train5pct/draw<d>/tasks)
#
# Usage:
#   cd mimic_iv_sweep
#   sbatch scripts/sweep_patient_only_data_scarcity_n25_30d_train5pct.sh
# ============================================================

#SBATCH --job-name=sweep-patient-only-data-scarcity-n25-30d-train5pct
#SBATCH --array=0-4
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
DRAW=$((IDX + 1))
N=25

REPO_DIR="${SLURM_SUBMIT_DIR}"
VENV="${REPO_DIR}/.venv/bin/activate"  # created by: cd mimic_iv_sweep && uv sync

TENSORIZED_DIR="/groups/mm6677_gp/data/MIMIC_MEDS/MEDS_cohort/processed"
LABELS_DIR="${REPO_DIR}/data/tasks_zach_uniform_event_n25_30d_train5pct/draw${DRAW}/tasks"
OUTPUT_DIR="${REPO_DIR}/outputs/patient_only_data_scarcity_n25_30d_train5pct/draw${DRAW}"

cd "${REPO_DIR}"
# shellcheck source=/dev/null
source "${VENV}"
mkdir -p logs "${OUTPUT_DIR}"

if [ ! -d "${LABELS_DIR}" ]; then
    echo "ERROR: ${LABELS_DIR} not found. Run scripts/subsample_train_labels.py for draw=${DRAW} fraction=0.05 first." >&2
    exit 1
fi

echo "=== Job info ==="
echo "  Array job     : ${SLURM_ARRAY_JOB_ID:-local}[${IDX}]"
echo "  Node          : ${SLURMD_NODENAME:-$(hostname)}"
echo "  GPU(s)        : ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "  N (tasks)     : ${N}"
echo "  Duration      : 30d"
echo "  Draw          : ${DRAW}/5"
echo "  Architecture  : patient_only (no retrieval), FULL capacity, train subsampled to 5%"
echo "  Started       : $(date)"
echo ""

# Same combo as sweep_patient_only_zach_uniform_event_n25_30d.sh -- only
# LABELS_DIR/OUTPUT_DIR/wandb_run_name (pointing at the 5%-subsampled
# train split) differ here.
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
    "wandb_run_name=patient-only-data-scarcity-train5pct-d30-draw${DRAW}-${SLURM_ARRAY_JOB_ID:-local}_${IDX}" \
    "$@"

echo ""
echo "=== Done: $(date) ==="
