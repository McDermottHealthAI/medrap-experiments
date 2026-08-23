#!/bin/bash
# ============================================================
# SLURM array: patient_only training on the Zach anchor_strategy=uniform_event
#              30d labels, with the patient encoder SEVERELY capacity-starved
#              (encoder.embedding_dim 128->16, num_heads 4->1, num_layers
#              2->1, ff_dim 256->32, max_seq_len 256->32 -- ~8x cut across
#              the board). Pairs with
#              sweep_marginalized_binary_qwen3_text_capacity_starved_n25_30d.sh,
#              which applies the IDENTICAL encoder/seq_len cut to the
#              retrieval-augmented architecture -- only the presence/absence
#              of retrieval differs between the two.
# ------------------------------------------------------------
# Motivation: every retrieval variant tried so far (learned linear query
# projector, frozen Qwen3-aligned projector, trainable residual adapter on
# top of it) has failed to make marginalized(binary) CONSISTENTLY beat
# patient_only at full model capacity. This tests a different hypothesis:
# maybe retrieval only helps when the base model doesn't have enough
# capacity to represent the task on its own from patient history alone. If
# retrieval carries real information, a capacity-starved patient_only
# should degrade much more sharply than a starved marginalized (which can
# lean on retrieved content the encoder itself doesn't need to memorize).
# If both degrade together, that's evidence the retrieved MedRAG/textbooks
# content isn't adding real information regardless of capacity pressure --
# narrowing the failure down to the corpus/data rather than model capacity
# or training mechanics.
#
# 5 array tasks = 5 draws, matching generate_labels_zach_uniform_event_n25_30d.sh.
#
# Prerequisites:
#   sbatch scripts/generate_labels_zach_uniform_event_n25_30d.sh
#
# Usage:
#   cd mimic_iv_sweep
#   sbatch scripts/sweep_patient_only_capacity_starved_n25_30d.sh
# ============================================================

#SBATCH --job-name=sweep-patient-only-capacity-starved-n25-30d
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
LABELS_DIR="${REPO_DIR}/data/tasks_zach_uniform_event_n25_30d/draw${DRAW}/tasks"
OUTPUT_DIR="${REPO_DIR}/outputs/patient_only_capacity_starved_n25_30d/draw${DRAW}"

cd "${REPO_DIR}"
# shellcheck source=/dev/null
source "${VENV}"
mkdir -p logs "${OUTPUT_DIR}"

if [ ! -d "${LABELS_DIR}" ]; then
    echo "ERROR: ${LABELS_DIR} not found. Run sbatch scripts/generate_labels_zach_uniform_event_n25_30d.sh for draw=${DRAW} first." >&2
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
echo "  encoder       : embedding_dim=16 num_heads=1 num_layers=1 ff_dim=32"
echo "  max_seq_len   : 32"
echo "  Started       : $(date)"
echo ""

# Same combo as sweep_patient_only_zach_uniform_event_n25_30d.sh -- only the
# encoder/max_seq_len/head.in_dim capacity cut (and OUTPUT_DIR/wandb_run_name)
# differ here.
medrap-train \
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
    training.trainer.max_epochs=3 \
    training.trainer.accelerator=gpu \
    training.trainer.devices=1 \
    training.trainer.gradient_clip_val=1.0 \
    training.trainer.log_every_n_steps=10 \
    training.module.lr=1e-3 \
    training.module.warmup_steps=200 \
    training.module.validation_auroc_log_per_task=true \
    fusion=passthrough \
    query_projector.in_dim=16 \
    head.in_dim=16 \
    training/loss=multitask_binary_bce \
    "output_dir=${OUTPUT_DIR}" \
    do_overwrite=true \
    "wandb_run_name=patient-only-capacity-starved-d30-draw${DRAW}-${SLURM_ARRAY_JOB_ID:-local}_${IDX}" \
    "$@"

echo ""
echo "=== Done: $(date) ==="
