#!/bin/bash
# ============================================================
# SLURM array: Generate multi-task binary code-occurrence labels for the
#              task-count ablation (correct methodology re-run), varying
#              num_tasks in {5, 25, 125, 625, 2500} x 5 draws each (25 array
#              tasks total).
#
# Uses the same anchor-fixed methodology as
# generate_labels_zach_uniform_event_n25_30d.sh (anchor_strategy=uniform_event,
# code_selection=random (default), horizon_days=30.0, 5 draws with the same
# seeds 101/202/303/404/505 so draw d picks a comparable cohort split across
# every N level and across the existing N=25/30d baseline).
#
# meds_data_dir points at a freshly rebuilt intermediate/ (the shared
# /groups/mm6677_gp/data/MIMIC_MEDS/MEDS_cohort/intermediate used by every
# other experiment in this repo was deleted 2026-08-25; rebuilt from the
# still-intact raw MEDS_cohort/data/ via MEDS_transform-pipeline with the
# confirmed original filtering params min_subjects_per_code=100,
# min_events_per_subject=10 -- see
# /groups/mm6677_gp/hs3627/medrap-experiments/mimic_iv_sweep/rebuild_intermediate/).
# tensorized_dir reuses the existing restored tensorized cohort (unaffected
# by the intermediate/ deletion, tokenization doesn't depend on num_tasks).
#
# Outputs land in:
#   data/tasks_task_count_ablation_n<N>_30d/draw<d>/tasks/{train,tuning,held_out}.parquet
#
# Usage:
#   cd mimic_iv_sweep
#   sbatch scripts/generate_labels_task_count_ablation_n5_2500_30d.sh
# ============================================================

#SBATCH --job-name=gen-labels-task-count-ablation
#SBATCH --array=0-24
#SBATCH --partition=cpu
#SBATCH --account=mm6677_gp
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

IDX=${SLURM_ARRAY_TASK_ID}

N_LEVELS=(5 25 125 625 2500)
DRAW_SEEDS=(101 202 303 404 505)

N_IDX=$((IDX / 5))
DRAW_IDX=$((IDX % 5))

NUM_TASKS=${N_LEVELS[$N_IDX]}
SEED=${DRAW_SEEDS[$DRAW_IDX]}
DRAW=$((DRAW_IDX + 1))

REPO_DIR="${SLURM_SUBMIT_DIR}"
VENV="${REPO_DIR}/.venv/bin/activate"  # created by: cd mimic_iv_sweep && uv sync

MEDS_DATA_DIR="/groups/mm6677_gp/hs3627/medrap-experiments/mimic_iv_sweep/rebuild_intermediate/intermediate"
TENSORIZED_DIR="/groups/mm6677_gp/zzw2102/data/MIMIC_MEDS_restored/processed"
OUTPUT_DIR="${REPO_DIR}/data/tasks_task_count_ablation_n${NUM_TASKS}_30d/draw${DRAW}"

cd "${REPO_DIR}"
# shellcheck source=/dev/null
source "${VENV}"
mkdir -p logs "${OUTPUT_DIR}"

echo "=== Job info ==="
echo "  Array job      : ${SLURM_ARRAY_JOB_ID:-local}[${IDX}]"
echo "  Node           : ${SLURMD_NODENAME:-$(hostname)}"
echo "  N (tasks)      : ${NUM_TASKS}"
echo "  Duration       : 30d (fixed)"
echo "  Draw           : ${DRAW}/5 (seed=${SEED})"
echo "  Code selection : random"
echo "  Anchor strategy: uniform_event"
echo "  meds_data_dir  : ${MEDS_DATA_DIR}"
echo "  Output         : ${OUTPUT_DIR}"
echo "  Started        : $(date)"
echo ""

medrap-preprocess \
    "meds_data_dir=${MEDS_DATA_DIR}" \
    "tensorized_dir=${TENSORIZED_DIR}" \
    "output_dir=${OUTPUT_DIR}" \
    "num_tasks=${NUM_TASKS}" \
    horizon_days=30.0 \
    min_history_days=1.0 \
    "seed=${SEED}" \
    anchor_strategy=uniform_event \
    do_overwrite=true \
    "$@"

echo ""
echo "=== Done: $(date) ==="
echo "Labels saved to ${OUTPUT_DIR}/tasks"
