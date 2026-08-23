#!/bin/bash
# ============================================================
# SLURM array: Generate multi-task binary code-occurrence labels using
#              Zach's anchor_strategy="uniform_event" design
#              (McDermottHealthAI/MedRAP#100), integrated with everything
#              else this experiment repo depends on (degenerate-code
#              rejection #98, code_selection, eval config #95/97) since #100
#              was opened against a stale main and doesn't have those on its
#              own -- see McDermottHealthAI/MedRAP#experiment/zach-uniform-event-plus-stack.
# ------------------------------------------------------------
# Fixed N=25 random task codes, 5 independent random draws, 30-day fixed
# occurrence window only -- the 30-day half of the same slice covered by
# generate_labels_zach_uniform_event_n25_7d.sh; see
# generate_labels_duration_variance_n25.sh for the full 7d+30d version
# already run with the plain real-event fix, MedRAP#99.
#
# anchor_strategy=uniform_event is passed explicitly even though it's the
# branch's default, for self-documentation: this run exists specifically to
# test Zach's TIMELINE-token-excluding refinement over #99's anchor
# sampling (Zach's _clinical_events() filter -- see the module docstring in
# task_generation.py on the integration branch).
#
# 5 array tasks = 5 draws, same draw seeds as generate_labels_duration_variance_n25.sh
# (101/202/303/404/505) so "draw 1" here picks the same task codes as
# "draw 1" there -- old (plain real-event) vs. new (Zach's refined
# real-event) is an apples-to-apples comparison.
#
# Outputs land in a separate tree from the other duration-variance results:
#   data/tasks_zach_uniform_event_n25_30d/draw<d>/tasks/{train,tuning,held_out}.parquet
#
# Usage:
#   cd mimic_iv_sweep
#   sbatch scripts/generate_labels_zach_uniform_event_n25_30d.sh
# ============================================================

#SBATCH --job-name=gen-labels-zach-uniform-event-n25-30d
#SBATCH --array=0-4
#SBATCH --partition=cpu
#SBATCH --account=mm6677_gp
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

IDX=${SLURM_ARRAY_TASK_ID}

DRAW_SEEDS=(101 202 303 404 505)
SEED=${DRAW_SEEDS[$IDX]}
DRAW=$((IDX + 1))

REPO_DIR="${SLURM_SUBMIT_DIR}"
VENV="${REPO_DIR}/.venv/bin/activate"  # created by: cd mimic_iv_sweep && uv sync

MEDS_DATA_DIR="/groups/mm6677_gp/data/MIMIC_MEDS/MEDS_cohort/intermediate"
TENSORIZED_DIR="/groups/mm6677_gp/data/MIMIC_MEDS/MEDS_cohort/processed"
OUTPUT_DIR="${REPO_DIR}/data/tasks_zach_uniform_event_n25_30d/draw${DRAW}"

cd "${REPO_DIR}"
# shellcheck source=/dev/null
source "${VENV}"
mkdir -p logs "${OUTPUT_DIR}"

echo "=== Job info ==="
echo "  Array job      : ${SLURM_ARRAY_JOB_ID:-local}[${IDX}]"
echo "  Node           : ${SLURMD_NODENAME:-$(hostname)}"
echo "  N (tasks)      : 25"
echo "  Duration       : 30d (fixed)"
echo "  Draw           : ${DRAW}/5 (seed=${SEED})"
echo "  Code selection : random"
echo "  Anchor strategy: uniform_event (Zach's PR#100 design)"
echo "  Output         : ${OUTPUT_DIR}"
echo "  Started        : $(date)"
echo ""

medrap-preprocess \
    "meds_data_dir=${MEDS_DATA_DIR}" \
    "tensorized_dir=${TENSORIZED_DIR}" \
    "output_dir=${OUTPUT_DIR}" \
    num_tasks=25 \
    horizon_days=30.0 \
    min_history_days=1.0 \
    "seed=${SEED}" \
    anchor_strategy=uniform_event \
    do_overwrite=true \
    "$@"

echo ""
echo "=== Done: $(date) ==="
echo "Labels saved to ${OUTPUT_DIR}/tasks"
