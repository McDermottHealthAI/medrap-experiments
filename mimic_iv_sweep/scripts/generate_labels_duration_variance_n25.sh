#!/bin/bash
# ============================================================
# SLURM array: Generate multi-task binary code-occurrence labels for a
#              duration x variance study -- fixed N=25 random task codes,
#              5 independent random draws EACH, at TWO fixed occurrence-
#              window durations: 7 days and 30 days. Answers "does the
#              patient_only vs. marginalized(binary) gap hold up at both a
#              short and a long fixed horizon, across different random
#              samples of 25 tasks?"
# ------------------------------------------------------------
# Uses MedRAP#98's degenerate-code rejection (duration_distribution="fixed",
# the only mode it applies to): each draw retries invalid codes until 25
# valid ones are found, instead of silently shipping degenerate labels like
# the pre-#98 random-task sweeps did.
#
# 10 array tasks = 2 durations x 5 draws, flattened as
# IDX = duration_idx * 5 + draw_idx (duration_idx in [0,1], draw_idx in [0,4]).
# Same draw seeds as generate_labels_variance_n1248.sh (101/202/303/404/505,
# deliberately distinct from seed=42 used elsewhere) so a "draw 1" here is
# the same task-code draw as "draw 1" there, for the same N -- but N here
# is fixed at 25, not swept.
#
# Same population-matching rationale as generate_labels_n1248.sh: task
# codes/labels are sampled from the *filtered* population training actually
# sees (MEDS_cohort/intermediate), not the raw MEDS_cohort/data root.
#
# Outputs land keyed by duration and draw:
#   data/tasks_duration_variance/d<D>/draw<d>/tasks/{train,tuning,held_out}.parquet
#
# Usage:
#   cd mimic_iv_sweep
#   sbatch scripts/generate_labels_duration_variance_n25.sh
#   sbatch --array=0-4 scripts/generate_labels_duration_variance_n25.sh   # only 7-day, all 5 draws
#   sbatch --array=5-9 scripts/generate_labels_duration_variance_n25.sh  # only 30-day, all 5 draws
# ============================================================

#SBATCH --job-name=gen-labels-duration-variance-n25
#SBATCH --array=0-9
#SBATCH --partition=cpu
#SBATCH --account=mm6677_gp
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

IDX=${SLURM_ARRAY_TASK_ID}

DURATIONS=(7 30)
DRAW_SEEDS=(101 202 303 404 505)

DURATION_IDX=$((IDX / 5))
DRAW_IDX=$((IDX % 5))
DURATION_DAYS=${DURATIONS[$DURATION_IDX]}
SEED=${DRAW_SEEDS[$DRAW_IDX]}
DRAW=$((DRAW_IDX + 1))

REPO_DIR="${SLURM_SUBMIT_DIR}"
VENV="${REPO_DIR}/.venv/bin/activate"  # created by: cd mimic_iv_sweep && uv sync

MEDS_DATA_DIR="/groups/mm6677_gp/data/MIMIC_MEDS/MEDS_cohort/intermediate"
TENSORIZED_DIR="/groups/mm6677_gp/data/MIMIC_MEDS/MEDS_cohort/processed"
OUTPUT_DIR="${REPO_DIR}/data/tasks_duration_variance/d${DURATION_DAYS}/draw${DRAW}"

cd "${REPO_DIR}"
# shellcheck source=/dev/null
source "${VENV}"
mkdir -p logs "${OUTPUT_DIR}"

echo "=== Job info ==="
echo "  Array job     : ${SLURM_ARRAY_JOB_ID:-local}[${IDX}]"
echo "  Node          : ${SLURMD_NODENAME:-$(hostname)}"
echo "  N (tasks)     : 25"
echo "  Duration      : ${DURATION_DAYS}d (fixed)"
echo "  Draw          : ${DRAW}/5 (seed=${SEED})"
echo "  Code selection: random"
echo "  Output        : ${OUTPUT_DIR}"
echo "  Started       : $(date)"
echo ""

medrap-preprocess \
    "meds_data_dir=${MEDS_DATA_DIR}" \
    "tensorized_dir=${TENSORIZED_DIR}" \
    "output_dir=${OUTPUT_DIR}" \
    num_tasks=25 \
    "horizon_days=${DURATION_DAYS}.0" \
    min_history_days=1.0 \
    "seed=${SEED}" \
    do_overwrite=true \
    "$@"

# Gate the labels we just wrote. This is a CHECK, not a filter: it fails the
# run, it never drops the offending tasks and reports the rest. Under
# `set -euo pipefail` a non-zero exit here aborts the job -- which is the
# point: a broken label config costs a few CPU-minutes here instead of hours
# of GPU time in the sweep that reads these labels. Per duration x draw, so
# one bad cell fails its own array task and leaves the other nine alone.
#
# --min-positives 1 means "fail only on single-class tasks", where AUROC is
# undefined. The floor is deliberately that low because these labels use the
# default anchor_strategy=uniform_lifetime, whose median task lands ~2
# positives per split; a real floor (say 100) would fail every cell in this
# array. For labels that can carry a real floor, see
# scripts/generate_labels_anchored_n1248.sh.
#
# This config is duration_distribution="fixed", so MedRAP#98's degenerate-code
# rejection already redraws single-class codes upstream: a FAIL here means
# that rejection did not do its job (most likely a splits mismatch, or labels
# built by a pin other than pyproject.toml's), not merely an unlucky draw.
python scripts/check_task_balance.py "${OUTPUT_DIR}/tasks" --min-positives 1 --quiet

echo ""
echo "=== Done: $(date) ==="
echo "Labels saved to ${OUTPUT_DIR}/tasks"
