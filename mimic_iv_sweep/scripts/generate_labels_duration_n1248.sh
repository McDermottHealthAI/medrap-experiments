#!/bin/bash
# ============================================================
# SLURM array: Generate multi-task binary code-occurrence labels with
#              RANDOM CODE SELECTION and RANDOM PER-TASK DURATIONS (not a
#              single shared horizon_days), for N in {1, 2, 4, 8, 16, 32, 64,
#              128}. Combines this directory's random task-code selection
#              (code_selection=random, the medrap-preprocess default) with
#              ../mimic_iv_sweep_frequent's random-duration variant
#              (duration_distribution=log-uniform) -- the "random tasks,
#              random durations" combination neither directory has run
#              alone.
# ------------------------------------------------------------
# Identical to generate_labels_n1248.sh except duration_distribution=
# log-uniform, min_duration_days=1.0, max_duration_days=90.0 instead of a
# single shared horizon_days=7.0: each of the N randomly-selected task codes
# also gets its own occurrence-window duration, drawn independently from
# [1, 90] days (log-uniform biases toward shorter windows while still
# covering the full range). Prediction time is still a single random anchor
# per subject, sized off the LONGEST sampled duration among that split's
# tasks (anchor_horizon_days = max(durations)) -- see
# McDermottHealthAI/MedRAP#96.
#
# Requires medrap >= McDermottHealthAI/MedRAP@fa8c29d (duration_distribution
# support, pinned in pyproject.toml; not yet merged to MedRAP main).
#
# Note: random-code-selection labels are already known to be extremely
# sparse at low N (see sweep_marginalized_n1248.sh's README caveats --
# n_valid_tasks=0 for N=1/2 the entire run on the fixed-duration random
# labels). Randomizing the duration on top doesn't fix that; if anything, a
# short log-uniform-sampled duration (near the 1-day floor) on an already-
# rare random code makes a degenerate (single-class) task even more likely.
# Don't expect this to look better than
# ../mimic_iv_sweep_frequent/data/tasks_duration -- this run isolates
# "random codes + random durations" as its own combination, not a fix for
# random-code sparsity.
#
# Outputs for index i land in a SEPARATE directory from the fixed-duration
# random-task labels, so this never overwrites data/tasks/n<N> (which every
# other script in this directory reads from):
#   data/tasks_duration/n<N>/tasks/{train,tuning,held_out}.parquet
#
# mem=64G: same OOM sizing rationale as generate_labels_n1248.sh (peak RSS
# tracks the full train-split size read eagerly by task_generation.py).
#
# Usage:
#   cd mimic_iv_sweep
#   sbatch scripts/generate_labels_duration_n1248.sh
#   sbatch --array=2 scripts/generate_labels_duration_n1248.sh   # only N=4
# ============================================================

#SBATCH --job-name=marginalized-gen-labels-duration
#SBATCH --array=0-7
#SBATCH --partition=cpu
#SBATCH --account=mm6677_gp
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

IDX=${SLURM_ARRAY_TASK_ID}

NS=(1 2 4 8 16 32 64 128)
N=${NS[$IDX]}

REPO_DIR="${SLURM_SUBMIT_DIR}"
VENV="${REPO_DIR}/.venv/bin/activate"  # created by: cd mimic_iv_sweep && uv sync

# Same population-matching rationale as generate_labels_n1248.sh:
# task codes/labels must be sampled from the *filtered* population that
# training actually sees, not the raw MEDS_cohort/data root.
MEDS_DATA_DIR="/groups/mm6677_gp/data/MIMIC_MEDS/MEDS_cohort/intermediate"
TENSORIZED_DIR="/groups/mm6677_gp/data/MIMIC_MEDS/MEDS_cohort/processed"
OUTPUT_DIR="${REPO_DIR}/data/tasks_duration/n${N}"

cd "${REPO_DIR}"
# shellcheck source=/dev/null
source "${VENV}"
mkdir -p logs "${OUTPUT_DIR}"

echo "=== Job info ==="
echo "  Array job          : ${SLURM_ARRAY_JOB_ID:-local}[${IDX}]"
echo "  Node                : ${SLURMD_NODENAME:-$(hostname)}"
echo "  N (tasks)           : ${N}"
echo "  Code selection      : random"
echo "  duration_distribution: log-uniform, [1.0, 90.0] days"
echo "  Output              : ${OUTPUT_DIR}"
echo "  Started             : $(date)"
echo ""

medrap-preprocess \
    "meds_data_dir=${MEDS_DATA_DIR}" \
    "tensorized_dir=${TENSORIZED_DIR}" \
    "output_dir=${OUTPUT_DIR}" \
    "num_tasks=${N}" \
    min_history_days=1.0 \
    seed=42 \
    duration_distribution=log-uniform \
    min_duration_days=1.0 \
    max_duration_days=90.0 \
    do_overwrite=true \
    "$@"

# Gate the labels we just wrote. This is a CHECK, not a filter: it fails the
# run, it never drops the offending tasks and reports the rest. Under
# `set -euo pipefail` a non-zero exit here aborts the job -- which is the
# point: a broken label config costs a few CPU-minutes here instead of hours
# of GPU time in the sweep that reads these labels. It matters more here than
# in the fixed-horizon scripts: MedRAP#98's degenerate-code rejection only
# runs for duration_distribution="fixed", so this log-uniform config keeps
# whatever _select_task_codes draws and this gate is the only thing standing
# between a degenerate draw and the GPU queue.
#
# --min-positives 1 means "fail only on single-class tasks", where AUROC is
# undefined. The floor is deliberately that low because these labels use the
# default anchor_strategy=uniform_lifetime, whose median task lands ~2
# positives per split; a real floor (say 100) would fail every N in this
# array. For labels that can carry a real floor, see
# scripts/generate_labels_anchored_n1248.sh.
python scripts/check_task_balance.py "${OUTPUT_DIR}/tasks" --min-positives 1 --quiet

echo ""
echo "=== Done: $(date) ==="
echo "Labels saved to ${OUTPUT_DIR}/tasks"
