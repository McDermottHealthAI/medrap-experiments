#!/bin/bash
# ============================================================
# SLURM array: Generate multi-task binary code-occurrence labels with
#              RANDOM PER-TASK DURATIONS (not a single shared horizon_days),
#              for N in {1, 2, 4, 8, 16, 32, 64, 128} -- most-frequent-code
#              selection, same as generate_labels_n1248_frequent.sh.
# ------------------------------------------------------------
# Identical to generate_labels_n1248_frequent.sh except
# duration_distribution=log-uniform, min_duration_days=1.0,
# max_duration_days=90.0 instead of a single shared horizon_days=7.0: each
# sampled task code gets its own occurrence-window duration, drawn
# independently from [1, 90] days (log-uniform biases toward shorter windows
# while still covering the full range). Prediction time is still a single
# random anchor per subject, sized off the LONGEST sampled duration among
# that split's tasks (anchor_horizon_days = max(durations)) -- see
# McDermottHealthAI/MedRAP#96.
#
# Motivation: this repo's task labels have so far all used one fixed 7-day
# occurrence window for every task, regardless of N or code_selection. This
# tests whether letting each task have its own randomly-sampled window
# (following EveryQuery's https://github.com/payalchandak/EveryQuery
# duration-sampling approach, ported -- not the package itself, see MedRAP#96
# for why) changes results, on the same most-frequent-selected codes used
# throughout this directory.
#
# Requires medrap >= McDermottHealthAI/MedRAP@fa8c29d (duration_distribution
# support, pinned in pyproject.toml; not yet merged to MedRAP main).
#
# Outputs for index i land in a SEPARATE directory from the fixed-horizon
# labels, so this never overwrites data/tasks/n<N> (which every other script
# in this directory reads from):
#   data/tasks_duration/n<N>/tasks/{train,tuning,held_out}.parquet
#
# mem=64G: same OOM sizing rationale as generate_labels_n1248_frequent.sh
# (peak RSS tracks the full train-split size read eagerly by
# task_generation.py).
#
# Usage:
#   cd mimic_iv_sweep_frequent
#   sbatch scripts/generate_labels_duration_n1248_frequent.sh
#   sbatch --array=2 scripts/generate_labels_duration_n1248_frequent.sh   # only N=4
# ============================================================

#SBATCH --job-name=marginalized-gen-labels-duration-frequent
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
VENV="${REPO_DIR}/.venv/bin/activate"  # created by: cd mimic_iv_sweep_frequent && uv sync

# Same population-matching rationale as generate_labels_n1248_frequent.sh:
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
echo "  Code selection      : most_frequent"
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
    code_selection=most_frequent \
    duration_distribution=log-uniform \
    min_duration_days=1.0 \
    max_duration_days=90.0 \
    do_overwrite=true \
    "$@"

# Gate the labels we just wrote. This is a CHECK, not a filter: it fails the
# run, it never drops the offending tasks and reports the rest. Under
# `set -euo pipefail` a non-zero exit here aborts the job -- which is the
# point: a broken label config costs a few CPU-minutes here instead of hours
# of GPU time in the sweep that reads these labels. This directory's pin
# (MedRAP@164c2ef) has no degenerate-code rejection at all -- MedRAP#89
# removed positive-rate/count filtering and #98 had not landed -- so
# generate_tasks keeps whatever it draws and this gate is the only check.
# A short log-uniform duration on top of that only makes single-class more
# likely.
#
# --min-positives 1 means "fail only on single-class tasks", where AUROC is
# undefined. The floor is deliberately that low because the anchor is still
# drawn uniformly over the subject's (birth-anchored) lifetime, which leaves
# the median task at ~2 positives per split; a real floor (say 100) would
# fail every N in this array. Raising it needs anchor_strategy=uniform_event,
# which this directory's pin predates -- see
# ../mimic_iv_sweep/scripts/generate_labels_anchored_n1248.sh.
python scripts/check_task_balance.py "${OUTPUT_DIR}/tasks" --min-positives 1 --quiet

echo ""
echo "=== Done: $(date) ==="
echo "Labels saved to ${OUTPUT_DIR}/tasks"
