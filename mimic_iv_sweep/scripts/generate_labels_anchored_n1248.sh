#!/bin/bash
# ============================================================
# SLURM array: Generate multi-task binary code-occurrence labels with
#              EVENT-ANCHORED prediction times (anchor_strategy=uniform_event)
#              for N in {1, 2, 4, 8, 16, 32, 64, 128} -- otherwise identical
#              to generate_labels_n1248.sh.
# ------------------------------------------------------------
# What changes: the per-subject prediction anchor is now a real clinical
# EVENT, drawn uniformly over that subject's own event timestamps, so it
# always lands on a moment where something was actually recorded. Every other
# script in this directory uses the default anchor_strategy=uniform_lifetime,
# which draws uniformly over *calendar time* in the window
# [first_event + min_history_days, last_event - horizon_days]. Both
# strategies use that same window; they differ only in the measure sampled
# over it.
#
# Why the calendar draw is wrong here: on MEDS_cohort/intermediate -- the
# cohort medrap-preprocess actually reads -- first_event is effectively
# BIRTH. The intermediate cohort carries 211,488 TIMELINE// rows, and
# TIMELINE//START sits at the identical timestamp as meds.birth_code for 100%
# of subjects, so the median gap from birth to the first non-birth event is
# 0.000 days (it is 17,287 days on the raw MEDS_cohort, which has no
# TIMELINE// rows at all). That also makes excluding meds.birth_code by
# itself a NO-OP -- and those code exclusions only ever applied to task-code
# selection anyway, never to the anchor bounds. Net effect: a ~59-year anchor
# window laid over ~1 year of actual clinical activity, so the anchor usually
# lands in an empty stretch of a life and the horizon_days window after it
# contains nothing. Measured 0.10% positive labels.
#
# Measured effect, median in-window positive rate over the 8 most frequent
# codes at a 7-day horizon on a 999-subject shard:
#   uniform_lifetime (every other script here) : 0.0010
#   uniform_event    (this script)             : 0.3493   <- ~349x
#
# NOT COMPARABLE TO data/tasks/: these labels ask a different question ("given
# a real clinical event, what happens in the next 7 days?") over a different
# subject population -- uniform_event additionally drops subjects with no
# clinical event inside the window, rather than falling back to a calendar
# anchor. Positive rates, n_valid_tasks, and any AUROC computed on them do not
# belong on the same axis as a data/tasks/ run. Compare anchored to anchored
# only; a headline number mixing the two is measuring the anchor, not the
# model.
#
# Requires medrap with anchor_strategy support (McDermottHealthAI/MedRAP
# feat/anchor-strategy-uniform-event). pyproject.toml's current pin,
# MedRAP@0a4a1fc, PREDATES it -- and the preprocess config is a structured
# dataclass, so an unknown anchor_strategy key is a hard Hydra error, not a
# silent ignore. Bump the pin and re-run `uv sync` before submitting this.
#
# Outputs for index i land in a SEPARATE directory tree, so this never
# overwrites data/tasks/n<N> (which every other script in this directory
# reads from):
#   data/tasks_anchored/n<N>/tasks/{train,tuning,held_out}.parquet
#
# mem=64G: same OOM sizing rationale as generate_labels_n1248.sh (peak RSS
# tracks the full train-split size read eagerly by task_generation.py).
#
# Usage:
#   cd mimic_iv_sweep
#   sbatch scripts/generate_labels_anchored_n1248.sh
#   sbatch --array=2 scripts/generate_labels_anchored_n1248.sh   # only N=4
# ============================================================

#SBATCH --job-name=marginalized-gen-labels-anchored
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

# NOTE: task codes/labels must be sampled from the *filtered* population that
# training actually sees (min_subjects_per_code/min_events_per_subject
# already applied), not the raw MEDS_cohort/data root -- see
# McDermottHealthAI/medrap-experiments@dba9672. It also matters for the
# anchoring above: MEDS_cohort/intermediate is the cohort that carries the
# TIMELINE// rows described in the header.
MEDS_DATA_DIR="/groups/mm6677_gp/data/MIMIC_MEDS/MEDS_cohort/intermediate"
TENSORIZED_DIR="/groups/mm6677_gp/data/MIMIC_MEDS/MEDS_cohort/processed"
OUTPUT_DIR="${REPO_DIR}/data/tasks_anchored/n${N}"

cd "${REPO_DIR}"
# shellcheck source=/dev/null
source "${VENV}"
mkdir -p logs "${OUTPUT_DIR}"

echo "=== Job info ==="
echo "  Array job      : ${SLURM_ARRAY_JOB_ID:-local}[${IDX}]"
echo "  Node           : ${SLURMD_NODENAME:-$(hostname)}"
echo "  N (tasks)      : ${N}"
echo "  Anchor strategy: uniform_event"
echo "  Output         : ${OUTPUT_DIR}"
echo "  Started        : $(date)"
echo ""

medrap-preprocess \
    "meds_data_dir=${MEDS_DATA_DIR}" \
    "tensorized_dir=${TENSORIZED_DIR}" \
    "output_dir=${OUTPUT_DIR}" \
    "num_tasks=${N}" \
    horizon_days=7.0 \
    min_history_days=1.0 \
    seed=42 \
    anchor_strategy=uniform_event \
    do_overwrite=true \
    "$@"

# Gate the labels we just wrote. This is a CHECK, not a filter: it fails the
# run, it never drops the offending tasks and reports the rest. Under
# `set -euo pipefail` a non-zero exit here aborts the job -- which is the
# point: a broken label config costs a few CPU-minutes here instead of hours
# of GPU time in the sweep that reads these labels.
#
# --min-positives 100 is a REAL floor, unlike the --min-positives 1 every
# uniform_lifetime script in this directory has to settle for. It is
# affordable only because of anchor_strategy=uniform_event: at a median
# in-window positive rate of ~0.35, 100 positives is a small fraction of a
# percent of any split, so a task that cannot clear it has not been unlucky
# -- the anchoring did not do what this script exists to do, and an AUROC
# computed on that task would be noise. If this fails, fix the label config
# and regenerate; do not lower the floor to make the run go green.
python scripts/check_task_balance.py "${OUTPUT_DIR}/tasks" --min-positives 100 --quiet

echo ""
echo "=== Done: $(date) ==="
echo "Labels saved to ${OUTPUT_DIR}/tasks"
