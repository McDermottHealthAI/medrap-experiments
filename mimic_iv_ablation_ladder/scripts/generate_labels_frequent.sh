#!/bin/bash
# ============================================================
# SLURM array: Generate multi-task binary code-occurrence labels with
#              EVENT-ANCHORED prediction times (anchor_strategy=uniform_event)
#              and MOST-FREQUENT code selection (code_selection=most_frequent)
#              for N in {1, 2, 4, 8, 16, 32, 64, 128} -- the `frequent` half of
#              the ablation ladder's two-table label set.
# ------------------------------------------------------------
# This is the label source for SEL_IDX=1 in every training and eval script in
# this directory. Its sibling, generate_labels_random.sh, produces SEL_IDX=0.
# Run both before the sweep; the plan's execution order puts the `random` table
# first, so this array can be submitted second without losing any power.
#
# Two deltas from ../mimic_iv_sweep/scripts/generate_labels_n1248.sh, and both
# change what the labels MEAN:
#
# 1. anchor_strategy=uniform_event. The per-subject prediction anchor is now a
#    real clinical EVENT, drawn uniformly over that subject's own event
#    timestamps, so it always lands on a moment where something was actually
#    recorded. It is NOT a uniform draw over the subject's lifetime, which is
#    what every script in ../mimic_iv_sweep and ../mimic_iv_sweep_frequent uses
#    (the default anchor_strategy=uniform_lifetime draws uniformly over
#    *calendar time* in [first_event + min_history_days,
#    last_event - horizon_days]). Both strategies share that window; they
#    differ only in the measure sampled over it.
#
#    The calendar draw is wrong on the cohort medrap-preprocess actually reads.
#    MEDS_cohort/intermediate carries 211,488 TIMELINE// rows and TIMELINE//START
#    sits at the same timestamp as meds.birth_code, so first_event is
#    effectively BIRTH: a ~59-year anchor window laid over ~1 year of real
#    clinical activity, and the horizon after the anchor is usually empty.
#
#    Measured on a real MEDS_cohort/intermediate shard, most_frequent selection
#    -- i.e. exactly this script's configuration -- at a 7-day horizon: positive
#    rate 0.0022 -> 0.2057, a ~92x increase. (The `random` sibling gets a median
#    in-window positive rate of ~0.026 from the same change; the two selections
#    interact with the anchor far more strongly than expected, which is why the
#    ladder is reported as two tables rather than one pooled result.) Also
#    measured on that shard: 554/554 anchors land on a real clinical-event
#    timestamp, 0 at birth, minimum 18.5 years post-birth; and ~17% of
#    otherwise-eligible subjects are dropped because they have no clinical event
#    inside the window (uniform_event drops them rather than falling back to a
#    calendar anchor).
#
# 2. code_selection=most_frequent. Task codes are the N codes with the highest
#    distinct-subject count in the train split -- deterministic, and the seed is
#    ignored for selection. "Most frequent" ranks by distinct-subject count, not
#    event-row count (McDermottHealthAI/MedRAP@425a321). Note the consequence
#    for the statistics: these draws are NESTED across N (the N=8 codes are a
#    subset of the N=16 codes), so per-N results are correlated and pooling
#    across N overstates significance. The `random` sibling's draws are
#    independent across N (verified: N=8 intersect N=16 = empty), so only that
#    table supports pooling.
#
# NOT COMPARABLE TO ANY EXISTING data/tasks/ TREE. These labels ask a different
# question ("given a real clinical event, what happens in the next 7 days?")
# over a different subject population than data/tasks/ in ../mimic_iv_sweep,
# ../mimic_iv_sweep_frequent, or anywhere else -- including that directory's own
# most_frequent labels, which are uniform_lifetime-anchored. Positive rates,
# n_valid_tasks, and any AUROC computed on them do not belong on the same axis
# as a published uniform_lifetime run. Compare anchored to anchored only; a
# headline number mixing the two is measuring the anchor, not the model.
#
# Requires medrap with anchor_strategy support (McDermottHealthAI/MedRAP
# feat/anchor-strategy-uniform-event, pinned by SHA in pyproject.toml). The
# preprocess config is a structured dataclass, so an unknown anchor_strategy
# key is a hard Hydra struct error rather than a silent ignore: against an old
# pin this array fails immediately instead of quietly writing uniform_lifetime
# labels.
#
# Outputs for index i land in:
#   data/tasks_frequent/n<N>/tasks/{train,tuning,held_out}.parquet
#
# mem=64G: task_generation.py reads each split shard eagerly (pl.read_parquet,
# not a lazy scan) before filtering, so peak RSS tracks the full train-split
# size. 16G OOM'd at ~16.4GB RSS within 32s on the full MIMIC-IV cohort.
#
# Array index -> N (1-D; the 3-D SEL_IDX/SEED_IDX/N_IDX arithmetic in the
# training and eval scripts does NOT apply here -- the code selection is set by
# which of the two label scripts you submit, not by the index):
#   0 -> N=1
#   1 -> N=2
#   2 -> N=4
#   3 -> N=8
#   4 -> N=16
#   5 -> N=32
#   6 -> N=64
#   7 -> N=128
#
# Usage:
#   cd mimic_iv_ablation_ladder
#   sbatch scripts/generate_labels_frequent.sh
#   sbatch --array=2 scripts/generate_labels_frequent.sh   # only N=4
# ============================================================

#SBATCH --job-name=ladder-gen-labels-frequent
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
VENV="${REPO_DIR}/.venv/bin/activate"  # created by: cd mimic_iv_ablation_ladder && uv sync

# NOTE: task codes/labels must be sampled from the *filtered* population that
# training actually sees (min_subjects_per_code/min_events_per_subject already
# applied), not the raw MEDS_cohort/data root -- see
# McDermottHealthAI/medrap-experiments@dba9672. It also matters for the
# anchoring above: MEDS_cohort/intermediate is the cohort that carries the
# TIMELINE// rows described in the header.
MEDS_DATA_DIR="/groups/mm6677_gp/data/MIMIC_MEDS/MEDS_cohort/intermediate"
TENSORIZED_DIR="/groups/mm6677_gp/data/MIMIC_MEDS/MEDS_cohort/processed"
OUTPUT_DIR="${REPO_DIR}/data/tasks_frequent/n${N}"

cd "${REPO_DIR}"
# shellcheck source=/dev/null
source "${VENV}"
mkdir -p logs "${OUTPUT_DIR}"

echo "=== Job info ==="
echo "  Array job      : ${SLURM_ARRAY_JOB_ID:-local}[${IDX}]"
echo "  Node           : ${SLURMD_NODENAME:-$(hostname)}"
echo "  N (tasks)      : ${N}"
echo "  Code selection : most_frequent"
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
    code_selection=most_frequent \
    anchor_strategy=uniform_event \
    do_overwrite=true \
    "$@"

# Gate the labels we just wrote. This is a CHECK, not a filter: it fails the
# run, it never drops the offending tasks and reports the rest. Under
# `set -euo pipefail` a non-zero exit here aborts the job -- which is INTENDED.
# run_all.sh chains the training arrays off this one with --dependency=afterok,
# so a failed gate stops the sweep before any GPU time is spent instead of
# producing a 4-column table from labels that cannot support one. A broken
# label config costs a few CPU-minutes here rather than hours of GPU time.
#
# --min-positives 25 is a REAL floor, unlike the --min-positives 1 every
# uniform_lifetime script in the sibling directories has to settle for. It is
# affordable only because of anchor_strategy=uniform_event. The floor is held
# at 25 -- the same value generate_labels_random.sh uses -- so that both tables
# of the ladder are admitted under an identical rule; 25 was chosen to sit below
# the projected per-task minimum of ~38 positives under RANDOM selection, and at
# this script's ~0.206 in-window positive rate (~4,000 projected positives per
# task) it is a very wide margin indeed. Anything that fails it here is a
# genuine failure of the anchoring or of the label join, not bad luck.
#
# Do NOT copy the --min-positives 100 of
# ../mimic_iv_sweep/scripts/generate_labels_anchored_n1248.sh. That floor was
# justified by a most_frequent measurement while the script itself passes no
# code_selection and so runs the random default at ~8x lower prevalence -- it
# aborts the array AFTER label generation has already completed. If this gate
# fires, fix the label config and regenerate; do not lower the floor to make
# the run go green.
python scripts/check_task_balance.py "${OUTPUT_DIR}/tasks" --min-positives 25 --quiet

echo ""
echo "=== Done: $(date) ==="
echo "Labels saved to ${OUTPUT_DIR}/tasks"
