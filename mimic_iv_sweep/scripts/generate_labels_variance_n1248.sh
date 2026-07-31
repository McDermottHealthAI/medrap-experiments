#!/bin/bash
# ============================================================
# SLURM array: Generate multi-task binary code-occurrence labels for a
#              variance study -- N in {1, 2, 4, 8, 16, 32, 64, 128}, EACH
#              REPEATED FOR 5 INDEPENDENT RANDOM DRAWS of the N task codes
#              (code_selection=random, a different seed per draw). Answers:
#              "does patient_only vs. marginalized(binary)'s AUROC
#              difference hold up across different random samples of N
#              tasks, or is a single draw just noise?"
# ------------------------------------------------------------
# 40 array tasks = 8 N values x 5 draws, flattened as
# IDX = n_idx * 5 + draw_idx (n_idx in [0,7], draw_idx in [0,4]).
#
# Each draw uses its own seed (DRAW_SEEDS below) -- deliberately distinct
# from seed=42 used by every other script in this directory/mimic_iv_sweep_frequent,
# so none of these 5 draws accidentally coincides with (or is mistaken for)
# the existing single-draw sweeps.
#
# Same population-matching rationale as generate_labels_n1248.sh: task
# codes/labels are sampled from the *filtered* population training actually
# sees (MEDS_cohort/intermediate), not the raw MEDS_cohort/data root.
#
# Outputs land in a SEPARATE directory tree from every other label set in
# this repo, keyed by draw:
#   data/tasks_variance/draw<d>/n<N>/tasks/{train,tuning,held_out}.parquet
#
# mem=64G: same OOM sizing rationale as generate_labels_n1248.sh.
#
# Usage:
#   cd mimic_iv_sweep
#   sbatch scripts/generate_labels_variance_n1248.sh
#   sbatch --array=0-4 scripts/generate_labels_variance_n1248.sh    # only N=1, all 5 draws
#   sbatch --array=0,5,10,15,20,25,30,35 scripts/generate_labels_variance_n1248.sh  # only draw 0, all N
# ============================================================

#SBATCH --job-name=marginalized-gen-labels-variance
#SBATCH --array=0-39
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
DRAW_SEEDS=(101 202 303 404 505)

N_IDX=$((IDX / 5))
DRAW_IDX=$((IDX % 5))
N=${NS[$N_IDX]}
SEED=${DRAW_SEEDS[$DRAW_IDX]}
DRAW=$((DRAW_IDX + 1))

REPO_DIR="${SLURM_SUBMIT_DIR}"
VENV="${REPO_DIR}/.venv/bin/activate"  # created by: cd mimic_iv_sweep && uv sync

MEDS_DATA_DIR="/groups/mm6677_gp/data/MIMIC_MEDS/MEDS_cohort/intermediate"
TENSORIZED_DIR="/groups/mm6677_gp/data/MIMIC_MEDS/MEDS_cohort/processed"
OUTPUT_DIR="${REPO_DIR}/data/tasks_variance/draw${DRAW}/n${N}"

cd "${REPO_DIR}"
# shellcheck source=/dev/null
source "${VENV}"
mkdir -p logs "${OUTPUT_DIR}"

echo "=== Job info ==="
echo "  Array job     : ${SLURM_ARRAY_JOB_ID:-local}[${IDX}]"
echo "  Node          : ${SLURMD_NODENAME:-$(hostname)}"
echo "  N (tasks)     : ${N}"
echo "  Draw          : ${DRAW}/5 (seed=${SEED})"
echo "  Code selection: random"
echo "  Output        : ${OUTPUT_DIR}"
echo "  Started       : $(date)"
echo ""

medrap-preprocess \
    "meds_data_dir=${MEDS_DATA_DIR}" \
    "tensorized_dir=${TENSORIZED_DIR}" \
    "output_dir=${OUTPUT_DIR}" \
    "num_tasks=${N}" \
    horizon_days=7.0 \
    min_history_days=1.0 \
    "seed=${SEED}" \
    do_overwrite=true \
    "$@"

# Gate the labels we just wrote. This is a CHECK, not a filter: it fails the
# run, it never drops the offending tasks and reports the rest. Under
# `set -euo pipefail` a non-zero exit here aborts the job -- which is the
# point: a broken label config costs a few CPU-minutes here instead of hours
# of GPU time in the sweep that reads these labels. Per-draw, so one bad draw
# fails its own array task and leaves the other four alone.
#
# --min-positives 1 means "fail only on single-class tasks", where AUROC is
# undefined. The floor is deliberately that low because these labels use the
# default anchor_strategy=uniform_lifetime, whose median task lands ~2
# positives per split; a real floor (say 100) would fail every draw in this
# array. For labels that can carry a real floor, see
# scripts/generate_labels_anchored_n1248.sh.
python scripts/check_task_balance.py "${OUTPUT_DIR}/tasks" --min-positives 1 --quiet

echo ""
echo "=== Done: $(date) ==="
echo "Labels saved to ${OUTPUT_DIR}/tasks"
