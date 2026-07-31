#!/bin/bash
# ============================================================
# SLURM array: Generate multi-task binary code-occurrence labels
#              for N in {1, 2, 4, 8, 16, 32, 64, 128} -- the marginalized-retrieval
#              random-task experiment (see sweep_marginalized_n1248.sh).
# ------------------------------------------------------------
# Submits one job per N value. Each job calls `medrap-preprocess` with the
# already-tensorized cohort (tensorized_dir), so only the task-generation
# stage runs (MEDS-transforms and MTD_preprocess are skipped).
#
# Requires medrap >= McDermottHealthAI/MedRAP#92 (pinned in pyproject.toml):
# task codes are sampled uniformly at random from the train split, with no
# positive-rate/count filtering -- a sampled code can turn out rare or
# degenerate (single-class) on a given split.
#
# Outputs for index i land in:
#   data/tasks/n<N>/tasks/{train,tuning,held_out}.parquet
#
# mem=64G: task_generation.py reads each split shard eagerly (pl.read_parquet,
# not a lazy scan) before filtering, so peak RSS tracks the full train-split
# size. 16G OOM'd at ~16.4GB RSS within 32s on the full MIMIC-IV cohort.
#
# Usage:
#   cd mimic_iv_sweep
#   sbatch scripts/generate_labels_n1248.sh
#   sbatch --array=2 scripts/generate_labels_n1248.sh   # only N=4
# ============================================================

#SBATCH --job-name=marginalized-gen-labels
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
# McDermottHealthAI/medrap-experiments@dba9672.
MEDS_DATA_DIR="/groups/mm6677_gp/data/MIMIC_MEDS/MEDS_cohort/intermediate"
TENSORIZED_DIR="/groups/mm6677_gp/data/MIMIC_MEDS/MEDS_cohort/processed"
OUTPUT_DIR="${REPO_DIR}/data/tasks/n${N}"

cd "${REPO_DIR}"
# shellcheck source=/dev/null
source "${VENV}"
mkdir -p logs "${OUTPUT_DIR}"

echo "=== Job info ==="
echo "  Array job : ${SLURM_ARRAY_JOB_ID:-local}[${IDX}]"
echo "  Node      : ${SLURMD_NODENAME:-$(hostname)}"
echo "  N (tasks) : ${N}"
echo "  Output    : ${OUTPUT_DIR}"
echo "  Started   : $(date)"
echo ""

medrap-preprocess \
    "meds_data_dir=${MEDS_DATA_DIR}" \
    "tensorized_dir=${TENSORIZED_DIR}" \
    "output_dir=${OUTPUT_DIR}" \
    "num_tasks=${N}" \
    horizon_days=7.0 \
    min_history_days=1.0 \
    seed=42 \
    do_overwrite=true \
    "$@"

# Gate the labels we just wrote. This is a CHECK, not a filter: it fails the
# run, it never drops the offending tasks and reports the rest. Under
# `set -euo pipefail` a non-zero exit here aborts the job -- which is the
# point: a broken label config costs a few CPU-minutes here instead of hours
# of GPU time in the sweep that reads these labels.
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
