#!/bin/bash
# ============================================================
# SLURM array: Generate multi-task binary code-occurrence labels
#              for N in {1, 2, 4, 8, 16, 32, 64, 128} -- most-frequent-code variant of
#              ../mimic_iv_sweep/scripts/generate_labels_n1248.sh.
# ------------------------------------------------------------
# Identical to generate_labels_n1248.sh except code_selection=most_frequent:
# task codes are the N codes with the highest distinct-subject count in the
# train split (deterministic, seed ignored for selection), instead of a
# uniform-random draw. Prediction time is still sampled randomly per subject,
# same as the random-task variant -- only which codes are chosen as tasks
# differs.
#
# Motivation: mimic_iv_sweep's README notes only a handful of codes (e.g.
# Blood Pressure, Weight, BMI) have enough in-window positive volume for a
# stable per-task AUROC; the random sampler in generate_labels_n1248.sh
# overwhelmingly picks rare/degenerate codes instead (see
# McDermottHealthAI/MedRAP#92). This variant tests whether biasing task
# selection toward the most frequent codes gives usable positive rates.
# "Most frequent" ranks by distinct-subject count, not event-row count --
# see McDermottHealthAI/MedRAP@425a321.
#
# Requires medrap >= McDermottHealthAI/MedRAP@425a321 (code_selection support,
# pinned in pyproject.toml; not yet merged to MedRAP main).
#
# Outputs for index i land in:
#   data/tasks/n<N>/tasks/{train,tuning,held_out}.parquet
#
# mem=64G: same OOM sizing rationale as generate_labels_n1248.sh (peak RSS
# tracks the full train-split size read eagerly by task_generation.py).
#
# Usage:
#   cd mimic_iv_sweep_frequent
#   sbatch scripts/generate_labels_n1248_frequent.sh
#   sbatch --array=2 scripts/generate_labels_n1248_frequent.sh   # only N=4
# ============================================================

#SBATCH --job-name=marginalized-gen-labels-frequent
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

# Same population-matching rationale as ../mimic_iv_sweep/scripts/generate_labels_n1248.sh:
# task codes/labels must be sampled from the *filtered* population that training
# actually sees, not the raw MEDS_cohort/data root.
MEDS_DATA_DIR="/groups/mm6677_gp/data/MIMIC_MEDS/MEDS_cohort/intermediate"
TENSORIZED_DIR="/groups/mm6677_gp/data/MIMIC_MEDS/MEDS_cohort/processed"
OUTPUT_DIR="${REPO_DIR}/data/tasks/n${N}"

cd "${REPO_DIR}"
# shellcheck source=/dev/null
source "${VENV}"
mkdir -p logs "${OUTPUT_DIR}"

echo "=== Job info ==="
echo "  Array job     : ${SLURM_ARRAY_JOB_ID:-local}[${IDX}]"
echo "  Node          : ${SLURMD_NODENAME:-$(hostname)}"
echo "  N (tasks)     : ${N}"
echo "  Code selection: most_frequent"
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
    seed=42 \
    code_selection=most_frequent \
    do_overwrite=true \
    "$@"

echo ""
echo "=== Done: $(date) ==="
echo "Labels saved to ${OUTPUT_DIR}/tasks"
