#!/bin/bash
# ============================================================
# Rebuild the MIMIC-IV cohort from the raw MEDS data, owned by this experiment:
#   stage 1 (MEDS-transforms: min_subjects_per_code=100, min_events_per_subject=10)
#     -> data/cohort/intermediate
#   stage 2 (MTD_preprocess tokenize + tensorize)
#     -> data/cohort/tensorized
# Nothing is read from anyone else's processed/intermediate copies.
#
# medrap-preprocess always generates one task set after the pipeline; it lands
# in data/cohort/tasks and is not used (generate_tasks.sh makes the real ones).
#
# Usage:
#   cd mimic_iv_scaling_laws
#   sbatch scripts/build_cohort.sh   (or scripts/submit_tasks.sh, which chains everything)
# ============================================================

#SBATCH --job-name=sl-build-cohort
#SBATCH --partition=cpu
#SBATCH --account=mm6677_gp
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR}"
VENV="${REPO_DIR}/.venv/bin/activate"  # created by: cd mimic_iv_scaling_laws && uv sync

RAW_MEDS_DIR="/groups/mm6677_gp/data/MIMIC_MEDS/MEDS_cohort"  # raw, unfiltered MEDS root (data/ + metadata/)
COHORT_DIR="${REPO_DIR}/data/cohort"

cd "${REPO_DIR}"
# shellcheck source=/dev/null
source "${VENV}"
mkdir -p logs "${COHORT_DIR}"

# Work around a polars 1.30 parquet-read panic in tokenization; see the module docstring.
export PYTHONPATH="${REPO_DIR}/scripts/polars_patch${PYTHONPATH:+:${PYTHONPATH}}"

echo "=== Building cohort: ${RAW_MEDS_DIR} -> ${COHORT_DIR} ($(date)) ==="

medrap-preprocess \
    "meds_data_dir=${RAW_MEDS_DIR}" \
    "output_dir=${COHORT_DIR}" \
    min_subjects_per_code=100 \
    min_events_per_subject=10 \
    do_overwrite=true

echo "=== Done: $(date) ==="
