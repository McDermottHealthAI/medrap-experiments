#!/usr/bin/env bash
# ============================================================
# SLURM: filter the shared MIMIC-IV MEDS cohort, then tensorize it.
# ------------------------------------------------------------
# New for the post-refactor structure: mimic_iv/ never tensorized its
# own data (it read a lab-shared, pre-tensorized cohort directly). This
# script exercises the new `medrap-preprocess` rare-code/sparse-subject
# filtering stage before handing off to meds_torchdata's MTD_preprocess,
# so this experiment owns its own tensorized cohort end-to-end.
#
# Reads from: /groups/mm6677_gp/data/MIMIC_MEDS/MEDS_cohort  (read-only)
# Writes to:  data/MEDS_cohort_filtered, then data/tensorized  (this repo)
#
# Usage:
#   sbatch scripts/prepare_meds_data.sh
#   sbatch scripts/prepare_meds_data.sh min_subjects_per_code=50
#
# Extra arguments are forwarded to `medrap-preprocess` as Hydra overrides.
# ============================================================

#SBATCH --job-name=medrap-smoke-prep-meds
#SBATCH --partition=cpu
#SBATCH --account=mm6677_gp
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR}"
cd "${REPO_DIR}"
mkdir -p logs

MEDS_COHORT_DIR="/groups/mm6677_gp/data/MIMIC_MEDS/MEDS_cohort"
FILTERED_DIR="${REPO_DIR}/data/MEDS_cohort_filtered"
TENSORIZED_DIR="${REPO_DIR}/data/tensorized"

MIN_SUBJECTS_PER_CODE=25
MIN_EVENTS_PER_SUBJECT=5

echo "=== Job info ==="
echo "  Job ID  : ${SLURM_JOB_ID:-local}"
echo "  Node    : ${SLURMD_NODENAME:-$(hostname)}"
echo "  Started : $(date)"
echo ""

echo "=== Filtering rare codes / sparse subjects ==="
echo "  MEDS cohort : ${MEDS_COHORT_DIR}  (read-only)"
echo "  Output      : ${FILTERED_DIR}"

medrap-preprocess \
    meds_data_dir="${MEDS_COHORT_DIR}" \
    output_dir="${FILTERED_DIR}" \
    min_subjects_per_code="${MIN_SUBJECTS_PER_CODE}" \
    min_events_per_subject="${MIN_EVENTS_PER_SUBJECT}" \
    do_overwrite=true \
    "$@"

echo ""
echo "=== Tensorizing filtered cohort ==="
echo "  Input  : ${FILTERED_DIR}"
echo "  Output : ${TENSORIZED_DIR}"

MTD_preprocess \
    MEDS_dataset_dir="${FILTERED_DIR}" \
    output_dir="${TENSORIZED_DIR}"

echo ""
echo "=== Done: $(date) ==="
echo "Tensorized cohort: ${TENSORIZED_DIR}"
