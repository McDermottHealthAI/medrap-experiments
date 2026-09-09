#!/bin/bash
# ============================================================
# SLURM array: regenerated N=25/30d base labels (v2) using a MedRAP commit
#              that includes BOTH degenerate-code rejection (#102, merged
#              to main) and the retrieval-score-scaling fix
#              (#113, fix/scale-marginalized-retrieval-dot-product, open --
#              this repo's venv is pinned directly to that branch's HEAD
#              commit, not merged main, per standing no-auto-merge policy).
# ------------------------------------------------------------
# Same params as generate_labels_zach_uniform_event_n25_30d.sh, except:
#   - meds_data_dir points at the rebuilt intermediate/ (the shared one at
#     /groups/mm6677_gp/data/MIMIC_MEDS/MEDS_cohort/intermediate was
#     deleted 2026-08-25; rebuilt at
#     /groups/mm6677_gp/hs3627/medrap-experiments/mimic_iv_sweep/rebuild_intermediate/intermediate
#     using the confirmed original filtering params).
#   - output_dir suffixed _v2 so the original (pre-fix) labels stay
#     available for comparison rather than being overwritten.
#
# Usage:
#   cd mimic_iv_sweep
#   sbatch scripts/generate_labels_zach_uniform_event_n25_30d_v2.sh
# ============================================================

#SBATCH --job-name=gen-labels-zach-uniform-event-n25-30d-v2
#SBATCH --array=0-4
#SBATCH --partition=cpu
#SBATCH --account=mm6677_gp
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

IDX=${SLURM_ARRAY_TASK_ID}

DRAW_SEEDS=(101 202 303 404 505)
SEED=${DRAW_SEEDS[$IDX]}
DRAW=$((IDX + 1))

REPO_DIR="${SLURM_SUBMIT_DIR}"
VENV="${REPO_DIR}/.venv/bin/activate"  # created by: cd mimic_iv_sweep && uv sync

MEDS_DATA_DIR="${REPO_DIR}/rebuild_intermediate/intermediate"
TENSORIZED_DIR="/groups/mm6677_gp/zzw2102/data/MIMIC_MEDS_restored/processed"
OUTPUT_DIR="${REPO_DIR}/data/tasks_zach_uniform_event_n25_30d_v2/draw${DRAW}"

cd "${REPO_DIR}"
# shellcheck source=/dev/null
source "${VENV}"
mkdir -p logs "${OUTPUT_DIR}"

echo "=== Job info ==="
echo "  Array job      : ${SLURM_ARRAY_JOB_ID:-local}[${IDX}]"
echo "  Node           : ${SLURMD_NODENAME:-$(hostname)}"
echo "  N (tasks)      : 25"
echo "  Duration       : 30d (fixed)"
echo "  Draw           : ${DRAW}/5 (seed=${SEED})"
echo "  Code selection : random"
echo "  Anchor strategy: uniform_event"
echo "  MedRAP commit  : includes degenerate-code rejection + retrieval-scaling fix"
echo "  Output         : ${OUTPUT_DIR}"
echo "  Started        : $(date)"
echo ""

medrap-preprocess \
    "meds_data_dir=${MEDS_DATA_DIR}" \
    "tensorized_dir=${TENSORIZED_DIR}" \
    "output_dir=${OUTPUT_DIR}" \
    num_tasks=25 \
    horizon_days=30.0 \
    min_history_days=1.0 \
    "seed=${SEED}" \
    anchor_strategy=uniform_event \
    do_overwrite=true \
    "$@"

echo ""
echo "=== Done: $(date) ==="
echo "Labels saved to ${OUTPUT_DIR}/tasks"
