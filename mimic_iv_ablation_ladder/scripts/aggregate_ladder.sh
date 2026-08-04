#!/bin/bash
# ============================================================
# CPU job: the final link in the dependency chain -- assemble the
#          4-column ablation-ladder AUROC table and write RESULTS.md.
# ------------------------------------------------------------
# Runs scripts/aggregate_ladder.py, which imports aggregate_results.py's run
# discovery, metrics.csv parsing, acceptance gates, cell keying and
# Student-t helpers, then drives three paired comparisons against the shared
# real_docs baseline:
#
#   C1  patient_only vs real_docs    does retrieval help at all?
#   C2  real_docs    vs random_docs  is retrieval SELECTING useful docs?
#   C3  real_docs    vs null_docs    do document CONTENTS contribute?
#
# run_all.sh submits this with --dependency=afterany on all four eval
# arrays, NOT afterok, so a table is produced from whatever completed. That
# is only safe because the report opens with an explicit coverage block:
# expected cells (40 per arm), cells found, and every missing (seed, N)
# listed by name. A table built from 12 of 40 cells says so at the top.
#
# CPU only, and small: it reads one metrics.csv per run (280 single-row
# files for a complete selection) and does arithmetic. No GPU, no model, no
# dataset. 30 minutes is roughly 30x what it needs.
#
# Takes the code selection as $1. The array range is the selection
# everywhere else in this experiment (SEL_IDX = IDX / 40), but this job is
# not an array, so it has to be told which one to report on. Any further
# arguments are forwarded to aggregate_ladder.py.
#
# Output:
#   RESULTS.md              -- the canonical path run_all.sh points at
#   RESULTS_<selection>.md  -- the same report, kept per selection
#
# Both are written every time. RESULTS.md alone would be destroyed by the
# second selection's run, and the two tables are meant to be read side by
# side (random draws are independent across N so pooling is sound;
# most_frequent is nested, so it is not).
#
# Prerequisites:
#   sbatch scripts/eval_ladder_patient_only.sh   (column 1)
#   sbatch scripts/eval_ladder_real_docs.sh      (column 2)
#   sbatch scripts/eval_ladder_random_docs.sh    (column 3)
#   sbatch scripts/eval_ladder_null_docs.sh      (column 4)
#   scripts/aggregate_results.py must sit beside aggregate_ladder.py --
#   it is imported by path, not by package name.
#
# Usage:
#   cd mimic_iv_ablation_ladder
#   sbatch scripts/aggregate_ladder.sh random
#   sbatch scripts/aggregate_ladder.sh frequent
#   sbatch scripts/aggregate_ladder.sh random --verbose   # + per-run tables
# ============================================================

#SBATCH --job-name=aggregate-ladder
#SBATCH --partition=cpu
#SBATCH --account=mm6677_gp
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

SELECTION=${1:-random}

REPO_DIR="${SLURM_SUBMIT_DIR}"
VENV="${REPO_DIR}/.venv/bin/activate"  # created by: cd mimic_iv_ablation_ladder && uv sync
AGGREGATOR="${REPO_DIR}/scripts/aggregate_ladder.py"
HELPERS="${REPO_DIR}/scripts/aggregate_results.py"
OUTPUTS_ROOT="${REPO_DIR}/outputs"
RESULTS_MD="${REPO_DIR}/RESULTS.md"
RESULTS_KEEP="${REPO_DIR}/RESULTS_${SELECTION}.md"

cd "${REPO_DIR}"
# shellcheck source=/dev/null
source "${VENV}"
mkdir -p logs

if [ ! -f "${AGGREGATOR}" ]; then
    echo "ERROR: ${AGGREGATOR} not found. Run sbatch scripts/aggregate_ladder.sh from the experiment root first." >&2
    exit 1
fi

if [ ! -f "${HELPERS}" ]; then
    echo "ERROR: ${HELPERS} not found. Copy it from ../mimic_iv_sweep_frequent/scripts/aggregate_results.py first." >&2
    exit 1
fi

if [ ! -d "${OUTPUTS_ROOT}" ]; then
    echo "ERROR: ${OUTPUTS_ROOT} not found. Run sbatch scripts/eval_ladder_real_docs.sh first." >&2
    exit 1
fi

echo "=== Job info ==="
echo "  Job id         : ${SLURM_JOB_ID:-local}"
echo "  Node           : ${SLURMD_NODENAME:-$(hostname)}"
echo "  Code selection : ${SELECTION}"
echo "  Outputs root   : ${OUTPUTS_ROOT}"
echo "  Split          : test (medrap-eval eval_mode=test, MEDS held_out)"
echo "  Rep component  : (seed|rep)\\d+ (collapses BOTH the seed and rep levels)"
echo "  RESULTS.md     : ${RESULTS_MD}"
echo "  Per-selection  : ${RESULTS_KEEP}"
echo "  Started        : $(date)"
echo ""

# The aggregator exits 1 when no arm produced a single run -- but it still writes the
# coverage block explaining that, which is the whole point of running under afterany. So
# the status is captured rather than allowed to abort the job under `set -e`: copy the
# report, print the footer, then exit with it.
STATUS=0
python "${AGGREGATOR}" \
    --selection "${SELECTION}" \
    --outputs-root "${OUTPUTS_ROOT}" \
    --split test \
    --rep-component '(seed|rep)\d+' \
    -o "${RESULTS_MD}" \
    "${@:2}" || STATUS=$?

if [ -f "${RESULTS_MD}" ]; then
    cp "${RESULTS_MD}" "${RESULTS_KEEP}"
fi

if [ "${STATUS}" -ne 0 ]; then
    echo "WARNING: aggregate_ladder.py exited ${STATUS} -- no runs were found for any arm." >&2
    echo "         ${RESULTS_MD} still holds the coverage block naming the globs it tried." >&2
fi

echo ""
echo "=== Done: $(date) ==="
echo "Table written to ${RESULTS_MD} (and ${RESULTS_KEEP})"
exit "${STATUS}"
