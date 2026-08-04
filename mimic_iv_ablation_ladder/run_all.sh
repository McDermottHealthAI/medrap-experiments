#!/bin/bash
# ============================================================
# Driver: submit the whole ablation ladder for ONE code selection and
#         return in seconds. Login node, NOT an sbatch script.
# ------------------------------------------------------------
# Every array is submitted with --dependency links, so SLURM sequences the
# pipeline and aggregation is the last job -- RESULTS.md lands on disk
# without anyone watching.
#
#   prepare_retrieval.sh              GPU, up to 3h, the long pole
#   generate_labels_<selection>.sh    CPU array 0-7, ~1 min each
#   build_null_retrieval_db.py        inline, seconds, no SLURM job
#   sweep_patient_only_ladder.sh      train arm 1   afterok labels
#   sweep_marginalized_binary_ladder.sh  train arm 2  afterok labels + corpus
#   eval_ladder_patient_only.sh       column 1      afterany
#   eval_ladder_real_docs.sh          column 2      afterany
#   eval_ladder_random_docs.sh        column 3      afterany, 3x the range
#   eval_ladder_null_docs.sh          column 4      afterany
#   aggregate_ladder.sh               RESULTS.md    afterany on all four
#
# THE ARRAY RANGE IS THE SELECTION. Every ladder script derives
# SEL_IDX = IDX / 40 (or IDX / 120 for random_docs) from SELECTIONS=(random
# frequent), so --selection just picks the half of the range to submit:
#
#   selection   80-task arrays   240-task array
#   random      0-39             0-119
#   frequent    40-79            120-239
#
# Same scripts, run twice. The plan's execution order is random first,
# most_frequent second: that halves time-to-first-table and costs no
# statistical power, since all 40 paired cells are retained either way.
#
# Four dependency choices are deliberate:
#
#   * afterok on the LABELS. check_task_balance.py --min-positives 25 is a
#     real gate; if it fails, nothing downstream runs. Bad labels should
#     stop the sweep, not produce a table from them.
#   * afterok on the retrieval corpus for the marginalized arm only.
#     patient_only has no retriever, so it does not wait on the 3h embed.
#   * afterany on the EVALS, not aftercorr. aftercorr needs matching array
#     indices and eval_ladder_random_docs has 120 against training's 40, so
#     it would mis-map. afterany plus each eval script's
#     [ ! -f "${CHECKPOINT_PATH}" ] guard means a dead training cell is a
#     clean skip, not a corrupt row.
#   * afterany on aggregation, so a table is produced from whatever
#     completed -- safe only because RESULTS.md opens with a coverage block
#     that names every missing (seed, N).
#
# %12 throttle throughout, and it is not a knob worth turning. Measured on
# this account: 1 job alone = 32 batch/s, 10 jobs = 117 batch/s aggregate,
# 36 jobs = 134 batch/s. Going 36-wide buys +15% and triples exposure to the
# 8h --time wall (a past array lost 10 of 20 tasks that way).
#
# Re-running is safe. The retrieval corpus, the labels and the null corpus
# are each skipped when they already exist, so a second invocation -- for
# the other selection, or after a partial failure -- resubmits only the
# training/eval/aggregation stages.
#
# Usage:
#   cd mimic_iv_ablation_ladder
#   uv sync && mkdir -p logs
#   ./run_all.sh --selection random      # then walk away
#   ./run_all.sh --selection frequent    # after the first table lands
# ============================================================

set -euo pipefail

SELECTION="random"
THROTTLE=12

usage() {
    echo "usage: ./run_all.sh [--selection {random,frequent}]"
    echo ""
    echo "  --selection random     arrays 0-39   / 0-119     (default)"
    echo "  --selection frequent   arrays 40-79  / 120-239"
}

while [ $# -gt 0 ]; do
    case "$1" in
        --selection)
            if [ $# -lt 2 ]; then
                echo "ERROR: --selection needs a value (random or frequent)." >&2
                exit 1
            fi
            SELECTION="$2"
            shift 2
            ;;
        --selection=*)
            SELECTION="${1#*=}"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

case "${SELECTION}" in
    random)
        RANGE_80="0-39"
        RANGE_240="0-119"
        ;;
    frequent)
        RANGE_80="40-79"
        RANGE_240="120-239"
        ;;
    *)
        echo "ERROR: --selection must be random or frequent (got '${SELECTION}')." >&2
        exit 1
        ;;
esac

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${REPO_DIR}/.venv/bin/activate"  # created by: cd mimic_iv_ablation_ladder && uv sync
RETRIEVAL_DB="${REPO_DIR}/data/retrieval_db"
NULL_DB="${REPO_DIR}/data/retrieval_db_null"
TENSORIZED_DIR="/groups/mm6677_gp/data/MIMIC_MEDS/MEDS_cohort/processed"
LABELS_DIR="${REPO_DIR}/data/tasks_${SELECTION}"
SCRIPTS_DIR="${REPO_DIR}/scripts"

cd "${REPO_DIR}"
mkdir -p logs "${REPO_DIR}/data"

# logs/ must exist BEFORE sbatch -- SLURM opens the log file before the job body runs, and
# a missing directory fails the task with no output to explain it.
if [ ! -f "${VENV}" ]; then
    echo "ERROR: ${VENV} not found. Run 'cd ${REPO_DIR} && uv sync' first." >&2
    exit 1
fi

if [ ! -d "${TENSORIZED_DIR}" ]; then
    echo "ERROR: ${TENSORIZED_DIR} not found. This is the lab-shared MEDS cohort -- check the mount." >&2
    exit 1
fi

NS=(1 2 4 8 16 32 64 128)
REQUIRED=(
    "scripts/prepare_retrieval.sh"
    "scripts/build_null_retrieval_db.py"
    "scripts/generate_labels_${SELECTION}.sh"
    "scripts/check_task_balance.py"
    "scripts/sweep_patient_only_ladder.sh"
    "scripts/sweep_marginalized_binary_ladder.sh"
    "scripts/eval_ladder_patient_only.sh"
    "scripts/eval_ladder_real_docs.sh"
    "scripts/eval_ladder_random_docs.sh"
    "scripts/eval_ladder_null_docs.sh"
    "scripts/aggregate_ladder.sh"
    "scripts/aggregate_ladder.py"
    "scripts/aggregate_results.py"
)
for SCRIPT in "${REQUIRED[@]}"; do
    if [ ! -f "${REPO_DIR}/${SCRIPT}" ]; then
        echo "ERROR: ${REPO_DIR}/${SCRIPT} not found. The experiment directory is incomplete." >&2
        exit 1
    fi
done

echo "=== Ablation ladder: submitting selection=${SELECTION} ==="
echo "  Repo           : ${REPO_DIR}"
echo "  80-task arrays : ${RANGE_80}%${THROTTLE}"
echo "  240-task array : ${RANGE_240}%${THROTTLE}"
echo "  Labels         : ${LABELS_DIR}"
echo "  Retrieval DB   : ${RETRIEVAL_DB}"
echo "  Null DB        : ${NULL_DB}"
echo "  Started        : $(date)"
echo ""

# --- stage 1: the retrieval corpus (one-time, independent of labels) ---------------
# Submitted in parallel with the labels rather than ahead of them: it is the long pole
# (up to 3h embedding 125,847 textbook chunks) and only the marginalized arm waits on it.
DEP_RET=()
if [ -d "${RETRIEVAL_DB}" ]; then
    echo "  [1/8] retrieval corpus : SKIPPED -- ${RETRIEVAL_DB} already exists"
else
    J_RET=$(sbatch --parsable "${SCRIPTS_DIR}/prepare_retrieval.sh")
    DEP_RET=(":${J_RET}")
    echo "  [1/8] retrieval corpus : ${J_RET}  scripts/prepare_retrieval.sh"
fi

# --- stage 2: task labels for this selection ---------------------------------------
LABELS_MISSING=0
for N in "${NS[@]}"; do
    if [ ! -d "${LABELS_DIR}/n${N}/tasks" ]; then
        LABELS_MISSING=1
    fi
done

DEP_LAB=()
if [ "${LABELS_MISSING}" -eq 0 ]; then
    echo "  [2/8] task labels      : SKIPPED -- ${LABELS_DIR}/n{1..128}/tasks already exist"
else
    J_LAB=$(sbatch --parsable "${SCRIPTS_DIR}/generate_labels_${SELECTION}.sh")
    DEP_LAB=(":${J_LAB}")
    echo "  [2/8] task labels      : ${J_LAB}  scripts/generate_labels_${SELECTION}.sh"
fi

# --- stage 3: the null corpus (inline -- 16 rows, seconds, no SLURM job) ------------
if [ -d "${NULL_DB}" ]; then
    echo "  [3/8] null corpus      : SKIPPED -- ${NULL_DB} already exists"
else
    # shellcheck source=/dev/null
    source "${VENV}"
    python "${SCRIPTS_DIR}/build_null_retrieval_db.py" --output-dir "${NULL_DB}"
    echo "  [3/8] null corpus      : built inline at ${NULL_DB}"
fi

# --- stages 4-5: training -----------------------------------------------------------
# patient_only needs labels only; marginalized needs labels AND the retrieval corpus.
DEP_PO=()
if [ ${#DEP_LAB[@]} -gt 0 ]; then
    DEP_PO=(--dependency="afterok${DEP_LAB[0]}")
fi
J_PO=$(sbatch --parsable "${DEP_PO[@]}" --array="${RANGE_80}%${THROTTLE}" \
    "${SCRIPTS_DIR}/sweep_patient_only_ladder.sh")
echo "  [4/8] train patient_only : ${J_PO}  array=${RANGE_80}%${THROTTLE}"

MB_DEPS=""
if [ ${#DEP_LAB[@]} -gt 0 ]; then
    MB_DEPS="${MB_DEPS}${DEP_LAB[0]}"
fi
if [ ${#DEP_RET[@]} -gt 0 ]; then
    MB_DEPS="${MB_DEPS}${DEP_RET[0]}"
fi
DEP_MB=()
if [ -n "${MB_DEPS}" ]; then
    DEP_MB=(--dependency="afterok${MB_DEPS}")
fi
J_MB=$(sbatch --parsable "${DEP_MB[@]}" --array="${RANGE_80}%${THROTTLE}" \
    "${SCRIPTS_DIR}/sweep_marginalized_binary_ladder.sh")
echo "  [5/8] train marginalized : ${J_MB}  array=${RANGE_80}%${THROTTLE}"

# --- stages 6-7: the four eval columns ---------------------------------------------
J_E1=$(sbatch --parsable --dependency="afterany:${J_PO}" --array="${RANGE_80}%${THROTTLE}" \
    "${SCRIPTS_DIR}/eval_ladder_patient_only.sh")
J_E2=$(sbatch --parsable --dependency="afterany:${J_MB}" --array="${RANGE_80}%${THROTTLE}" \
    "${SCRIPTS_DIR}/eval_ladder_real_docs.sh")
J_E3=$(sbatch --parsable --dependency="afterany:${J_MB}" --array="${RANGE_240}%${THROTTLE}" \
    "${SCRIPTS_DIR}/eval_ladder_random_docs.sh")
J_E4=$(sbatch --parsable --dependency="afterany:${J_MB}" --array="${RANGE_80}%${THROTTLE}" \
    "${SCRIPTS_DIR}/eval_ladder_null_docs.sh")
echo "  [6/8] eval col 1 patient_only : ${J_E1}  array=${RANGE_80}%${THROTTLE}"
echo "  [6/8] eval col 2 real_docs    : ${J_E2}  array=${RANGE_80}%${THROTTLE}"
echo "  [7/8] eval col 3 random_docs  : ${J_E3}  array=${RANGE_240}%${THROTTLE}  (3 reps)"
echo "  [7/8] eval col 4 null_docs    : ${J_E4}  array=${RANGE_80}%${THROTTLE}"

# --- stage 8: aggregation ------------------------------------------------------------
J_AGG=$(sbatch --parsable --dependency="afterany:${J_E1}:${J_E2}:${J_E3}:${J_E4}" \
    "${SCRIPTS_DIR}/aggregate_ladder.sh" "${SELECTION}")
echo "  [8/8] aggregate RESULTS.md    : ${J_AGG}  scripts/aggregate_ladder.sh ${SELECTION}"

echo ""
echo "=== Submitted: $(date) ==="
echo "Watch it:"
echo "  squeue -u \$USER            # empty == done"
echo "Read the table when job ${J_AGG} finishes:"
echo "  cat ${REPO_DIR}/RESULTS.md              # also kept as RESULTS_${SELECTION}.md"
echo ""
FIRST_METRICS="${REPO_DIR}/outputs/marginalized_binary_ladder/${SELECTION}/seed1001/n1/loggers/csv/version_0/metrics.csv"
echo "WATCH THE FIRST MARGINALIZED JOB. If retrieval/train/differentiable/effective_k_mean"
echo "is still ~1.0 once it has logged a validation pass, fix 1"
echo "(marginalized_score_similarity=cosine) did not take, the retrieval arm is collapsed,"
echo "and C2/C3 are meaningless -- cancel and investigate rather than spending the window:"
echo "  python -c \"import pandas,sys;print(pandas.read_csv(sys.argv[1])['retrieval/train/differentiable/effective_k_mean'].dropna().tail())\" \\"
echo "      ${FIRST_METRICS}"
