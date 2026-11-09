#!/bin/bash
# Submit scaling_laws_v2 runs (design.md §7) to SLURM: one array job (one task per draw) per cell.
#
#   scaling_laws_v2/submit.sh tasks    # T = 1,000 task labels for draws 1-3 (CPU); run first
#   scaling_laws_v2/submit.sh stage0   # patient-only search: 3 W x 3 lr x 2 dropout, D 153K (54 runs)
#   LR=... DROPOUT=... scaling_laws_v2/submit.sh stage1
#                                      # the ladder (design.md §7): 7 configs x 2 W x 2 D (84 runs)
#
# Runs whose tests already exist or that are already queued are skipped, so it is safe to resubmit.

set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs
QUEUED=$(squeue -u "${USER}" -h -o "%j") || { echo "squeue failed; try again" >&2; exit 1; }
case "${1:-}" in
    stage0 | stage1)
        for need in ../scaling_laws/data/cohort/tensorized/data ../scaling_laws/data/retrieval_db \
            data/tasks1000_D153220/draw3/tasks; do
            [ -e "${need}" ] || { echo "missing ${need}; nothing submitted" >&2; exit 1; }
        done
        ;;
esac

# c0406's GPU reports uncorrectable ECC errors (FAISS cannot allocate on it); override with EXCLUDE.
SLURM=(--account=${ACCOUNT:-mm6677_gp} --exclude="${EXCLUDE-c0406}" --cpus-per-task=8 --output=logs/%x_%A_%a.out --error=logs/%x_%A_%a.err)
DRAWS=(1 2 3)

submit_train() {  # CONFIG W D LR DROPOUT
    local config=$1 w=$2 d=$3 lr=$4 dropout=$5 missing=() last=test_randomK
    local run="outputs/${config}_W${w}_D${d}_lr${lr}_do${dropout}"
    case "${config}" in
        patient_only) last=test_last ;;
        *random_docs) last=test_topK ;;
    esac
    [ "${config}" = patient_only ] || run="${run}_K${K:-4}"
    local name="v2-${run#outputs/}"
    ! grep -qx "${name}" <<<"${QUEUED}" || return 0
    for draw in "${DRAWS[@]}"; do
        compgen -G "${run}/draw${draw}/${last}/loggers/csv/*/metrics.csv" >/dev/null || missing+=("${draw}")
    done
    [ "${#missing[@]}" -gt 0 ] || return 0
    sbatch "${SLURM[@]}" --time=12:00:00 --mem=32G --partition="${GPU_PARTITION:-gpu,h100}" --gres=gpu:1 \
        --array="$(IFS=,; echo "${missing[*]}")" --job-name="${name}" \
        --wrap="LR=${lr} DROPOUT=${dropout} K=${K:-4} ./run.sh train ${config} ${w} ${d} \${SLURM_ARRAY_TASK_ID}"
}

case "${1:-}" in
    tasks)
        missing=()
        for draw in "${DRAWS[@]}"; do
            [ -f "data/tasks1000_D153220/draw${draw}/tasks/train.parquet" ] || missing+=("${draw}")
        done
        [ "${#missing[@]}" -gt 0 ] || exit 0
        grep -qx v2-tasks <<<"${QUEUED}" || sbatch "${SLURM[@]}" --partition=cpu --mem=64G --time=06:00:00 \
            --array="$(IFS=,; echo "${missing[*]}")" --job-name=v2-tasks \
            --wrap="./run.sh tasks \${SLURM_ARRAY_TASK_ID}"
        ;;
    stage0)
        for w in 64x2x4 128x2x4 256x4x8; do
            for lr in 3e-4 1e-3 3e-3; do
                for dropout in 0 0.1; do
                    submit_train patient_only "${w}" 153220 "${lr}" "${dropout}"
                done
            done
        done
        ;;
    stage1)
        : "${LR:?set LR from Stage 0}" "${DROPOUT:?set DROPOUT from Stage 0}"
        for config in patient_only v1_retrieval v1_random_docs v1_masked retrieval random_docs retrieval_only; do
            for w in 64x2x4 256x4x8; do
                for d in 10000 153220; do
                    submit_train "${config}" "${w}" "${d}" "${LR}" "${DROPOUT}"
                done
            done
        done
        ;;
    *)
        echo "usage: $0 tasks | stage0 | stage1" >&2
        exit 1
        ;;
esac
