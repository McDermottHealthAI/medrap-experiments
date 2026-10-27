#!/bin/bash
# Submit the scaling-law runs in design.md to SLURM: one array job (one task per draw) per cell.
#
#   scaling_laws/submit.sh tasks   # task labels for every T and draw (CPU); run first
#   scaling_laws/submit.sh pilot   # one small run of each configuration (W 4, D 1K, T 25, draw 1)
#   scaling_laws/submit.sh grid    # main grid (3,600 runs) + random-documents control (45 runs)
#
# Cells whose outputs already exist are skipped, so it is safe to resubmit after failures.

set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

SLURM=(--account=mm6677_gp --cpus-per-task=8 --output=logs/%x_%A_%a.out --error=logs/%x_%A_%a.err)
CONFIGS=(patient_only retrieval_learned_query retrieval_qwen3_query)
WS=(4 8 16 32 64 128 128x2x4 256x4x8)
DS=(1000 3000 10000 30000 100000 153220)
TS=(5 25 125 625 2500)

submit_train() {  # CONFIG W D T DRAW...
    local config=$1 w=$2 d=$3 t=$4 missing=() resources=(--time=06:00:00 --mem=32G)
    shift 4
    for draw in "$@"; do
        compgen -G "outputs/${config}_W${w}_D${d}_T${t}/draw${draw}_test/loggers/csv/*/metrics.csv" \
            >/dev/null || missing+=("${draw}")
    done
    [ "${#missing[@]}" -gt 0 ] || return 0
    case "${config}" in
        retrieval_qwen3_query) resources=(--time=24:00:00 --mem=64G) ;;  # frozen Qwen3 forward per patient
        retrieval_*) resources=(--time=12:00:00 --mem=32G) ;;
    esac
    sbatch "${SLURM[@]}" "${resources[@]}" --partition=gpu,h100 --gres=gpu:1 \
        --array="$(IFS=,; echo "${missing[*]}")" --job-name="sl-${config}-W${w}-D${d}-T${t}" \
        --wrap="./run.sh train ${config} ${w} ${d} ${t} \${SLURM_ARRAY_TASK_ID}"
}

case "${1:-}" in
    tasks)
        for t in "${TS[@]}"; do
            missing=()
            for draw in 1 2 3 4 5; do
                [ -f "data/tasks${t}_D153220/draw${draw}/tasks/train.parquet" ] || missing+=("${draw}")
            done
            [ "${#missing[@]}" -gt 0 ] || continue
            sbatch "${SLURM[@]}" --partition=cpu --mem=64G --time=04:00:00 \
                --array="$(IFS=,; echo "${missing[*]}")" --job-name="sl-tasks-T${t}" \
                --wrap="./run.sh tasks ${t} \${SLURM_ARRAY_TASK_ID}"
        done
        ;;
    pilot)
        for config in "${CONFIGS[@]}" retrieval_random_docs; do
            submit_train "${config}" 4 1000 25 1
        done
        ;;
    grid)
        for config in "${CONFIGS[@]}"; do
            for w in "${WS[@]}"; do
                for d in "${DS[@]}"; do
                    for t in "${TS[@]}"; do
                        submit_train "${config}" "${w}" "${d}" "${t}" 1 2 3 4 5
                    done
                done
            done
        done
        for w in 4 16 64; do
            for d in 1000 10000 100000; do
                submit_train retrieval_random_docs "${w}" "${d}" 625 1 2 3 4 5
            done
        done
        ;;
    *)
        echo "usage: $0 tasks | pilot | grid" >&2
        exit 1
        ;;
esac
