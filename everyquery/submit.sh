#!/bin/bash
# Submit design.md's stages to SLURM.
#
#   everyquery/submit.sh data      preprocess, then labels for draws 1-3 and the evaluation grid
#   everyquery/submit.sh docs      encode the retrieval database (one GPU job)
#   everyquery/submit.sh pilot     one Stage 0 run (base, lr 2e-4, draw 1) to time training
#   everyquery/submit.sh stage0    Stage 0: 3 sizes x 2 learning rates x 3 draws, patient only
#
# Runs with predictions already written, and jobs already queued, are skipped.

set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs
QUEUED=$(squeue -u "${USER}" -h -o "%j") || { echo "squeue failed; try again" >&2; exit 1; }
SLURM=(--account=mm6677_gp --output=logs/%x_%j.out --error=logs/%x_%j.err)
CPU=(--partition=cpu --cpus-per-task=16 --mem=128G --time=24:00:00)
GPU=(--partition=gpu,h100 --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=24:00:00)

queued() { grep -qx "$1" <<<"${QUEUED}"; }

submit_train() {  # CONFIG SIZE LR DRAW
    local name="eq-$1-$2-lr$3-draw$4"
    [ ! -f "outputs/$1_$2_lr$3/draw$4/predictions.parquet" ] || return 0
    # Without the data every job fails at loading; stop instead of filling the queue.
    for need in data/cohort/metadata/codes.parquet "data/labels/draw$4/train" "data/labels/draw$4/tuning" data/eval_grid/eval; do
        [ -e "${need}" ] || { echo "missing ${need}; run ./submit.sh data first" >&2; exit 1; }
    done
    queued "${name}" || sbatch "${SLURM[@]}" "${GPU[@]}" --job-name="${name}" --wrap="./run.sh train $*"
}

case "${1:-}" in
    data)
        if [ -f data/cohort/metadata/codes.parquet ]; then
            after=()
        else
            queued eq-preprocess && { echo "eq-preprocess already queued" >&2; exit 0; }
            pre=$(sbatch --parsable "${SLURM[@]}" "${CPU[@]}" --job-name=eq-preprocess --wrap="./run.sh preprocess")
            after=(--dependency="afterok:${pre}")
        fi
        for draw in 1 2 3; do
            queued "eq-labels-draw${draw}" || sbatch "${SLURM[@]}" "${CPU[@]}" ${after[@]+"${after[@]}"} \
                --job-name="eq-labels-draw${draw}" --wrap="./run.sh labels ${draw}"
        done
        queued eq-eval-grid || sbatch "${SLURM[@]}" "${CPU[@]}" ${after[@]+"${after[@]}"} \
            --job-name=eq-eval-grid --wrap="./run.sh eval_grid"
        ;;
    docs)
        # FAISS-GPU is not used here (CPU index build); the Qwen3 encoder runs on any GPU.
        queued eq-encode-docs || sbatch "${SLURM[@]}" "${GPU[@]}" --time=06:00:00 \
            --job-name=eq-encode-docs --wrap="./run.sh encode_docs"
        ;;
    pilot)
        submit_train patient_only base 2e-4 1
        ;;
    stage0)
        for size in small base large; do
            for lr in 2e-4 5e-4; do
                for draw in 1 2 3; do
                    submit_train patient_only "${size}" "${lr}" "${draw}"
                done
            done
        done
        ;;
    *)
        echo "usage: $0 data | docs | pilot | stage0" >&2
        exit 1
        ;;
esac
