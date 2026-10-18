#!/bin/bash
# v2 pilot: every arm with the head bias initialized from the training base rates
# (MedRAP#119, via v2env/), results in outputs_v2/. Answers, before the full v2 grid:
#   1. does the bias fix close patient_only's small-model gap to the null-docs arm?
#   2. is qwen3_text retrieval patient-specific, and does it beat null docs?
#
# Usage:  cd mimic_iv_scaling_laws && (cd v2env && uv sync) && scripts/submit_v2_pilot.sh

set -euo pipefail
export SL_VENV="${PWD}/v2env/.venv" SL_OUTPUT_ROOT=outputs_v2 SL_BIAS_INIT=1
T=625
for ARM in patient_only retrieval_null_docs retrieval retrieval_qwen3; do
    for DIM in 4 64; do
        for N in 1000 10000 100000; do
            missing=()
            for IDX in 0 1 2 3 4; do
                compgen -G "${SL_OUTPUT_ROOT}/${ARM}_dim${DIM}_trainN${N}_tasks${T}_test/draw$((IDX + 1))/loggers/csv/*/metrics.csv" \
                    >/dev/null || missing+=("${IDX}")
            done
            [ "${#missing[@]}" -eq 0 ] && continue
            extra=()
            [ "${ARM}" = retrieval_qwen3 ] && extra=(--time=24:00:00 --mem=64G)
            sbatch "${extra[@]}" --array="$(IFS=,; echo "${missing[*]}")" \
                --job-name="sl-v2-${ARM}-dim${DIM}-trainN${N}-tasks${T}" scripts/train.sh "${ARM}" "${DIM}" "${N}" "${T}"
        done
    done
done
