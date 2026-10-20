#!/bin/bash
# Bias-recalibration check: predict tuning/held_out probabilities for a slice of the 25-task
# main grid, for scripts/bias_check.py. Skips runs whose predictions already exist.
#
# Usage:  cd mimic_iv_scaling_laws && scripts/submit_bias_check.sh

set -euo pipefail
T=25
for ARM in patient_only retrieval; do
    for DIM in 4 16 128; do
        for N in 1000 10000 100000; do
            missing=()
            for IDX in 0 1 2 3 4; do
                [ -f "outputs/predictions/${ARM}_dim${DIM}_trainN${N}_tasks${T}/draw$((IDX + 1))/probabilities.parquet" ] \
                    || missing+=("${IDX}")
            done
            [ "${#missing[@]}" -eq 0 ] && continue
            sbatch --array="$(IFS=,; echo "${missing[*]}")" --job-name="sl-pred-${ARM}-dim${DIM}-trainN${N}-tasks${T}" \
                scripts/predict_probabilities.sh "${ARM}" "${DIM}" "${N}" "${T}"
        done
    done
done
