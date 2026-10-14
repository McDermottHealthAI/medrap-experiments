#!/bin/bash
# Write the train-subsampled label dirs for every PCT x draw in grid.sh,
# from the full-size N=25 30d labels mimic_iv_sweep already generated.
# tuning/held_out are copied unchanged, so every grid point is tested on
# the same held_out split for a given draw.
#
# Usage:
#   cd mimic_iv_scaling_laws && scripts/make_labels.sh

set -euo pipefail
source scripts/grid.sh

SOURCE_ROOT="/groups/mm6677_gp/hs3627/medrap-experiments/mimic_iv_sweep/data/tasks_zach_uniform_event_n25_30d"

for PCT in "${PCTS[@]}"; do
    for DRAW in 1 2 3 4 5; do
        .venv/bin/python scripts/subsample_train_labels.py \
            "${SOURCE_ROOT}/draw${DRAW}/tasks" \
            "data/tasks_n25_30d_train${PCT}pct/draw${DRAW}/tasks" \
            --fraction "$(echo "scale=2; ${PCT} / 100" | bc)" \
            --seed "$((1000 * DRAW + PCT))"
    done
done
