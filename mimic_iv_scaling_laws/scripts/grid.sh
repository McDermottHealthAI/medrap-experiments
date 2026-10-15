# Scaling-law grid, sourced by submit_tasks.sh, generate_tasks.sh and submit_sweep.sh. Edit here only.
ARMS=(patient_only retrieval)
DIMS=(4 8 16 32 64 128)
N_PATIENTS=(1000 3000 10000 30000 100000)  # train.parquet has one row per patient (~153k at full size)
TASK_COUNTS=(5 25 125 625 2500)              # random task codes per draw; same levels as mimic_iv_sweep's task-count ablation
DRAW_SEEDS=(101 202 303 404 505)           # medrap-preprocess seed for draw 1..5
