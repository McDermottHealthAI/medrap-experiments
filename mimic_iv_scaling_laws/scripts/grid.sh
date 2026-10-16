# Scaling-law grid, sourced by submit_tasks.sh, generate_tasks.sh and submit_sweep.sh. Edit here only.
ARMS=(patient_only retrieval)
DIMS=(4 8 16 32 64 128)
N_PATIENTS=(1000 3000 10000 30000 100000)  # train.parquet has one row per patient (~153k at full size)
DRAW_SEEDS=(101 202 303 404 505)           # medrap-preprocess seed for draw 1..5

# Main grid: every ARMS x DIMS x N_PATIENTS cell at this many tasks.
MAIN_NUM_TASKS=25
# Task-count ablation: every ARMS x N_PATIENTS cell at ABLATION_DIM for each of these task
# counts (MAIN_NUM_TASKS at ABLATION_DIM comes from the main grid).
ABLATION_TASK_COUNTS=(5 125 625 2500)
ABLATION_DIM=32

TASK_COUNTS=("${MAIN_NUM_TASKS}" "${ABLATION_TASK_COUNTS[@]}")  # every task count that needs labels
