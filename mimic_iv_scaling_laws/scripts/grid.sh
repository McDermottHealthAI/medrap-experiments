# Scaling-law grid, sourced by submit_tasks.sh, generate_tasks.sh and submit_sweep.sh. Edit here only.
ARMS=(patient_only retrieval)
DIMS=(4 8 16 32 64 128)
N_PATIENTS=(1000 3000 10000 30000 100000)  # train.parquet has one row per patient (~153k at full size)
DRAW_SEEDS=(101 202 303 404 505)           # medrap-preprocess seed for draw 1..5

# Main grid: every ARMS x DIMS x N_PATIENTS cell at each of these task counts. 25 matches the
# earlier N=25 results; 625 has ~3-5x less task-sampling noise across draws.
MAIN_TASK_COUNTS=(25 625)
# Task-count ablation: every ARMS x N_PATIENTS cell at ABLATION_DIM for each of these task
# counts (25 and 625 at ABLATION_DIM come from the main grid).
ABLATION_TASK_COUNTS=(5 125 2500)
ABLATION_DIM=32
# Knowledge-vs-capacity control: the retrieval model trained and tested on documents with no
# knowledge (random corpus docs / the content-free corpus), on a slice of the main grid.
CONTROL_ARMS=(retrieval_random_docs retrieval_null_docs)
CONTROL_DIMS=(4 16 64)
CONTROL_N_PATIENTS=(1000 10000 100000)
CONTROL_NUM_TASKS=625

TASK_COUNTS=(5 25 125 625 2500)  # every task count that needs labels

# v2 grid (head bias initialized from base rates; MedRAP#119 via v2env/; results in outputs_v2/).
# Capacity: the 1-layer/1-head widths plus two realistic encoders (WIDTHxLAYERSxHEADS); data:
# the five sizes plus the full train split (153,220 patients in every draw).
V2_CAPACITIES=(4 8 16 32 64 128 128x2x4 256x4x8)
V2_N_PATIENTS=(1000 3000 10000 30000 100000 153220)
V2_NUM_TASKS=625
V2_ARMS=(patient_only retrieval_null_docs retrieval_qwen3)
