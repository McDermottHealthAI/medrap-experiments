# Scaling-law grid, sourced by make_labels.sh and submit_sweep.sh. Edit here only.
ARMS=(patient_only retrieval)
DIMS=(4 8 16 32 64 128)
N_PATIENTS=(1000 3000 10000 30000 100000)  # train.parquet has one row per patient (~153k at full size)
