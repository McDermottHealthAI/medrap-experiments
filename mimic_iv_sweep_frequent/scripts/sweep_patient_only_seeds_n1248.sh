#!/bin/bash
# ============================================================
# SLURM array: patient-only (no retrieval), N in {1, 2, 4, 8, 16, 32, 64, 128}
#              x 5 model seeds -- frequent-code task labels. Replicates
#              sweep_patient_only_n1248.sh across training stochasticity so
#              each N can be reported as a mean +/- CI over seeds instead of
#              a single unreplicated point.
# ------------------------------------------------------------
# Identical to sweep_patient_only_n1248.sh except one extra override,
# seed=${MODEL_SEED}, and the per-seed output_dir/wandb_run_name. Every
# other Hydra override is copied verbatim from that script, including the
# full-tuning-split validation protocol (training.trainer.limit_val_batches=1.0,
# training.trainer.val_check_interval=0.5), so a seed spread measured here is
# directly comparable to the existing single-seed n1248 numbers.
#
# Why this exists: every training script in this repo inherits `seed: 42`
# from medrap's conf/_train.yaml, and no script has ever overridden it. The
# entire N-sweep is therefore one draw from the training-noise distribution,
# and a bump or dip at some N cannot currently be distinguished from an
# unlucky seed. This script supplies the missing replication.
#
# What is being measured: `seed=<...>` reaches
# lightning.seed_everything(cfg.seed, workers=True) in medrap's cli.py
# train_main (line 429 in the currently pinned build). That single call seeds
# weight initialization, dropout AND dataloader order together -- Lightning's
# workers=True additionally reseeds each DataLoader worker. So the spread
# across these 5 runs is TOTAL training variance, not initialization variance
# alone. That is the right quantity for a mean +/- CI on "what does this
# architecture score at this N", but it should be stated plainly: this
# experiment cannot attribute the spread to init versus data order, and it is
# not the harder-to-vary quantity some seed studies report.
#
# Pairs with sweep_marginalized_binary_seeds_n1248.sh, which applies the same
# 5 seeds to the retrieval arm. Seeds are an ADDITION to, never a replacement
# for, the paired-over-draws design in ../mimic_iv_sweep: pairing already
# removes 87.7% of the noise in the arm difference (arms correlate 0.985; sd
# of the paired delta 0.0110 vs 0.0896 independent).
#
# MODEL_SEEDS=(1001 2002 3003 4004 5005) are deliberately distinct from the
# default 42 and from the label-draw seeds 101/202/303/404/505, matching this
# repo's rationale of keeping seed families disjoint.
#
# SCALE WARNING: this is 40 GPU jobs (5 seeds x 8 N values) for THIS arm
# alone, and the marginalized arm is another 40 -- 80 L40S jobs total for the
# pair, versus the 16 the two single-seed n1248 sweeps cost. Size it against
# ../mimic_iv_sweep/scripts/probe_seed_variance_n25.sh's sigma_seed estimate
# before submitting the whole array; the large-N tasks are the expensive ones,
# so `--array=0-39:8`-style subsets or a small-N-first submission are the
# cheap way to sanity-check the setup first.
#
# 40 array tasks = 5 seeds x 8 N values, flattened as
# IDX = seed_idx * 8 + n_idx (seed_idx in [0,4], n_idx in [0,7]).
#
# Array index -> (model seed, N):
#   0-7   -> seed 1001, N = 1, 2, 4, 8, 16, 32, 64, 128
#   8-15  -> seed 2002, N = 1, 2, 4, 8, 16, 32, 64, 128
#   16-23 -> seed 3003, N = 1, 2, 4, 8, 16, 32, 64, 128
#   24-31 -> seed 4004, N = 1, 2, 4, 8, 16, 32, 64, 128
#   32-39 -> seed 5005, N = 1, 2, 4, 8, 16, 32, 64, 128
#
# Prerequisites:
#   sbatch scripts/generate_labels_n1248_frequent.sh
#
# Usage:
#   cd mimic_iv_sweep_frequent
#   sbatch scripts/sweep_patient_only_seeds_n1248.sh
#   sbatch --array=0-7 scripts/sweep_patient_only_seeds_n1248.sh   # seed 1001 only
# ============================================================

#SBATCH --job-name=sweep-patient-only-seeds-n1248-frequent
#SBATCH --array=0-39
#SBATCH --partition=gpu
#SBATCH --account=mm6677_gp
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

IDX=${SLURM_ARRAY_TASK_ID}

MODEL_SEEDS=(1001 2002 3003 4004 5005)
NS=(1 2 4 8 16 32 64 128)

SEED_IDX=$((IDX / 8))
N_IDX=$((IDX % 8))

MODEL_SEED=${MODEL_SEEDS[$SEED_IDX]}
N=${NS[$N_IDX]}

REPO_DIR="${SLURM_SUBMIT_DIR}"
VENV="${REPO_DIR}/.venv/bin/activate"  # created by: cd mimic_iv_sweep_frequent && uv sync

TENSORIZED_DIR="/groups/mm6677_gp/data/MIMIC_MEDS/MEDS_cohort/processed"
LABELS_DIR="${REPO_DIR}/data/tasks/n${N}/tasks"
OUTPUT_DIR="${REPO_DIR}/outputs/patient_only_seeds_n1248/seed${MODEL_SEED}/n${N}"

cd "${REPO_DIR}"
# shellcheck source=/dev/null
source "${VENV}"
mkdir -p logs "${OUTPUT_DIR}"

if [ ! -d "${LABELS_DIR}" ]; then
    echo "ERROR: ${LABELS_DIR} not found. Run sbatch scripts/generate_labels_n1248_frequent.sh first." >&2
    exit 1
fi

echo "=== Job info ==="
echo "  Array job       : ${SLURM_ARRAY_JOB_ID:-local}[${IDX}]"
echo "  Node            : ${SLURMD_NODENAME:-$(hostname)}"
echo "  GPU(s)          : ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "  N (tasks)       : ${N}"
echo "  Model seed      : ${MODEL_SEED} (replicate $((SEED_IDX + 1))/5; default would be 42)"
echo "  Code selection  : most_frequent"
echo "  Architecture    : patient_only (no retrieval)"
echo "  Started         : $(date)"
echo ""

# Same combo as sweep_patient_only_n1248.sh -- only seed, OUTPUT_DIR, and
# wandb_run_name differ here. See that script for why query_projector.in_dim=128
# is still required even though fusion=passthrough discards the
# query_projector output.
medrap-train \
    encoder=rope \
    pooling=masked_mean \
    head=linear \
    "head.out_dim=${N}" \
    training/task=multitask_binary \
    "training.task.num_tasks=${N}" \
    training/datamodule=meds_multitask \
    "training.datamodule.config.tensorized_cohort_dir=${TENSORIZED_DIR}" \
    training.datamodule.config.max_seq_len=256 \
    "training.datamodule.config.task_labels_dir=${LABELS_DIR}" \
    "training.datamodule.mt_labels_dir=${LABELS_DIR}" \
    "training.datamodule.num_tasks=${N}" \
    training.datamodule.batch_size=32 \
    training.datamodule.config.seq_sampling_strategy=to_end \
    training/trainer=lightning_wandb \
    training.trainer.max_epochs=3 \
    training.trainer.accelerator=gpu \
    training.trainer.devices=1 \
    training.trainer.gradient_clip_val=1.0 \
    training.trainer.log_every_n_steps=10 \
    training.trainer.limit_val_batches=1.0 \
    training.trainer.val_check_interval=0.5 \
    training.module.lr=1e-3 \
    training.module.warmup_steps=200 \
    training.module.validation_auroc_log_per_task=true \
    fusion=passthrough \
    query_projector.in_dim=128 \
    head.in_dim=128 \
    training/loss=multitask_binary_bce \
    "seed=${MODEL_SEED}" \
    "output_dir=${OUTPUT_DIR}" \
    do_overwrite=true \
    "wandb_run_name=patient-only-seeds-frequent-seed${MODEL_SEED}-n${N}-${SLURM_ARRAY_JOB_ID:-local}_${IDX}" \
    "$@"

echo ""
echo "=== Done: $(date) ==="
