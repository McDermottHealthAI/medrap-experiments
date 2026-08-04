#!/bin/bash
# ============================================================
# SLURM array: patient-only (no retrieval), 2 code selections x 5 model
#              seeds x N in {1, 2, 4, 8, 16, 32, 64, 128} -- column 1 of
#              the ablation ladder. Trains the no-retrieval baseline whose
#              checkpoints eval_ladder_patient_only.sh scores on held_out.
# ------------------------------------------------------------
# Every Hydra override is copied verbatim from
# ../mimic_iv_sweep_frequent/scripts/sweep_patient_only_seeds_n1248.sh --
# including the full-tuning-split validation protocol
# (training.trainer.limit_val_batches=1.0, training.trainer.val_check_interval=0.5)
# -- with exactly two changes: this array is 3-D (code selection is now an
# array dimension rather than a property of the directory), and
# training.datamodule.num_workers=8 is added.
#
# num_workers=8 is a pure throughput change and cannot perturb any number.
# training.datamodule.config.seq_sampling_strategy=to_end resolves to
# `return seq_len - max_seq_len` in meds_torchdata/types.py -- there is no RNG
# anywhere in the subsequence path -- the label join is keyed on
# (subject_id, prediction_time), and lightning.seed_everything(cfg.seed,
# workers=True) seeds every DataLoader worker regardless. The scripts have
# always requested --cpus-per-task=16 and left 16 of them idle while every
# training log warned that the dataloader was the bottleneck.
#
# Architecture note: query_projector, retriever and retrieval_encoder are
# deliberately left at their conf/_train.yaml defaults. fusion=passthrough
# discards the query_projector output, but the module is still built, so the
# checkpoint contains query_projector.linear.weight [4, 128] (from
# sequence_mean.yaml's out_dim: 4) and the InMemoryRetriever's persistent
# buffers. query_projector.in_dim=128 is therefore still required, and the
# matching eval script must not override any of those three as GROUPS --
# medrap's cli.py calls load_state_dict(strict=True), so a group override is
# a hard RuntimeError after training has already been paid for. No
# marginalized_* key appears in this arm at all.
#
# 80 array tasks = 2 selections x 5 seeds x 8 N values, flattened as
# IDX = sel_idx * 40 + seed_idx * 8 + n_idx
# (sel_idx in [0,1], seed_idx in [0,4], n_idx in [0,7]).
#
# Array index -> (code selection, model seed, N):
#   0-7   -> random,   seed 1001, N = 1, 2, 4, 8, 16, 32, 64, 128
#   8-15  -> random,   seed 2002, N = 1, 2, 4, 8, 16, 32, 64, 128
#   16-23 -> random,   seed 3003, N = 1, 2, 4, 8, 16, 32, 64, 128
#   24-31 -> random,   seed 4004, N = 1, 2, 4, 8, 16, 32, 64, 128
#   32-39 -> random,   seed 5005, N = 1, 2, 4, 8, 16, 32, 64, 128
#   40-47 -> frequent, seed 1001, N = 1, 2, 4, 8, 16, 32, 64, 128
#   48-55 -> frequent, seed 2002, N = 1, 2, 4, 8, 16, 32, 64, 128
#   56-63 -> frequent, seed 3003, N = 1, 2, 4, 8, 16, 32, 64, 128
#   64-71 -> frequent, seed 4004, N = 1, 2, 4, 8, 16, 32, 64, 128
#   72-79 -> frequent, seed 5005, N = 1, 2, 4, 8, 16, 32, 64, 128
#
# The array range IS the code selection: --array=0-39 runs `random` only and
# --array=40-79 runs `frequent` only, which is how run_all.sh --selection
# picks between them. Running one selection at a time keeps all 40 paired
# cells, all 5 seeds and all 8 N values -- it only defers the second table.
#
# MODEL_SEEDS=(1001 2002 3003 4004 5005) are deliberately distinct from the
# default 42 and from the label-draw seeds 101/202/303/404/505, matching this
# repo's rationale of keeping seed families disjoint.
#
# Prerequisites:
#   sbatch scripts/generate_labels_random.sh
#   sbatch scripts/generate_labels_frequent.sh
#
# Usage:
#   cd mimic_iv_ablation_ladder
#   sbatch --array=0-39%12 scripts/sweep_patient_only_ladder.sh   # random
#   sbatch --array=40-79%12 scripts/sweep_patient_only_ladder.sh  # frequent
#   sbatch --array=0 scripts/sweep_patient_only_ladder.sh         # smoke test
# ============================================================

#SBATCH --job-name=sweep-patient-only-ladder
#SBATCH --array=0-79
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

SELECTIONS=(random frequent)
MODEL_SEEDS=(1001 2002 3003 4004 5005)
NS=(1 2 4 8 16 32 64 128)

SEL_IDX=$((IDX / 40))
REM=$((IDX % 40))
SEED_IDX=$((REM / 8))
N_IDX=$((REM % 8))

SELECTION=${SELECTIONS[$SEL_IDX]}
MODEL_SEED=${MODEL_SEEDS[$SEED_IDX]}
N=${NS[$N_IDX]}

REPO_DIR="${SLURM_SUBMIT_DIR}"
VENV="${REPO_DIR}/.venv/bin/activate"  # created by: cd mimic_iv_ablation_ladder && uv sync

TENSORIZED_DIR="/groups/mm6677_gp/data/MIMIC_MEDS/MEDS_cohort/processed"
LABELS_DIR="${REPO_DIR}/data/tasks_${SELECTION}/n${N}/tasks"
OUTPUT_DIR="${REPO_DIR}/outputs/patient_only_ladder/${SELECTION}/seed${MODEL_SEED}/n${N}"

cd "${REPO_DIR}"
# shellcheck source=/dev/null
source "${VENV}"
mkdir -p logs "${OUTPUT_DIR}"

if [ ! -d "${LABELS_DIR}" ]; then
    echo "ERROR: ${LABELS_DIR} not found. Run sbatch scripts/generate_labels_${SELECTION}.sh first." >&2
    exit 1
fi

echo "=== Job info ==="
echo "  Array job       : ${SLURM_ARRAY_JOB_ID:-local}[${IDX}]"
echo "  Node            : ${SLURMD_NODENAME:-$(hostname)}"
echo "  GPU(s)          : ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "  Code selection  : ${SELECTION}"
echo "  N (tasks)       : ${N}"
echo "  Model seed      : ${MODEL_SEED} (replicate $((SEED_IDX + 1))/5; default would be 42)"
echo "  Architecture    : patient_only (no retrieval)"
echo "  Started         : $(date)"
echo ""

# Override set copied verbatim from
# ../mimic_iv_sweep_frequent/scripts/sweep_patient_only_seeds_n1248.sh, plus
# training.datamodule.num_workers=8. query_projector.in_dim=128 is still
# required even though fusion=passthrough discards the query_projector
# output; see the header for why it must not be overridden as a group.
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
    training.datamodule.num_workers=8 \
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
    "wandb_run_name=patient-only-ladder-${SELECTION}-seed${MODEL_SEED}-n${N}-${SLURM_ARRAY_JOB_ID:-local}_${IDX}" \
    "$@"

echo ""
echo "=== Done: $(date) ==="
