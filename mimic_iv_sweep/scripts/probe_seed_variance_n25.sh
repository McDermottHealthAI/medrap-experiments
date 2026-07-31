#!/bin/bash
# ============================================================
# SLURM array: model-seed variance PROBE -- both arms (patient_only,
#              marginalized_binary) x 5 model seeds at a SINGLE
#              (duration, draw) cell, d7/draw1, fixed N=25. Measures
#              sigma_seed directly so the size of any full seed x draw
#              grid can be chosen from data instead of guessed.
# ------------------------------------------------------------
# Every training script in this repo inherits `seed: 42` from medrap's
# conf/_train.yaml, and no script has ever overridden it. Training
# stochasticity has therefore been sampled exactly ONCE per cell, so no
# number of extra label draws can separate "this architecture is worse"
# from "this seed was unlucky" -- a draw resamples the task, never the
# optimizer.
#
# This is an ADDITION to the existing paired design, not a replacement for
# it. The mimic_iv_sweep paired-over-draws comparison already removes 87.7%
# of the noise in the arm difference: the two arms correlate 0.985 across
# draws, and the sd of the paired delta is 0.0110 versus 0.0896 if the arms
# were scored independently. Pairing stays. Model seeds are the missing
# component, and this script measures how large that component is before
# anything is committed.
#
# `seed=<model seed>` reaches lightning.seed_everything(cfg.seed,
# workers=True) in medrap's cli.py train_main (line 429 in the currently
# pinned build), which seeds weight init, dropout AND dataloader order
# together. What varies across the 5 runs within an arm is therefore TOTAL
# training stochasticity, not initialization variance alone.
#
# MODEL_SEEDS=(1001 2002 3003 4004 5005) are deliberately distinct from the
# default 42 and from the label-draw seeds 101/202/303/404/505 used by
# generate_labels_duration_variance_n25.sh, matching this repo's stated
# rationale of keeping seed families disjoint so a model seed can never
# silently coincide with a data seed.
#
# Both arms clone their override lists VERBATIM from
# sweep_patient_only_duration_variance_n25.sh and
# sweep_marginalized_binary_duration_variance_n25.sh (both already at the
# full-tuning-split validation protocol, limit_val_batches=1.0 /
# val_check_interval=0.5, and the marginalized arm at
# marginalized_score_similarity=cosine). The ONLY additions are
# seed=${MODEL_SEED}, output_dir, and wandb_run_name, so a seed spread
# measured here is directly comparable to the existing d7/draw1 runs.
#
# DECISION RULE -- this is a SIZING probe. Do NOT commit to the full
# seed x draw grid before reading its result:
#   sigma_seed << 0.011 -> training noise is negligible next to the paired
#                          across-draw spread the existing 5-draw design
#                          already resolves. Keep 5 draws as-is and add
#                          only 3 seeds at the headline cells, purely to
#                          report a seed CI on the headline numbers.
#   sigma_seed ~  0.011 -> training noise is the same order as the sd of
#                          the paired delta, i.e. it dominates any
#                          single-seed result. Go to 5 draws x 3 seeds
#                          (15 runs per arm per cell) so the reported mean
#                          averages over both noise sources.
# Estimate sigma_seed as the sd of final val/auroc across the 5 seeds
# WITHIN each arm (10 runs -> two estimates, 4 df each). The arm difference
# at this single cell is not the point of this script and must not be read
# as a result: n=1 draw, so it carries none of the pairing benefit above.
#
# Array index -> (arm, model seed), IDX = arm_idx * 5 + seed_idx:
#   0-4 -> patient_only,        seeds 1001, 2002, 3003, 4004, 5005
#   5-9 -> marginalized_binary, seeds 1001, 2002, 3003, 4004, 5005
#
# Prerequisites:
#   sbatch scripts/prepare_retrieval.sh   (marginalized_binary arm only --
#     the patient_only arm runs fusion=passthrough and never reads the corpus)
#   sbatch scripts/generate_labels_duration_variance_n25.sh   (only the
#     d7/draw1 labels are read here; the other 9 cells are untouched)
#
# Usage:
#   cd mimic_iv_sweep
#   sbatch scripts/probe_seed_variance_n25.sh
#   sbatch --array=0-4 scripts/probe_seed_variance_n25.sh   # patient_only arm only
#   sbatch --array=5-9 scripts/probe_seed_variance_n25.sh   # marginalized arm only
# ============================================================

#SBATCH --job-name=probe-seed-variance-n25
#SBATCH --array=0-9
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

ARMS=(patient_only marginalized_binary)
MODEL_SEEDS=(1001 2002 3003 4004 5005)

ARM_IDX=$((IDX / 5))
SEED_IDX=$((IDX % 5))

ARM=${ARMS[$ARM_IDX]}
MODEL_SEED=${MODEL_SEEDS[$SEED_IDX]}
DURATION_DAYS=7
DRAW=1
N=25

REPO_DIR="${SLURM_SUBMIT_DIR}"
VENV="${REPO_DIR}/.venv/bin/activate"  # created by: cd mimic_iv_sweep && uv sync

RETRIEVAL_DB="${REPO_DIR}/data/retrieval_db"
TENSORIZED_DIR="/groups/mm6677_gp/data/MIMIC_MEDS/MEDS_cohort/processed"
LABELS_DIR="${REPO_DIR}/data/tasks_duration_variance/d${DURATION_DAYS}/draw${DRAW}/tasks"
OUTPUT_DIR="${REPO_DIR}/outputs/probe_seed_variance_n25/${ARM}/seed${MODEL_SEED}"

cd "${REPO_DIR}"
# shellcheck source=/dev/null
source "${VENV}"
mkdir -p logs "${OUTPUT_DIR}"

# Only the marginalized_binary arm reads the corpus; the patient_only arm runs
# fusion=passthrough and never passes retriever.dataset_path, so guarding it
# unconditionally would abort the documented `--array=0-4` (patient_only only)
# submission on an artifact it does not need.
if [ "${ARM}" != "patient_only" ] && [ ! -d "${RETRIEVAL_DB}" ]; then
    echo "ERROR: ${RETRIEVAL_DB} not found. Run sbatch scripts/prepare_retrieval.sh first." >&2
    exit 1
fi

if [ ! -d "${LABELS_DIR}" ]; then
    echo "ERROR: ${LABELS_DIR} not found. Run sbatch scripts/generate_labels_duration_variance_n25.sh for duration=${DURATION_DAYS}d draw=${DRAW} first." >&2
    exit 1
fi

echo "=== Job info ==="
echo "  Array job   : ${SLURM_ARRAY_JOB_ID:-local}[${IDX}]"
echo "  Node        : ${SLURMD_NODENAME:-$(hostname)}"
echo "  GPU(s)      : ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "  N (tasks)   : ${N}"
echo "  Duration    : ${DURATION_DAYS}d"
echo "  Draw        : ${DRAW}/5 (fixed -- this probe varies the model seed only)"
echo "  Arm         : ${ARM}"
echo "  Model seed  : ${MODEL_SEED} (default would be 42)"
echo "  Started     : $(date)"
echo ""

if [ "${ARM}" = "patient_only" ]; then
    # Override list cloned verbatim from
    # sweep_patient_only_duration_variance_n25.sh; only seed/output_dir/
    # wandb_run_name are added.
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
        "wandb_run_name=probe-seed-variance-${ARM}-seed${MODEL_SEED}-${SLURM_ARRAY_JOB_ID:-local}_${IDX}" \
        "$@"
else
    # Override list cloned verbatim from
    # sweep_marginalized_binary_duration_variance_n25.sh; only seed/
    # output_dir/wandb_run_name are added.
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
        query_projector=sequence_mean_1024 \
        query_projector.in_dim=128 \
        retriever=hf_dataset \
        "retriever.dataset_path=${RETRIEVAL_DB}" \
        retriever.doc_ids_column=null \
        retriever.k=4 \
        retrieval_encoder=token_feature \
        retrieval_encoder.vocab_size=151936 \
        retrieval_encoder.embedding_dim=64 \
        fusion=cross_attention_perdoc_medium \
        marginalized_retrieval=true \
        marginalized_output_mode=binary \
        marginalized_score_similarity=cosine \
        head.in_dim=256 \
        training/loss=multitask_binary_bce_marginalized \
        "training.loss.num_tasks=${N}" \
        "seed=${MODEL_SEED}" \
        "output_dir=${OUTPUT_DIR}" \
        do_overwrite=true \
        "wandb_run_name=probe-seed-variance-${ARM}-seed${MODEL_SEED}-${SLURM_ARRAY_JOB_ID:-local}_${IDX}" \
        "$@"
fi

echo ""
echo "=== Done: $(date) ==="
