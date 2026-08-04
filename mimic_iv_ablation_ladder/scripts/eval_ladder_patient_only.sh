#!/bin/bash
# ============================================================
# SLURM array: COLUMN 1 of the ablation ladder -- evaluate every trained
#              patient_only checkpoint on the held-out test split, model
#              frozen, over 2 code selections x 5 model seeds x 8 N values.
# ------------------------------------------------------------
# This is the between-arm baseline (comparison C1: does retrieval help at
# all?). The patient_only architecture has no retriever and no fusion --
# RoPE encoder -> masked-mean pooling -> LinearHead(B,N) -- so this column
# is the score the retrieval arms have to beat. It is scored on exactly the
# same split, with exactly the same harness, as the three marginalized
# columns: eval_ladder_real_docs.sh, eval_ladder_random_docs.sh and
# eval_ladder_null_docs.sh. That is the whole point -- a between-arm delta
# is only interpretable if the arms differ in architecture and nothing else.
#
# No eval script for this architecture existed before. Do NOT adapt one of
# the eval_frozen_* scripts by editing paths: those are marginalized-arm
# scripts and carry a set of GROUP overrides that are actively wrong here,
# for a reason that fails LOUDLY but only after training has finished.
#
# WHAT MUST NOT BE OVERRIDDEN HERE, AND WHY
# cli.py calls load_state_dict(..., strict=True). sweep_patient_only_ladder.sh
# leaves query_projector, retriever and retrieval_encoder at the shared
# _train.yaml/_eval.yaml defaults (verified identical in both files:
# query_projector=sequence_mean, retriever=in_memory,
# retrieval_encoder=mean_pooled), so the checkpoint literally contains
# model.query_projector.linear.weight of shape [4, 128] -- from
# sequence_mean.yaml's out_dim: 4 -- plus the InMemoryRetriever's persistent
# buffers _doc_key_embeddings [2,4], _doc_tokens [2,2],
# _doc_attention_mask [2,2] and _doc_ids. Consequently:
#   - query_projector=sequence_mean_1024 would build a 128->1024 projector
#     against trained 128->4 weights -> size-mismatch RuntimeError.
#   - retriever=hf_dataset would drop those four buffers -> unexpected and
#     missing keys -> RuntimeError.
#   - retrieval_encoder=token_feature likewise changes the key set.
# So this script passes only the non-group tweaks the training script passed:
# query_projector.in_dim=128, head.in_dim=128, fusion=passthrough. Note that
# query_projector.in_dim=128 is still required even though fusion=passthrough
# discards the projector output -- model.forward() calls
# query_projector(encoder_out) unconditionally.
#
# NO marginalized_* KEYS AT ALL. marginalized_retrieval,
# marginalized_output_mode and especially marginalized_score_similarity=cosine
# belong to the retrieval arms only. marginalized_score_similarity is a plain
# Python attribute rather than a state_dict entry, so setting it here would
# not error -- it would silently change how a model that never marginalized
# anything is scored. It is absent by design; the _eval.yaml defaults
# (marginalized_retrieval: false, marginalized_score_similarity: dot) match
# what training used.
#
# training/loss=multitask_binary_bce -- the PLAIN loss, not
# multitask_binary_bce_marginalized. Copied verbatim from the training
# script, as every architecture-defining override is; dropping the loss
# config was a real bug once (commit 88ad907). The patient_only training
# script passes no training.loss.num_tasks, so neither does this.
#
# training/trainer=lightning_wandb is REQUIRED, not cosmetic. The _eval.yaml
# default trainer is lightning_eval, which sets logger: false -- no
# metrics.csv is written, so aggregate_ladder.py sees no run at all and this
# column comes out silently empty.
#
# eval_mode=test runs the FULL MEDS held_out split (meds_torchdata's
# Lightning datamodule maps "held_out" -> Lightning's "test", "tuning" ->
# "val"). Do NOT add limit_val_batches: nothing limits test batches, and the
# flag would be a no-op that implies a limit exists.
#
# Only the training-only knobs are dropped relative to
# sweep_patient_only_ladder.sh: max_epochs, gradient_clip_val,
# log_every_n_steps, module.lr, module.warmup_steps, limit_val_batches,
# val_check_interval, datamodule.num_workers and seed. (seed is not merely
# unnecessary here -- _eval.yaml declares no `seed` field, so passing it is a
# hard Hydra struct error.)
#
# 80 array tasks = 2 code selections x 5 model seeds x 8 N values, flattened
# as IDX = sel_idx * 40 + seed_idx * 8 + n_idx. The array RANGE selects the
# code selection: --array=0-39 is `random`, --array=40-79 is `frequent`.
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
# Prerequisites:
#   sbatch scripts/generate_labels_random.sh      (or generate_labels_frequent.sh)
#   sbatch scripts/sweep_patient_only_ladder.sh   (checkpoints must exist)
#
# Usage:
#   cd mimic_iv_ablation_ladder
#   sbatch --array=0-39%12 scripts/eval_ladder_patient_only.sh    # random selection
#   sbatch --array=40-79%12 scripts/eval_ladder_patient_only.sh   # frequent selection
#   sbatch --array=0 scripts/eval_ladder_patient_only.sh          # random, seed 1001, N=1
#
# Aggregating: this column is the C1 baseline. --split test is REQUIRED --
# eval_mode=test runs log test/auroc/* and no val/* column at all, so the
# aggregator's default --split val would find nothing. The rep component
# regex must collapse BOTH the seed and rep directory levels, since the four
# ladder columns sit at different depths:
#
#   python scripts/aggregate_ladder.py --split test --rep-component '(seed|rep)\d+'
# ============================================================

#SBATCH --job-name=eval-ladder-patient-only
#SBATCH --array=0-79
#SBATCH --partition=gpu
#SBATCH --account=mm6677_gp
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
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

# No RETRIEVAL_DB: the patient_only architecture never touches a corpus.
TENSORIZED_DIR="/groups/mm6677_gp/data/MIMIC_MEDS/MEDS_cohort/processed"
LABELS_DIR="${REPO_DIR}/data/tasks_${SELECTION}/n${N}/tasks"
TRAIN_OUTPUT_DIR="${REPO_DIR}/outputs/patient_only_ladder/${SELECTION}/seed${MODEL_SEED}/n${N}"
CHECKPOINT_PATH="${TRAIN_OUTPUT_DIR}/checkpoints/last.ckpt"
EVAL_OUTPUT_DIR="${REPO_DIR}/outputs/eval_patient_only/${SELECTION}/seed${MODEL_SEED}/n${N}"

cd "${REPO_DIR}"
# shellcheck source=/dev/null
source "${VENV}"
mkdir -p logs "${EVAL_OUTPUT_DIR}"

if [ ! -d "${LABELS_DIR}" ]; then
    echo "ERROR: ${LABELS_DIR} not found. Run sbatch scripts/generate_labels_${SELECTION}.sh first." >&2
    exit 1
fi

if [ ! -f "${CHECKPOINT_PATH}" ]; then
    echo "ERROR: ${CHECKPOINT_PATH} not found. Run sbatch scripts/sweep_patient_only_ladder.sh first." >&2
    exit 1
fi

echo "=== Job info ==="
echo "  Array job      : ${SLURM_ARRAY_JOB_ID:-local}[${IDX}]"
echo "  Node           : ${SLURMD_NODENAME:-$(hostname)}"
echo "  GPU(s)         : ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "  Ladder column  : 1/4 -- patient_only (C1 baseline)"
echo "  Code selection : ${SELECTION}"
echo "  Model seed     : ${MODEL_SEED} (replicate $((SEED_IDX + 1))/5)"
echo "  N (tasks)      : ${N}"
echo "  Architecture   : patient_only (no retriever, no fusion)"
echo "  eval_mode      : test (MEDS held_out split, no batch limit)"
echo "  Checkpoint     : ${CHECKPOINT_PATH}"
echo "  W&B run name   : eval-patient-only-${SELECTION}-seed${MODEL_SEED}-n${N}-${SLURM_ARRAY_JOB_ID:-local}_${IDX}"
echo "  Started        : $(date)"
echo ""

# Same architecture as sweep_patient_only_ladder.sh (must match the
# checkpoint's trained architecture for state_dict loading to succeed), minus
# the training-only knobs, plus checkpoint_path and eval_mode=test. No
# query_projector/retriever/retrieval_encoder GROUP overrides and no
# marginalized_* keys -- see the header.
medrap-eval \
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
    training.trainer.accelerator=gpu \
    training.trainer.devices=1 \
    training.module.validation_auroc_log_per_task=true \
    fusion=passthrough \
    query_projector.in_dim=128 \
    head.in_dim=128 \
    training/loss=multitask_binary_bce \
    "checkpoint_path=${CHECKPOINT_PATH}" \
    eval_mode=test \
    "output_dir=${EVAL_OUTPUT_DIR}" \
    do_overwrite=true \
    "wandb_run_name=eval-patient-only-${SELECTION}-seed${MODEL_SEED}-n${N}-${SLURM_ARRAY_JOB_ID:-local}_${IDX}" \
    "$@"

echo ""
echo "=== Done: $(date) ==="
