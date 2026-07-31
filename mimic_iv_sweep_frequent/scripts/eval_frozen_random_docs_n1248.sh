#!/bin/bash
# ============================================================
# SLURM array: inference-time random-document ablation on the trained
#              marginalized (binary) checkpoints -- N in
#              {1, 2, 4, 8, 16, 32, 64, 128} x 5 repetitions, model frozen,
#              the real retrieved documents replaced by random ones at
#              eval time only (retriever.ablation_mode=random_docs).
# ------------------------------------------------------------
# Answers "how much of the marginalized model's AUROC actually comes from
# the retrieved content, versus from the patient sequence alone?" Nothing is
# retrained here: this loads the checkpoint written by
# sweep_marginalized_binary_n1248.sh
# (outputs/marginalized_binary_n1248/n<N>/checkpoints/last.ckpt) and swaps
# only what the retriever hands the fusion module. Every number here is
# meant to be read against the same-N number from
# eval_frozen_real_docs_n1248.sh, which is this exact harness with
# ablation_mode=none.
#
# Differs from eval_marginalized_binary_top1_n1248.sh in three ways:
#   1. retriever.ablation_mode=random_docs (that script runs no ablation).
#   2. retriever.k=4 -- the k the checkpoints were TRAINED with. Do NOT copy
#      that script's retriever.k=1: fewer documents is a different ablation,
#      and mixing it in would confound document count with document
#      relevance, which is the only thing this script is trying to measure.
#   3. A rep dimension (see below).
#
# Why eval_mode=test and NOT eval_mode=validate: training/trainer=lightning_wandb
# carries limit_val_batches=200, so eval_mode=validate would score only the
# first 200 batches x batch_size 32 = 6,400 rows of the tuning split -- a
# prefix, not the split. Nothing limits test batches, so eval_mode=test runs
# the full MEDS held_out split (meds_torchdata's Lightning datamodule maps
# "held_out" -> Lightning's "test", "tuning" -> "val").
#
# Why 5 reps: the random draw is UNSEEDED. medrap-eval never calls
# seed_everything and _eval.yaml declares no `seed` field, so every run picks
# a different set of random documents and no run is reproducible. Rather than
# report one arbitrary draw, this runs 5 independent repetitions per N;
# report mean +/- sd of test/auroc across the reps, not a single rep.
#
# Every OTHER Hydra override is copied VERBATIM from
# sweep_marginalized_binary_n1248.sh. If the architecture args drift, either
# load_state_dict fails outright or -- worse -- the weights load into a
# silently different model. That includes
# training/loss=multitask_binary_bce_marginalized and
# training.loss.num_tasks=<N>; dropping the loss config was a real bug once
# (commit 88ad907). Only the training-only knobs are dropped: max_epochs,
# gradient_clip_val, log_every_n_steps, module.lr, module.warmup_steps,
# limit_val_batches, val_check_interval.
#
# marginalized_score_similarity=cosine IS set, and must be: it is a
# scoring-function override rather than a weight shape, so it has to MATCH
# the checkpoint being loaded. sweep_marginalized_binary_n1248.sh now trains
# with cosine (rationale: NULL_RESULT_DIAGNOSIS.md), so this eval must pass
# it too. The flag is a plain Python attribute, not a state_dict entry, so
# omitting it would load cleanly and then silently score documents by a
# different rule than training used. That is exactly the class of bug this
# ablation exists to detect, so it must not be introduced by the ablation
# itself. Run this only against checkpoints produced by the post-change
# sweep; results are NOT comparable to pre-change (dot-trained) numbers.
#
# Requires medrap built from the commit pinned in pyproject.toml, with the
# same prerequisites as eval_marginalized_binary_top1_n1248.sh
# (McDermottHealthAI/MedRAP#94 for test/auroc/* logging under trainer.test(),
# #95 for the eval-config key declarations Hydra struct mode needs --
# marginalized_retrieval, marginalized_score_similarity,
# marginalized_output_mode, wandb_project, wandb_run_name -- and #97 for
# do_overwrite on medrap-eval), plus a build whose retriever config declares
# ablation_mode -- otherwise Hydra rejects the override outright.
#
# 40 array tasks = 8 N values x 5 reps, flattened as
# IDX = n_idx * 5 + rep_idx (n_idx in [0,7], rep_idx in [0,4]) -- same idiom
# as ../mimic_iv_sweep/scripts/generate_labels_variance_n1248.sh.
#
# Array index -> (N, rep):
#   0-4   -> N=1,   reps 1-5
#   5-9   -> N=2,   reps 1-5
#   10-14 -> N=4,   reps 1-5
#   15-19 -> N=8,   reps 1-5
#   20-24 -> N=16,  reps 1-5
#   25-29 -> N=32,  reps 1-5
#   30-34 -> N=64,  reps 1-5
#   35-39 -> N=128, reps 1-5
#
# Prerequisites:
#   sweep_marginalized_binary_n1248.sh must have already finished for the N
#   values you want to evaluate (checkpoints must exist).
#   sbatch scripts/eval_frozen_real_docs_n1248.sh   (the paired control)
#
# Usage:
#   cd mimic_iv_sweep_frequent
#   sbatch scripts/eval_frozen_random_docs_n1248.sh
#   sbatch --array=0-4 scripts/eval_frozen_random_docs_n1248.sh   # N=1, all 5 reps
#   sbatch --array=0,5,10,15,20,25,30,35 scripts/eval_frozen_random_docs_n1248.sh   # rep 1 only, all N
#
# Aggregating the reps (this is what produces the "mean +/- sd of test/auroc
# across the reps" above, paired against the real-docs control). --split test
# is REQUIRED: eval_mode=test runs log test/auroc/* and no val/* column at
# all, so the aggregator's default --split val would find nothing.
# --rep-component collapses the rep<N> path component so the 5 reps of each N
# aggregate into one cell instead of five:
#
#   python scripts/aggregate_results.py --split test --rep-component 'rep\d+' \
#       --baseline  'outputs/frozen_real_docs_eval_n1248/n*' \
#       --baseline-label  'real docs' \
#       --treatment 'outputs/frozen_random_docs_eval_n1248/n*/rep*' \
#       --treatment-label 'random docs'
# ============================================================

#SBATCH --job-name=eval-frozen-random-docs-n1248-frequent
#SBATCH --array=0-39
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

NS=(1 2 4 8 16 32 64 128)

N_IDX=$((IDX / 5))
REP_IDX=$((IDX % 5))
N=${NS[$N_IDX]}
REP=$((REP_IDX + 1))

REPO_DIR="${SLURM_SUBMIT_DIR}"
VENV="${REPO_DIR}/.venv/bin/activate"  # created by: cd mimic_iv_sweep_frequent && uv sync

RETRIEVAL_DB="${REPO_DIR}/../mimic_iv_sweep/data/retrieval_db"
TENSORIZED_DIR="/groups/mm6677_gp/data/MIMIC_MEDS/MEDS_cohort/processed"
LABELS_DIR="${REPO_DIR}/data/tasks/n${N}/tasks"
TRAIN_OUTPUT_DIR="${REPO_DIR}/outputs/marginalized_binary_n1248/n${N}"
CHECKPOINT_PATH="${TRAIN_OUTPUT_DIR}/checkpoints/last.ckpt"
EVAL_OUTPUT_DIR="${REPO_DIR}/outputs/frozen_random_docs_eval_n1248/n${N}/rep${REP}"

cd "${REPO_DIR}"
# shellcheck source=/dev/null
source "${VENV}"
mkdir -p logs "${EVAL_OUTPUT_DIR}"

if [ ! -d "${RETRIEVAL_DB}" ]; then
    echo "ERROR: ${RETRIEVAL_DB} not found. Run sbatch ../mimic_iv_sweep/scripts/prepare_retrieval.sh first." >&2
    exit 1
fi

if [ ! -f "${CHECKPOINT_PATH}" ]; then
    echo "ERROR: ${CHECKPOINT_PATH} not found. Run sbatch scripts/sweep_marginalized_binary_n1248.sh for N=${N} first." >&2
    exit 1
fi

echo "=== Job info ==="
echo "  Array job     : ${SLURM_ARRAY_JOB_ID:-local}[${IDX}]"
echo "  Node          : ${SLURMD_NODENAME:-$(hostname)}"
echo "  GPU(s)        : ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "  N (tasks)     : ${N}"
echo "  ablation_mode : random_docs (frozen model, random docs at inference)"
echo "  Rep           : ${REP}/5 (UNSEEDED draw -- aggregate mean +/- sd)"
echo "  retriever.k   : 4 (same k the checkpoint was trained with)"
echo "  eval_mode     : test (MEDS held_out split, no batch limit)"
echo "  Checkpoint    : ${CHECKPOINT_PATH}"
echo "  W&B run name  : frozen-random-docs-frequent-n${N}-rep${REP}-${SLURM_ARRAY_JOB_ID:-local}_${IDX}"
echo "  Started       : $(date)"
echo ""

# Same architecture as sweep_marginalized_binary_n1248.sh (must match the
# checkpoint's trained architecture for state_dict loading to succeed),
# except retriever.ablation_mode=random_docs, eval_mode=test, and
# checkpoint_path/output_dir.
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
    query_projector=sequence_mean_1024 \
    query_projector.in_dim=128 \
    retriever=hf_dataset \
    "retriever.dataset_path=${RETRIEVAL_DB}" \
    retriever.doc_ids_column=null \
    retriever.k=4 \
    retriever.ablation_mode=random_docs \
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
    "checkpoint_path=${CHECKPOINT_PATH}" \
    eval_mode=test \
    "output_dir=${EVAL_OUTPUT_DIR}" \
    do_overwrite=true \
    "wandb_run_name=frozen-random-docs-frequent-n${N}-rep${REP}-${SLURM_ARRAY_JOB_ID:-local}_${IDX}" \
    "$@"

echo ""
echo "=== Done: $(date) ==="
