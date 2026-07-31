#!/bin/bash
# ============================================================
# SLURM array: paired CONTROL for the inference-time random-document
#              ablation -- evaluate each trained marginalized (binary)
#              checkpoint on the held-out test split with the REAL
#              retrieved documents (retriever.ablation_mode=none), model
#              frozen, N in {1, 2, 4, 8, 16, 32, 64, 128}.
# ------------------------------------------------------------
# Byte-for-byte the same harness as eval_frozen_random_docs_n1248.sh apart
# from retriever.ablation_mode=none, the missing rep dimension, output_dir,
# and wandb_run_name. That is the entire point: the ablation's effect is
# (this script's test/auroc) - (the random-docs test/auroc), and that
# subtraction is only meaningful if literally nothing else differs.
#
# Do NOT substitute the training-time val/auroc from
# sweep_marginalized_binary_n1248.sh for this control. That number comes
# from a different split: tuning, not held_out. (Since that sweep now sets
# training.trainer.limit_val_batches=1.0 it is at least the FULL tuning
# split rather than the old 200-batch/6,400-row prefix -- but full-tuning is
# still the wrong split. held_out is what this ablation is measured on.)
# Comparing random-docs-on-held_out against real-docs-on-tuning would
# confound the ablation with a change of split, and the resulting delta
# would be uninterpretable.
#
# eval_mode=test for the same reason as the random-docs script: nothing
# limits test batches, so eval_mode=test runs the FULL MEDS held_out split
# (meds_torchdata's Lightning datamodule maps "held_out" -> Lightning's
# "test", "tuning" -> "val"), whereas eval_mode=validate would silently
# score only a 6,400-row prefix.
#
# No rep dimension here: with ablation_mode=none there is no random draw at
# inference, so this is deterministic and one run per N is exact. The
# random-docs side needs 5 reps because its draw is unseeded (medrap-eval
# never calls seed_everything and _eval.yaml has no `seed` field); compare
# this single number against the mean +/- sd of those reps.
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
# different rule than training used -- exactly the class of bug this
# ablation exists to detect, and it must not be introduced by the ablation
# itself. Run this only against checkpoints produced by the post-change
# sweep; results are NOT comparable to pre-change (dot-trained) numbers.
#
# retriever.k=4 (the trained k), NOT the retriever.k=1 of
# eval_marginalized_binary_top1_n1248.sh -- document count must be held
# fixed across the ablation and its control.
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
# Array index -> N:
#   0 -> N=1
#   1 -> N=2
#   2 -> N=4
#   3 -> N=8
#   4 -> N=16
#   5 -> N=32
#   6 -> N=64
#   7 -> N=128
#
# Prerequisites:
#   sweep_marginalized_binary_n1248.sh must have already finished for the N
#   values you want to evaluate (checkpoints must exist).
#   sbatch scripts/eval_frozen_random_docs_n1248.sh   (the paired ablation)
#
# Usage:
#   cd mimic_iv_sweep_frequent
#   sbatch scripts/eval_frozen_real_docs_n1248.sh
#   sbatch --array=0 scripts/eval_frozen_real_docs_n1248.sh   # N=1 only
#
# Aggregating (paired against the random-docs ablation). --split test is
# REQUIRED: eval_mode=test runs log test/auroc/* and no val/* column at all,
# so the aggregator's default --split val would find nothing.
#
#   python scripts/aggregate_results.py --split test --rep-component 'rep\d+' \
#       --baseline  'outputs/frozen_real_docs_eval_n1248/n*' \
#       --baseline-label  'real docs' \
#       --treatment 'outputs/frozen_random_docs_eval_n1248/n*/rep*' \
#       --treatment-label 'random docs'
# ============================================================

#SBATCH --job-name=eval-frozen-real-docs-n1248-frequent
#SBATCH --array=0-7
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
N=${NS[$IDX]}

REPO_DIR="${SLURM_SUBMIT_DIR}"
VENV="${REPO_DIR}/.venv/bin/activate"  # created by: cd mimic_iv_sweep_frequent && uv sync

RETRIEVAL_DB="${REPO_DIR}/../mimic_iv_sweep/data/retrieval_db"
TENSORIZED_DIR="/groups/mm6677_gp/data/MIMIC_MEDS/MEDS_cohort/processed"
LABELS_DIR="${REPO_DIR}/data/tasks/n${N}/tasks"
TRAIN_OUTPUT_DIR="${REPO_DIR}/outputs/marginalized_binary_n1248/n${N}"
CHECKPOINT_PATH="${TRAIN_OUTPUT_DIR}/checkpoints/last.ckpt"
EVAL_OUTPUT_DIR="${REPO_DIR}/outputs/frozen_real_docs_eval_n1248/n${N}"

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
echo "  ablation_mode : none (control -- real retrieved docs, frozen model)"
echo "  retriever.k   : 4 (same k the checkpoint was trained with)"
echo "  eval_mode     : test (MEDS held_out split, no batch limit)"
echo "  Checkpoint    : ${CHECKPOINT_PATH}"
echo "  W&B run name  : frozen-real-docs-frequent-n${N}-${SLURM_ARRAY_JOB_ID:-local}_${IDX}"
echo "  Started       : $(date)"
echo ""

# Same architecture as sweep_marginalized_binary_n1248.sh (must match the
# checkpoint's trained architecture for state_dict loading to succeed), and
# identical to eval_frozen_random_docs_n1248.sh except
# retriever.ablation_mode=none and output_dir/wandb_run_name.
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
    retriever.ablation_mode=none \
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
    "wandb_run_name=frozen-real-docs-frequent-n${N}-${SLURM_ARRAY_JOB_ID:-local}_${IDX}" \
    "$@"

echo ""
echo "=== Done: $(date) ==="
