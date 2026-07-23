#!/bin/bash
# ============================================================
# SLURM array: evaluate the trained marginalized (binary) checkpoints against
#              the held-out test split, using only the single top-retrieved
#              document (retriever.k=1) instead of the k=4 used at training.
# ------------------------------------------------------------
# Loads the checkpoint written by sweep_marginalized_binary_n1248.sh
# (outputs/marginalized_binary_n1248/n<N>/checkpoints/last.ckpt) and runs
# medrap-eval eval_mode=test against it. eval_mode=test targets the MEDS
# held_out split (meds_torchdata's Lightning datamodule maps "held_out" ->
# Lightning's "test", "tuning" -> "val"), which none of this experiment's
# other runs have touched -- everything so far has only reported
# val/auroc/* (the tuning split).
#
# Requires McDermottHealthAI/MedRAP#94 (test/auroc/* logging for
# trainer.test(), merged to main) -- without it, medrap-eval eval_mode=test
# only reports test/loss and test/accuracy, no AUROC. Also requires
# McDermottHealthAI/MedRAP#95 (marginalized_retrieval/marginalized_output_mode/
# wandb_project/wandb_run_name declared on _eval.yaml) -- without it,
# overriding marginalized_retrieval=true or training/trainer=lightning_wandb
# for medrap-eval fails with "Could not override 'marginalized_retrieval'"
# (Hydra struct-mode rejects overrides for keys _eval.yaml never declared).
#
# Why retriever.k=1 works on a checkpoint trained with k=4: PerDocCrossAttentionFusion
# and the marginalized_output_mode=binary marginalization are both explicitly
# K-agnostic -- no weight or positional embedding is sized by K, and K=1
# marginalization is mathematically a no-op (see the _marginal_binary_logits
# doctest in medrap.model.model). So this checkpoint's weights are valid at
# any K; setting k=1 makes the model's prediction exactly equal to the
# single top-retrieved document's prediction, i.e. "no marginalization."
#
# Requires medrap built from the commit pinned in pyproject.toml (merges main
# -- PR #93 marginalized_output_mode, PR #94 test/auroc, PR #95 eval-config
# fixes -- with the still-unmerged feat/task-gen-most-frequent-codes for
# code_selection and PR #96 for duration_distribution).
#
# Uses training/trainer=lightning_wandb (same as the training sweeps) so
# results land in W&B like everything else in this experiment, instead of
# only printing to the SLURM job log.
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
#
# Usage:
#   cd mimic_iv_sweep_frequent
#   sbatch scripts/eval_marginalized_binary_top1_n1248.sh
#   sbatch --array=0 scripts/eval_marginalized_binary_top1_n1248.sh   # N=1 only
# ============================================================

#SBATCH --job-name=eval-marginalized-binary-top1-n1248
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
EVAL_OUTPUT_DIR="${REPO_DIR}/outputs/marginalized_binary_top1_eval_n1248/n${N}"

cd "${REPO_DIR}"
# shellcheck source=/dev/null
source "${VENV}"
mkdir -p logs "${EVAL_OUTPUT_DIR}"

if [ ! -f "${CHECKPOINT_PATH}" ]; then
    echo "ERROR: ${CHECKPOINT_PATH} not found. Run sweep_marginalized_binary_n1248.sh for N=${N} first." >&2
    exit 1
fi

echo "=== Job info ==="
echo "  Array job     : ${SLURM_ARRAY_JOB_ID:-local}[${IDX}]"
echo "  Node          : ${SLURMD_NODENAME:-$(hostname)}"
echo "  GPU(s)        : ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "  N (tasks)     : ${N}"
echo "  retriever.k   : 1 (top retrieved doc only; trained with k=4)"
echo "  eval_mode     : test (MEDS held_out split)"
echo "  Checkpoint    : ${CHECKPOINT_PATH}"
echo "  W&B run name  : marginalized-binary-top1-test-frequent-n${N}-${SLURM_ARRAY_JOB_ID:-local}_${IDX}"
echo "  Started       : $(date)"
echo ""

# Same architecture as sweep_marginalized_binary_n1248.sh (must match the
# checkpoint's trained architecture for state_dict loading to succeed),
# except retriever.k=1, eval_mode=test, and checkpoint_path/output_dir.
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
    retriever.k=1 \
    retrieval_encoder=token_feature \
    retrieval_encoder.vocab_size=151936 \
    retrieval_encoder.embedding_dim=64 \
    fusion=cross_attention_perdoc_medium \
    marginalized_retrieval=true \
    marginalized_output_mode=binary \
    head.in_dim=256 \
    training/loss=multitask_binary_bce_marginalized \
    "training.loss.num_tasks=${N}" \
    "checkpoint_path=${CHECKPOINT_PATH}" \
    eval_mode=test \
    "output_dir=${EVAL_OUTPUT_DIR}" \
    "wandb_run_name=marginalized-binary-top1-test-frequent-n${N}-${SLURM_ARRAY_JOB_ID:-local}_${IDX}" \
    "$@"

echo ""
echo "=== Done: $(date) ==="
