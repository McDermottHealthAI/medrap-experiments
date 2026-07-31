#!/bin/bash
# ============================================================
# SLURM array: cross-attention marginalized retrieval with
#              marginalized_output_mode=binary, k=10 retrieved documents
#              DURING TRAINING (sweep_marginalized_binary_n1248.sh uses k=4),
#              N in {1, 2, 4, 8, 16, 32, 64, 128}.
# ------------------------------------------------------------
# Identical to sweep_marginalized_binary_n1248.sh except retriever.k=10 --
# more documents marginalized over per prediction, both in the loss (via
# per_doc_logits/differentiable_doc_scores, shape (B, K, N) with K=10 now)
# and in the reported logits/AUROC (same K, since there's no train/inference
# k mismatch here -- see MedRAP#94's eval_marginalized_binary_top1_n1248.sh
# for the deliberately-mismatched-k ablation in the other direction, k=1 at
# eval on a k=4-trained checkpoint). Bumped SBATCH mem/time since more
# retrieved documents per batch means more compute and memory per step, most
# visible on the larger tensorized_encoder/PerDocCrossAttentionFusion batch
# (effectively B*K instead of B*4). Separate output_dir/wandb_run_name from
# the base sweep so this never collides with (or silently overwrites, via
# do_overwrite=true) the existing k=4 checkpoints.
#
# Validation protocol changed here (rationale: NULL_RESULT_DIAGNOSIS.md):
# training.trainer.limit_val_batches=1.0 scores the FULL tuning split on
# every validation pass instead of the 200-batch (6,400-row) prefix
# conf/training/trainer/lightning_wandb.yaml defaults to, and
# training.trainer.val_check_interval moves 0.2 -> 0.5 to offset the cost
# (two validation passes per epoch instead of five). That prefix held so few
# positives per task that a single positive changing rank could move the
# task's AUROC by up to ~0.5, which is the same size as the k=10-vs-k=4
# difference this script exists to measure. Numbers from this script are
# therefore NOT comparable to results published before this change.
#
# The marginalized document score is now cosine rather than dot
# (marginalized_score_similarity=cosine), same rationale, and it matters
# most here: the reason k=10 didn't systematically beat k=4 the first time
# is that the raw dot product left ||q|| acting as an implicit inverse
# temperature, so the softmax over the K retrieved documents saturated --
# effective_k_mean sat at ~1.0 even with ten documents available, making the
# marginalization arithmetically a no-op and driving the gradient into the
# retrieval scores to exactly zero. cosine bounds the scores to [-1, 1] and
# removes that degree of freedom, so this is the first run where k=10 can
# actually differ from k=4.
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
#   sbatch ../mimic_iv_sweep/scripts/prepare_retrieval.sh   (if not already run)
#   sbatch scripts/generate_labels_n1248_frequent.sh
#
# Usage:
#   cd mimic_iv_sweep_frequent
#   sbatch scripts/sweep_marginalized_binary_k10_n1248.sh
#   sbatch --array=0 scripts/sweep_marginalized_binary_k10_n1248.sh   # N=1 only
# ============================================================

#SBATCH --job-name=sweep-marginalized-binary-k10-n1248-frequent
#SBATCH --array=0-7
#SBATCH --partition=gpu
#SBATCH --account=mm6677_gp
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

IDX=${SLURM_ARRAY_TASK_ID}

NS=(1 2 4 8 16 32 64 128)
N=${NS[$IDX]}

REPO_DIR="${SLURM_SUBMIT_DIR}"
VENV="${REPO_DIR}/.venv/bin/activate"  # created by: cd mimic_iv_sweep_frequent && uv sync

# Shared with mimic_iv_sweep/ -- not rebuilt here.
RETRIEVAL_DB="${REPO_DIR}/../mimic_iv_sweep/data/retrieval_db"
TENSORIZED_DIR="/groups/mm6677_gp/data/MIMIC_MEDS/MEDS_cohort/processed"
LABELS_DIR="${REPO_DIR}/data/tasks/n${N}/tasks"
OUTPUT_DIR="${REPO_DIR}/outputs/marginalized_binary_k10_n1248/n${N}"

cd "${REPO_DIR}"
# shellcheck source=/dev/null
source "${VENV}"
mkdir -p logs "${OUTPUT_DIR}"

if [ ! -d "${RETRIEVAL_DB}" ]; then
    echo "ERROR: ${RETRIEVAL_DB} not found. Run sbatch ../mimic_iv_sweep/scripts/prepare_retrieval.sh first." >&2
    exit 1
fi

echo "=== Job info ==="
echo "  Array job              : ${SLURM_ARRAY_JOB_ID:-local}[${IDX}]"
echo "  Node                   : ${SLURMD_NODENAME:-$(hostname)}"
echo "  GPU(s)                 : ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "  N (tasks)              : ${N}"
echo "  Code selection         : most_frequent"
echo "  marginalized_output_mode: binary (MedRAP#93)"
echo "  retriever.k              : 10 (base sweep uses 4)"
echo "  Started                : $(date)"
echo ""

# Same combo as sweep_marginalized_binary_n1248.sh, except retriever.k=10.
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
    retriever.k=10 \
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
    "output_dir=${OUTPUT_DIR}" \
    do_overwrite=true \
    "wandb_run_name=marginalized-binary-k10-frequent-n${N}-${SLURM_ARRAY_JOB_ID:-local}_${IDX}" \
    "$@"

echo ""
echo "=== Done: $(date) ==="
