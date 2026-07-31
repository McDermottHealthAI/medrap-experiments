#!/bin/bash
# ============================================================
# SLURM array: Architecture ablation
# ------------------------------------------------------------
# Compares three architectures at N=1, N=8, and N=32 tasks:
#
#   patient-only   RoPE encoder → masked-mean pooling → head
#                  (no retrieval; fusion=passthrough, k=4 but ignored)
#   retrieval      RoPE + cross-attention medium + k=4 retrieved docs
#   marginalized   same as retrieval but with marginalized_retrieval=true,
#                  per-document fusion (cross_attention_perdoc_medium), and
#                  marginalized BCE loss (multitask_binary_bce_marginalized)
#
# Validation protocol changed here (rationale:
# ../mimic_iv_sweep_frequent/NULL_RESULT_DIAGNOSIS.md):
# training.trainer.limit_val_batches=1.0 scores the FULL tuning split on
# every validation pass instead of the 200-batch (6,400-row) prefix
# conf/training/trainer/lightning_wandb.yaml defaults to, and
# training.trainer.val_check_interval moves 0.2 -> 0.5 to offset the cost
# (two validation passes per epoch instead of five). That prefix held so few
# positives per task that a single positive changing rank could move the
# task's AUROC by up to ~0.5 -- the same size as every arm-vs-arm difference
# this ablation has ever reported. Both settings go in COMMON_ARGS, so all
# three arms share one validation protocol; numbers from this script are NOT
# comparable to results published before this change.
#
# The marginalized arm additionally scores documents by cosine rather than
# dot (marginalized_score_similarity=cosine). The raw dot product left ||q||
# acting as an implicit inverse temperature, so the softmax over the K
# retrieved documents saturated -- effective_k_mean sat at 1.00-1.15 across
# all 32 marginalized runs, making the marginalization arithmetically a
# no-op and driving the gradient into the retrieval scores to exactly zero.
# cosine bounds the scores to [-1, 1] and removes that degree of freedom.
# Same non-comparability caveat applies.
#
# Array index → (N, architecture):
#   0 → N=1,  patient-only
#   1 → N=1,  retrieval
#   2 → N=1,  marginalized
#   3 → N=8,  patient-only
#   4 → N=8,  retrieval
#   5 → N=8,  marginalized
#   6 → N=32, patient-only
#   7 → N=32, retrieval
#   8 → N=32, marginalized
#
# Prerequisites:
#   sbatch scripts/prepare_retrieval.sh
#   sbatch --array=0,3,5 scripts/generate_labels.sh  # N=1 (idx 0), N=8 (idx 3), N=32 (idx 5)
#
# Usage:
#   cd mimic_iv_sweep
#   sbatch scripts/sweep_architecture.sh
#   sbatch --array=3-5 scripts/sweep_architecture.sh   # N=8 only
# ============================================================

#SBATCH --job-name=sweep-arch
#SBATCH --array=0-8
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

NS=(1 1 1 8 8 8 32 32 32)
ARCHS=(patient_only retrieval marginalized patient_only retrieval marginalized patient_only retrieval marginalized)

N=${NS[$IDX]}
ARCH=${ARCHS[$IDX]}

REPO_DIR="${SLURM_SUBMIT_DIR}"
VENV="${REPO_DIR}/.venv/bin/activate"  # created by: cd mimic_iv_sweep && uv sync

RETRIEVAL_DB="${REPO_DIR}/data/retrieval_db"
TENSORIZED_DIR="/groups/mm6677_gp/data/MIMIC_MEDS/MEDS_cohort/processed"
LABELS_DIR="${REPO_DIR}/data/tasks/n${N}/tasks"
OUTPUT_DIR="${REPO_DIR}/outputs/architecture/${ARCH}_n${N}"

cd "${REPO_DIR}"
# shellcheck source=/dev/null
source "${VENV}"
mkdir -p logs "${OUTPUT_DIR}"

echo "=== Job info ==="
echo "  Array job  : ${SLURM_ARRAY_JOB_ID:-local}[${IDX}]"
echo "  Node       : ${SLURMD_NODENAME:-$(hostname)}"
echo "  GPU(s)     : ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "  N (tasks)  : ${N}"
echo "  Architecture: ${ARCH}"
echo "  Started    : $(date)"
echo ""

# --- shared args used by all architecture variants ---
COMMON_ARGS=(
    encoder=rope
    pooling=masked_mean
    head=linear
    "head.out_dim=${N}"
    training/task=multitask_binary
    "training.task.num_tasks=${N}"
    training/datamodule=meds_multitask
    "training.datamodule.config.tensorized_cohort_dir=${TENSORIZED_DIR}"
    training.datamodule.config.max_seq_len=256
    "training.datamodule.config.task_labels_dir=${LABELS_DIR}"
    "training.datamodule.mt_labels_dir=${LABELS_DIR}"
    "training.datamodule.num_tasks=${N}"
    training.datamodule.batch_size=32
    training.datamodule.config.seq_sampling_strategy=to_end
    training/trainer=lightning_wandb
    training.trainer.max_epochs=3
    training.trainer.accelerator=gpu
    training.trainer.devices=1
    training.trainer.gradient_clip_val=1.0
    training.trainer.log_every_n_steps=10
    training.trainer.limit_val_batches=1.0
    training.trainer.val_check_interval=0.5
    training.module.lr=1e-3
    training.module.warmup_steps=200
    "output_dir=${OUTPUT_DIR}"
    do_overwrite=true
)

if [[ "${ARCH}" == "patient_only" ]]; then
    # No retrieval: passthrough fusion ignores any retrieved docs. head.in_dim
    # matches the rope encoder output (128). model.forward() calls
    # query_projector(encoder_out) unconditionally even when fusion=passthrough
    # discards its output, so query_projector.in_dim must still match the rope
    # encoder's 128-dim output -- the default query_projector config's in_dim=1
    # crashes on the first forward pass otherwise (mat1/mat2 shape mismatch).
    medrap-train \
        "${COMMON_ARGS[@]}" \
        fusion=passthrough \
        query_projector.in_dim=128 \
        head.in_dim=128 \
        training/loss=multitask_binary_bce \
        "wandb_run_name=sweep-arch-${ARCH}-n${N}-${SLURM_ARRAY_JOB_ID:-local}_${IDX}" \
        "$@"

elif [[ "${ARCH}" == "retrieval" ]]; then
    # Standard retrieval: RoPE + cross-attention medium + k=4.
    # head.in_dim matches cross_attention_medium d_model (256).
    medrap-train \
        "${COMMON_ARGS[@]}" \
        query_projector=sequence_mean_1024 \
        query_projector.in_dim=128 \
        retriever=hf_dataset \
        "retriever.dataset_path=${RETRIEVAL_DB}" \
        retriever.doc_ids_column=null \
        retriever.k=4 \
        retrieval_encoder=token_feature \
        retrieval_encoder.vocab_size=151936 \
        retrieval_encoder.embedding_dim=64 \
        fusion=cross_attention_medium \
        head.in_dim=256 \
        training/loss=multitask_binary_bce \
        "wandb_run_name=sweep-arch-${ARCH}-n${N}-${SLURM_ARRAY_JOB_ID:-local}_${IDX}" \
        "$@"

elif [[ "${ARCH}" == "marginalized" ]]; then
    # Marginalized retrieval: requires marginalized_retrieval=true and a fusion
    # module with produces_per_document_state=True (cross_attention_perdoc_medium,
    # not cross_attention_medium) so the model emits per-doc logits for
    # MultiTaskBCEMarginalizedLoss to marginalize over.
    medrap-train \
        "${COMMON_ARGS[@]}" \
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
        marginalized_score_similarity=cosine \
        head.in_dim=256 \
        training/loss=multitask_binary_bce_marginalized \
        "training.loss.num_tasks=${N}" \
        "wandb_run_name=sweep-arch-${ARCH}-n${N}-${SLURM_ARRAY_JOB_ID:-local}_${IDX}" \
        "$@"
fi

echo ""
echo "=== Done: $(date) ==="
