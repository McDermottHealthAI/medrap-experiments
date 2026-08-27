#!/bin/bash
# ============================================================
# NWICU STEP-MATCHED (E1): identical to the nwicu_sweep script of the same
# name except training.trainer.max_epochs 3 -> 42, matching MIMIC's ~14.4k
# optimizer steps (343 steps/epoch x 42 = 14,406 vs MIMIC 14,364). Tests
# whether NWICU's machinery-not-content ablation verdict was caused by
# undertraining. See results/mimic_iv_explore/.
# ============================================================
# ============================================================
# NWICU REPLICATION of mimic_iv_sweep/scripts/sweep_marginalized_binary_learned_linear_capacity_starved_n25_30d.sh
# Generated as a mechanical clone: identical Hydra overrides and SBATCH
# resources; only the dataset paths (MIMIC_MEDS -> NWICU), job names, and
# wandb run names differ. See results/capacity_starved_retrieval_nwicu/.
# ============================================================
# ============================================================
# SLURM array: cross-attention marginalized retrieval
#              (marginalized_output_mode=binary, k=4, query_projector=
#              sequence_mean_1024 -- the ORIGINAL learned linear query
#              projector, PRE-MedRAP#101) on the Zach
#              anchor_strategy=uniform_event 30d labels, with the patient
#              encoder SEVERELY capacity-starved identically to
#              sweep_patient_only_capacity_starved_n25_30d.sh and
#              sweep_marginalized_binary_qwen3_text_capacity_starved_n25_30d.sh
#              (encoder.embedding_dim 128->16, num_heads 4->1, num_layers
#              2->1, ff_dim 256->32, max_seq_len 256->32 -- ~8x cut).
# ------------------------------------------------------------
# Companion to sweep_marginalized_binary_qwen3_text_capacity_starved_n25_30d.sh,
# NOT a replacement for it -- runs in parallel with the same starved
# patient_only baseline. Motivation for running this variant too: the
# frozen Qwen3-Embedding-0.6B forward pass in query_projector=qwen3_text is
# a FIXED per-batch cost independent of the patient encoder's capacity (it
# re-embeds rendered code text with a separate 0.6B model, not the tiny
# MedRAP encoder), so the qwen3_text-starved sweep barely trains any faster
# than its full-capacity counterpart despite the ~8x model shrink (~68
# steps/min full-capacity vs. ~77 steps/min starved, observed on the
# canary -- the capacity cut only shaves the small compute-only part of
# each step). query_projector=sequence_mean_1024 (a plain learned linear
# layer, no live embedding calls) has no such fixed cost, so THIS variant
# should train much faster and gives capacity-matched, wall-clock-cheap
# read on the same starvation hypothesis using a query projector that,
# unlike the frozen Qwen3 one, actually shrinks proportionally with the
# rest of the model. Since query_projector.in_dim is set to
# encoder.embedding_dim (16 here, same coupling as the original
# sweep_marginalized_binary_zach_uniform_event_n25_30d.sh baseline), this
# is a genuinely capacity-matched comparison to starved patient_only in a
# way the qwen3_text variant (whose query projector doesn't shrink at all)
# is not.
#
# Uses the SAME starved patient_only baseline
# (sweep_patient_only_capacity_starved_n25_30d.sh) as the qwen3_text-starved
# comparison -- patient_only doesn't use a query projector at all
# (fusion=passthrough), so it's shared across both marginalized variants.
#
# 5 array tasks = 5 draws, matching generate_labels_zach_uniform_event_n25_30d.sh.
#
# Prerequisites:
#   sbatch scripts/prepare_retrieval.sh
#   sbatch scripts/generate_labels_zach_uniform_event_n25_30d.sh
#
# Usage:
#   cd nwicu_null_random_doc_ablations
#   sbatch scripts/sweep_marginalized_binary_learned_linear_capacity_starved_n25_30d.sh
# ============================================================

#SBATCH --job-name=nwicu42-sweep-marginalized-binary-learned-linear-capacity-starved-n25-30d
#SBATCH --array=0-4
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
DRAW=$((IDX + 1))
N=25

REPO_DIR="${SLURM_SUBMIT_DIR}"
VENV="${REPO_DIR}/.venv/bin/activate"  # created by: cd nwicu_null_random_doc_ablations && uv sync

RETRIEVAL_DB="${REPO_DIR}/data/retrieval_db"
TENSORIZED_DIR="/groups/mm6677_gp/data/NWICU/processed"
LABELS_DIR="${REPO_DIR}/data/tasks_zach_uniform_event_n25_30d/draw${DRAW}/tasks"
OUTPUT_DIR="${REPO_DIR}/outputs/marginalized_binary_learned_linear_capacity_starved_n25_30d/draw${DRAW}"

cd "${REPO_DIR}"
# shellcheck source=/dev/null
source "${VENV}"
mkdir -p logs "${OUTPUT_DIR}"

if [ ! -d "${RETRIEVAL_DB}" ]; then
    echo "ERROR: ${RETRIEVAL_DB} not found. Run sbatch scripts/prepare_retrieval.sh first." >&2
    exit 1
fi

if [ ! -d "${LABELS_DIR}" ]; then
    echo "ERROR: ${LABELS_DIR} not found. Run sbatch scripts/generate_labels_zach_uniform_event_n25_30d.sh for draw=${DRAW} first." >&2
    exit 1
fi

echo "=== Job info ==="
echo "  Array job              : ${SLURM_ARRAY_JOB_ID:-local}[${IDX}]"
echo "  Node                   : ${SLURMD_NODENAME:-$(hostname)}"
echo "  GPU(s)                 : ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "  N (tasks)              : ${N}"
echo "  Duration               : 30d"
echo "  Draw                   : ${DRAW}/5"
echo "  marginalized_output_mode: binary (MedRAP#93)"
echo "  query_projector        : sequence_mean_1024 (original learned linear)"
echo "  encoder                : embedding_dim=16 num_heads=1 num_layers=1 ff_dim=32, CAPACITY-STARVED"
echo "  max_seq_len             : 32"
echo "  Started                : $(date)"
echo ""

# Same combo as sweep_marginalized_binary_zach_uniform_event_n25_30d.sh --
# only the encoder/max_seq_len capacity cut + query_projector.in_dim (and
# OUTPUT_DIR/wandb_run_name) differ here.
medrap-train \
    encoder=rope \
    encoder.embedding_dim=16 \
    encoder.num_heads=1 \
    encoder.num_layers=1 \
    encoder.ff_dim=32 \
    pooling=masked_mean \
    head=linear \
    "head.out_dim=${N}" \
    training/task=multitask_binary \
    "training.task.num_tasks=${N}" \
    training/datamodule=meds_multitask \
    "training.datamodule.config.tensorized_cohort_dir=${TENSORIZED_DIR}" \
    training.datamodule.config.max_seq_len=32 \
    "training.datamodule.config.task_labels_dir=${LABELS_DIR}" \
    "training.datamodule.mt_labels_dir=${LABELS_DIR}" \
    "training.datamodule.num_tasks=${N}" \
    training.datamodule.batch_size=32 \
    training.datamodule.config.seq_sampling_strategy=to_end \
    training/trainer=lightning_wandb \
    training.trainer.max_epochs=42 \
    training.trainer.accelerator=gpu \
    training.trainer.devices=1 \
    training.trainer.gradient_clip_val=1.0 \
    training.trainer.log_every_n_steps=10 \
    training.module.lr=1e-3 \
    training.module.warmup_steps=200 \
    training.module.validation_auroc_log_per_task=true \
    query_projector=sequence_mean_1024 \
    query_projector.in_dim=16 \
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
    head.in_dim=256 \
    training/loss=multitask_binary_bce_marginalized \
    "training.loss.num_tasks=${N}" \
    "output_dir=${OUTPUT_DIR}" \
    do_overwrite=true \
    "wandb_run_name=nwicu42-marginalized-binary-learned-linear-capacity-starved-d30-draw${DRAW}-${SLURM_ARRAY_JOB_ID:-local}_${IDX}" \
    "$@"

echo ""
echo "=== Done: $(date) ==="
