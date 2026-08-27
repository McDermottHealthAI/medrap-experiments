#!/bin/bash
# ============================================================
# NWICU STEP-MATCHED (E1, qwen3_text column): identical to the nwicu_sweep
# script of the same name except training.trainer.max_epochs 3 -> 42,
# matching MIMIC's ~14.4k optimizer steps. Fills the qwen3_text column of
# the central table. Full 42 epochs, no early stopping (user decision).
# ~6.5 h/job: the frozen Qwen3 forward taxes every train+val batch.
# ============================================================
# ============================================================
# NWICU REPLICATION of mimic_iv_sweep/scripts/sweep_marginalized_binary_qwen3_text_capacity_starved_n25_30d.sh
# Generated as a mechanical clone: identical Hydra overrides and SBATCH
# resources; only the dataset paths (MIMIC_MEDS -> NWICU), job names, and
# wandb run names differ. See results/capacity_starved_retrieval_nwicu/.
# ============================================================
# ============================================================
# SLURM array: cross-attention marginalized retrieval
#              (marginalized_output_mode=binary, k=4, query_projector=qwen3_text,
#              MedRAP#101) on the Zach anchor_strategy=uniform_event 30d
#              labels, with the patient encoder SEVERELY capacity-starved
#              identically to sweep_patient_only_capacity_starved_n25_30d.sh
#              (encoder.embedding_dim 128->16, num_heads 4->1, num_layers
#              2->1, ff_dim 256->32, max_seq_len 256->32 -- ~8x cut). The
#              frozen Qwen3TextQueryProjector renders queries independently
#              of the patient encoder's hidden state (it re-embeds raw
#              codes as text with a separate frozen Qwen3 model), so it is
#              UNAFFECTED by this cut -- only the encoder capacity and
#              patient-visible history length change, matching
#              patient_only's cut exactly. head.in_dim stays 256 (the
#              fusion's fixed internal d_model), also unaffected.
# ------------------------------------------------------------
# See sweep_patient_only_capacity_starved_n25_30d.sh for the full
# motivation: does a capacity-starved marginalized degrade less than a
# capacity-starved patient_only? If retrieval carries real information,
# this pairing should show a much wider gap than the full-capacity
# comparison in medrap-experiments#14. If both degrade together, the
# retrieved MedRAG/textbooks content isn't adding real information
# regardless of capacity pressure.
#
# 5 array tasks = 5 draws, matching generate_labels_zach_uniform_event_n25_30d.sh.
#
# Prerequisites:
#   sbatch scripts/prepare_retrieval.sh
#   sbatch scripts/generate_labels_zach_uniform_event_n25_30d.sh
#
# Usage:
#   cd nwicu_null_random_doc_ablations
#   sbatch scripts/sweep_marginalized_binary_qwen3_text_capacity_starved_n25_30d.sh
# ============================================================

#SBATCH --job-name=nwicu42-sweep-marginalized-binary-qwen3-text-capacity-starved-n25-30d
#SBATCH --array=0-4
#SBATCH --partition=gpu
#SBATCH --account=mm6677_gp
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=10:00:00
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
CODE_METADATA_PATH="${TENSORIZED_DIR}/metadata/codes.parquet"
LABELS_DIR="${REPO_DIR}/data/tasks_zach_uniform_event_n25_30d/draw${DRAW}/tasks"
OUTPUT_DIR="${REPO_DIR}/outputs/marginalized_binary_qwen3_text_capacity_starved_n25_30d/draw${DRAW}"

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

if [ ! -f "${CODE_METADATA_PATH}" ]; then
    echo "ERROR: ${CODE_METADATA_PATH} not found." >&2
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
echo "  query_projector        : qwen3_text (MedRAP#101)"
echo "  encoder                : embedding_dim=16 num_heads=1 num_layers=1 ff_dim=32, CAPACITY-STARVED"
echo "  max_seq_len             : 32"
echo "  Started                : $(date)"
echo ""

# Same combo as sweep_marginalized_binary_qwen3_text_query_n25_30d.sh --
# only the encoder/max_seq_len capacity cut (and OUTPUT_DIR/wandb_run_name)
# differ here. head.in_dim stays 256 (fusion's fixed d_model, independent
# of encoder.embedding_dim) and query_projector is unaffected (Qwen3
# embeddings come from raw codes, not the encoder's hidden state).
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
    query_projector=qwen3_text \
    query_projector.model_name_or_path=Qwen/Qwen3-Embedding-0.6B \
    "query_projector.code_metadata_path=${CODE_METADATA_PATH}" \
    query_projector.max_codes=32 \
    query_projector.device=cuda \
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
    "wandb_run_name=nwicu42-marginalized-binary-qwen3-text-capacity-starved-d30-draw${DRAW}-${SLURM_ARRAY_JOB_ID:-local}_${IDX}" \
    "$@"

echo ""
echo "=== Done: $(date) ==="
