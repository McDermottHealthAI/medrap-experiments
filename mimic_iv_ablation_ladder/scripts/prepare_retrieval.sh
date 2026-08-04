#!/bin/bash
# ============================================================
# SLURM: Build HF retrieval corpus + FAISS index (shared across
#        all sweep runs -- only needs to be run once).
# ------------------------------------------------------------
# Uses `medrap-prepare-retrieval-dataset` (flat post-refactor CLI).
# Writes to data/retrieval_db/ by default.
#
# Usage:
#   cd mimic_iv_ablation_ladder
#   sbatch scripts/prepare_retrieval.sh
# ============================================================

#SBATCH --job-name=sweep-prep-retrieval
#SBATCH --partition=gpu
#SBATCH --account=mm6677_gp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR}"
VENV="${REPO_DIR}/.venv/bin/activate"  # created by: cd mimic_iv_ablation_ladder && uv sync

cd "${REPO_DIR}"
# shellcheck source=/dev/null
source "${VENV}"
mkdir -p logs

echo "Node:    $(hostname)"
echo "GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo n/a)"
echo "Start:   $(date)"

RETRIEVAL_DB="${REPO_DIR}/data/retrieval_db"
mkdir -p "${RETRIEVAL_DB}"

medrap-prepare-retrieval-dataset \
  prep.source.path=MedRAG/textbooks prep.source.split=train \
  prep.document.fields='[title,content]' \
  prep.tokenizer.pretrained_model_name_or_path=Qwen/Qwen3-Embedding-0.6B \
  prep.embedder.model_name_or_path=Qwen/Qwen3-Embedding-0.6B prep.embedder.device=cuda \
  prep.index.source_id_column=id \
  "prep.output.output_dir=${RETRIEVAL_DB}" \
  prep.index.max_length=256 \
  prep.index.tokenization_batch_size=512 \
  prep.index.embedding_batch_size=256 \
  prep.index.encode_batch_size=32 \
  "$@"

echo "Done:    $(date)"
