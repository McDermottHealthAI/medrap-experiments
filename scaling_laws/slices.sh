#!/bin/bash
# Set up scaling_laws/ in a fresh clone and submit the slices to SLURM: the Qwen3-query slice (#3,
# 240 runs) and the random-documents control (#4, 45 runs). The main grid (#1, #2) runs elsewhere.
#
#   ACCOUNT=<slurm account> GPU_PARTITION=<partition> DATA=<data dir> bash slices.sh [DIR]
#
#   DIR            where to clone (default ./medrap-experiments-slices); rerun with the same DIR to
#                  resubmit failed or unsubmitted runs (finished and queued runs are skipped)
#   DATA           scaling_laws/data of the main run, or a copy of it (cohort/tensorized,
#                  retrieval_db, tasks25_D*, tasks625_D*); default: the C2B2 copy
#   GPU_PARTITION  L40S/A100-class GPUs; the faiss-gpu wheels fail on H100 (CUDA error 209)
#
# Needs git, uv, and network access for the first run (Python packages, Qwen3-Embedding-0.6B).
# Results land in DIR/scaling_laws/outputs/; send that folder back (it is small: CSV logs + configs).

set -euo pipefail
: "${ACCOUNT:?set ACCOUNT to your SLURM account}"
: "${GPU_PARTITION:?set GPU_PARTITION to an L40S or A100 partition}"
DATA="${DATA:-/groups/mm6677_gp/hs3627/medrap-experiments-scaling-laws-design/scaling_laws/data}"
DIR="${1:-medrap-experiments-slices}"

[ -d "${DIR}" ] || git clone -b scaling-laws-design \
    https://github.com/McDermottHealthAI/medrap-experiments.git "${DIR}"
cd "${DIR}/scaling_laws"
uv sync

mkdir -p data/cohort
[ -e data/cohort/tensorized ] || ln -s "${DATA}/cohort/tensorized" data/cohort/tensorized
for src in "${DATA}/retrieval_db" "${DATA}"/tasks25_D* "${DATA}"/tasks625_D*; do
    [ -e "data/$(basename "${src}")" ] || ln -s "${src}" "data/$(basename "${src}")"
done

.venv/bin/python -c "from transformers import AutoModel, AutoTokenizer as T; \
    m = 'Qwen/Qwen3-Embedding-0.6B'; AutoModel.from_pretrained(m); T.from_pretrained(m)"

ACCOUNT="${ACCOUNT}" GPU_PARTITION="${GPU_PARTITION}" ./submit.sh slices
