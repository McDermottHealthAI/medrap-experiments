#!/bin/bash
# Task generation and training for scaling_laws_v2 (design.md).
#
#   scaling_laws_v2/run.sh tasks DRAW
#       T = 1,000 random 30-day tasks for one draw, then train.parquet subsampled to every D.
#   scaling_laws_v2/run.sh train CONFIG W D DRAW
#       Train one run (checkpoints kept), then test it.
#
#   CONFIG  patient_only     #1 (rung A0)
#           v1_retrieval     v1's #2 (per-document cross-attention, learned query), v1 code (A1)
#           v1_random_docs   v1's #4 (as v1_retrieval, random documents), v1 code (reference)
#           v1_masked        v1's #2 with the masked mean over codes, MedRAP#123 (A2)
#           v1_random_docs_masked  v1_masked with random documents (ablation reference)
#           retrieval        fixed #2: gated fusion, starts as #1, MedRAP#124 (A3)
#           random_docs      fixed #2 with random documents (A4)
#           retrieval_only   fixed #R: the documents read with the retrieval query, no patient (A5)
#   W       WIDTHxLAYERSxHEADS, e.g. 128x2x4
#   D       training patients: 3000 10000 30000 100000 153220
#   DRAW    1-3 (task seed 101 * DRAW, training seed DRAW)
#
# Settings (environment, defaults shown; Stage 0 picks LR, DROPOUT and the checkpoint):
#   LR=1e-3  DROPOUT=0  K=4 (documents in training and in the main test, plus the null document)
#   CKPT=last (retrieval runs; patient_only tests both best and last)
#
# Tests, each in RUN/test_*: patient_only: best, last. Retrieval: top K + null (as trained), top 1 +
# null (v1's test), and K random documents + null (reliance on the retrieved documents).
#
# Environments: v1_retrieval and v1_random_docs use ../scaling_laws/.venv (MedRAP c3700ab, v1's code);
# everything else uses .venv here (pyproject.toml: MedRAP with the fixes). The cohort and retrieval
# database are v1's.

set -euo pipefail
# Bash reads a script while running it, so a git pull mid-job breaks running jobs ("Stale file
# handle"): run from a private copy.
if [ -z "${RUN_SH_COPY:-}" ]; then
    cd "$(dirname "$0")"
    RUN_SH_COPY=$(mktemp)
    cp run.sh "${RUN_SH_COPY}"
    export RUN_SH_COPY
    exec bash "${RUN_SH_COPY}" "$@"
fi
trap 'rm -f "${RUN_SH_COPY}"' EXIT
VENV=.venv
case "${1:-} ${2:-}" in "train v1_retrieval" | "train v1_random_docs") VENV=../scaling_laws/.venv ;; esac
# shellcheck source=/dev/null
[ ! -f "${VENV}/bin/activate" ] || source "${VENV}/bin/activate"
export WANDB_MODE="${WANDB_MODE:-offline}"

COHORT_DIR="${COHORT_DIR:-../scaling_laws/data/cohort}"
RETRIEVAL_DB="${RETRIEVAL_DB:-../scaling_laws/data/retrieval_db}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
# Override only for smoke tests on v1 labels (N_TASKS=625 OUTPUT_DIR=smoke). Not T: nodes set T=/tmp/.
T="${N_TASKS:-1000}"
D_VALUES=(3000 10000 30000 100000 153220)
LR="${LR:-1e-3}" DROPOUT="${DROPOUT:-0}" K="${K:-4}" CKPT="${CKPT:-last}"
# NULL_DOC=false drops the null document; with K=1 there is then no average over documents.
NULL_DOC="${NULL_DOC:-true}"
# Ablations of the v1 fusion (v1_masked), each true/false, each adding a suffix to the run name:
#   ZERO_INIT  cross-attention output projections start at zero (_z)
#   WARM       encoder starts from the patient-only run with the same W, D, lr, dropout, draw (_warm)
#   CENTER     search centered, unit-length keys: data/retrieval_db_centered from center_keys.py (_c)
ZERO_INIT="${ZERO_INIT:-false}" WARM="${WARM:-false}" CENTER="${CENTER:-false}"
# What the fusion reads from each retrieved document (GOAL.md, richer documents); suffix _DOCS:
#   e0      Qwen3 input token embeddings, no context (v1)
#   attend  a frozen Qwen3 runs over the document; fusion attends over every token of every layer
#           (LAYER_STRIDE=4 keeps every 4th layer; suffix _attend or _attend4)
#   last    the same, last-layer output per token only
#   mix     the same, a learned mix of all layers per token (ELMo)
#   pre     last-layer vectors precomputed once (encode_last_layer.py, data/docs_last_fp16.npy, 66 GB
#           held in RAM), looked up by retrieved row: as fast as e0, so K can be large
DOCS="${DOCS:-e0}" LAYER_STRIDE="${LAYER_STRIDE:-1}"
# ACCUM=2 halves the batch and accumulates gradients over 2 batches: the same effective batch of 32
# and the same optimizer steps, at half the memory (K 128 does not fit a 44 GB GPU otherwise).
ACCUM="${ACCUM:-1}"
[ "${CENTER}" = false ] || RETRIEVAL_DB=data/retrieval_db_centered

if [ "$1" = tasks ]; then
    DRAW=$2
    FULL="data/tasks${T}/draw${DRAW}"
    medrap-preprocess \
        "meds_data_dir=${COHORT_DIR}/intermediate" \
        "tensorized_dir=${COHORT_DIR}/tensorized" \
        "output_dir=${FULL}" \
        "num_tasks=${T}" \
        horizon_days=30.0 \
        min_history_days=1.0 \
        "seed=$((101 * DRAW))" \
        code_selection=random \
        anchor_strategy=uniform_event \
        do_overwrite=true
    # Same patients for every configuration in a draw; tuning/held_out are copied unchanged.
    for D in "${D_VALUES[@]}"; do
        python - "${FULL}/tasks" "data/tasks${T}_D${D}/draw${DRAW}/tasks" "${D}" "$((1000000 * DRAW + D))" <<'EOF'
import shutil, sys
from pathlib import Path
import polars as pl
src, dst, n, seed = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
dst.mkdir(parents=True, exist_ok=True)
train = pl.read_parquet(src / "train.parquet")
train.sample(n=min(n, train.height), seed=seed, shuffle=True).write_parquet(dst / "train.parquet")
for name in ("tuning.parquet", "held_out.parquet", "code_index.json", "metadata.json"):
    if (src / name).exists():
        shutil.copy2(src / name, dst / name)
EOF
    done
    exit 0
fi

[ "$1" = train ] || { echo "usage: $0 tasks DRAW | train CONFIG W D DRAW" >&2; exit 1; }
CONFIG=$2 W=$3 D=$4 DRAW=$5
IFS=x read -r WIDTH LAYERS HEADS <<<"${W}"
LABELS="data/tasks${T}_D${D}/draw${DRAW}/tasks"
RUN="${OUTPUT_DIR}/${CONFIG}_W${W}_D${D}_lr${LR}_do${DROPOUT}"
[ "${CONFIG}" = patient_only ] || RUN="${RUN}_K${K}"
[ "${NULL_DOC}" = true ] || RUN="${RUN}_nonull"
[ "${ZERO_INIT}" = false ] || RUN="${RUN}_z"
[ "${WARM}" = false ] || RUN="${RUN}_warm"
[ "${CENTER}" = false ] || RUN="${RUN}_c"
[ "${DOCS}" = e0 ] || RUN="${RUN}_${DOCS}"
[ "${LAYER_STRIDE}" = 1 ] || RUN="${RUN}${LAYER_STRIDE}"
RUN="${RUN}/draw${DRAW}"

ARGS=(
    encoder=rope
    "encoder.embedding_dim=${WIDTH}"
    "encoder.num_layers=${LAYERS:-1}"
    "encoder.num_heads=${HEADS:-1}"
    "encoder.ff_dim=$((2 * WIDTH))"
    "encoder.dropout=${DROPOUT}"
    pooling=masked_mean
    head=linear
    "head.out_dim=${T}"
    training/task=multitask_binary
    "training.task.num_tasks=${T}"
    training/datamodule=meds_multitask
    "training.datamodule.config.tensorized_cohort_dir=${COHORT_DIR}/tensorized"
    "training.datamodule.config.task_labels_dir=${LABELS}"
    "training.datamodule.mt_labels_dir=${LABELS}"
    "training.datamodule.num_tasks=${T}"
    training.datamodule.config.max_seq_len=256
    training.datamodule.config.seq_sampling_strategy=to_end
    "training.datamodule.batch_size=$((32 / ACCUM))"
    training/trainer=lightning_wandb
    training.trainer.accelerator=gpu
    # Keep only the last checkpoint (Stage 0: best = last in every run); no progress snapshots.
    +training.trainer.callbacks.0.save_top_k=0
    "training.trainer.callbacks.1.fractions=[]"
)
case "${CONFIG}" in
    patient_only)
        ARGS+=(fusion=passthrough "query_projector.in_dim=${WIDTH}" "head.in_dim=${WIDTH}"
            training/loss=multitask_binary_bce)
        ;;
    v1_retrieval | v1_random_docs | v1_masked | v1_random_docs_masked | retrieval | random_docs | retrieval_only \
        | residual | residual_random_docs | xres | xres_random_docs)
        case "${CONFIG}" in *random_docs*) ARGS+=(retriever.ablation_mode=random_docs) ;; esac
        ARGS+=(
            query_projector=sequence_mean_1024 "query_projector.in_dim=${WIDTH}"
            retriever=hf_dataset
            "retriever.dataset_path=${RETRIEVAL_DB}"
            retriever.doc_ids_column=null
            "retriever.append_null_doc=${NULL_DOC}"
            retriever.device=0
            retriever.cache_payloads=true
            retriever.payload_cache_device=cuda
            marginalized_retrieval=true
            marginalized_output_mode=binary
            training/loss=multitask_binary_bce_marginalized
            "training.loss.num_tasks=${T}"
        )
        case "${DOCS}" in
            e0) ARGS+=(retrieval_encoder=qwen3_token_embeddings) ;;
            pre) ARGS+=(retrieval_encoder=precomputed_token_states retrieval_encoder.path=data/docs_last_fp16.npy) ;;
            attend | last | mix) ARGS+=(retrieval_encoder=qwen3_hidden_states "retrieval_encoder.layers=${DOCS}"
                "retrieval_encoder.layer_stride=${LAYER_STRIDE}") ;;
            *) echo "unknown DOCS ${DOCS}" >&2; exit 1 ;;
        esac
        case "${CONFIG}" in
            v1_*) ARGS+=(fusion=cross_attention_perdoc_medium "fusion.dropout=${DROPOUT}" head.in_dim=256)
                [ "${ZERO_INIT}" = false ] || ARGS+=(fusion.zero_init_attn_out=true) ;;
            # xres: v1's per-code reader on the encoder's own states (no projection or output norm),
            # so the head reads the patient-only input plus the documents' updates.
            xres*) ARGS+=(fusion=cross_attention_perdoc_medium fusion.residual=true "fusion.d_model=${WIDTH}"
                "fusion.num_heads=${HEADS:-1}" "fusion.ff_dim=$((2 * WIDTH))" "fusion.dropout=${DROPOUT}"
                "head.in_dim=${WIDTH}") ;;
            *) ARGS+=(fusion=gated_doc "fusion.dropout=${DROPOUT}" "head.in_dim=${WIDTH}") ;;
        esac
        [ "${CONFIG}" != retrieval_only ] || ARGS+=(fusion.mode=query)
        # residual: the patient-only head input h plus each document's contribution, no gate.
        case "${CONFIG}" in residual*) ARGS+=(fusion.gated=false) ;; esac
        ;;
    *) echo "unknown CONFIG ${CONFIG}" >&2; exit 1 ;;
esac

case "${D}" in
    3000) EPOCHS=7 WARMUP=100 ;;
    10000) EPOCHS=5 WARMUP=200 ;;
    30000) EPOCHS=4 WARMUP=200 ;;
    100000 | 153220) EPOCHS=3 WARMUP=200 ;;
    *) echo "no epoch schedule for D=${D}" >&2; exit 1 ;;
esac
# The faiss-gpu wheels have no H100 kernels (CUDA error 209): search on the CPU there.
if nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | grep -q H100; then
    ARGS=("${ARGS[@]/#retriever.device=0/retriever.device=null}")
fi
NAME="${RUN//\//_}"
if [ "${WARM}" = true ]; then
    WARM_FROM="${OUTPUT_DIR}/patient_only_W${W}_D${D}_lr${LR}_do${DROPOUT}/draw${DRAW}/train/checkpoints/last.ckpt"
    [ -f "${WARM_FROM}" ] || { echo "missing ${WARM_FROM}" >&2; exit 1; }
    ARGS+=("training.module.init_encoder_from='${WARM_FROM}'")
fi
TRAIN_K=()
[ "${CONFIG}" = patient_only ] || TRAIN_K=("retriever.k=${K}")

# TEST_ONLY=true reruns only the tests of a finished training run (e.g. after a filesystem error).
[ "${TEST_ONLY:-false}" = true ] || medrap-train "${ARGS[@]}" ${TRAIN_K[@]+"${TRAIN_K[@]}"} \
    "+training.trainer.accumulate_grad_batches=${ACCUM}" \
    "training.trainer.max_epochs=${EPOCHS}" \
    "training.module.warmup_steps=${WARMUP}" \
    "training.module.lr=${LR}" \
    training.module.init_bias_from_base_rate=true \
    "seed=${DRAW}" \
    "output_dir=${RUN}/train" \
    "wandb_run_name=v2-${NAME}" \
    do_overwrite=true

BEST="${RUN}/train/checkpoints/last.ckpt"  # save_top_k=0: the last checkpoint is the only one
test_run() {  # NAME CHECKPOINT EXTRA_ARGS...
    local name=$1 ckpt=$2
    shift 2
    # Quoted for Hydra: best-checkpoint names contain '=' (epoch=5-step=542.ckpt).
    medrap-eval "${ARGS[@]}" "$@" eval_mode=test \
        "checkpoint_path='${ckpt}'" \
        "output_dir=${RUN}/test_${name}" \
        "wandb_run_name=v2-${NAME}-${name}" \
        do_overwrite=true
}
if [ "${CONFIG}" = patient_only ]; then
    test_run best "${BEST}"
    test_run last "${RUN}/train/checkpoints/last.ckpt"
else
    CK="${RUN}/train/checkpoints/last.ckpt"
    [ "${CKPT}" = last ] || CK="${BEST}"
    test_run topK "${CK}" "retriever.k=${K}"
    # The random-documents configurations already test on random documents (their ARGS).
    case "${CONFIG}" in
        *random_docs*) ;;
        *)
            test_run top1 "${CK}" retriever.k=1
            test_run randomK "${CK}" "retriever.k=${K}" retriever.ablation_mode=random_docs
            ;;
    esac
fi
# The group quota is shared: drop the frozen Qwen table and optimizer state once tests are done.
# medrap-train also copies the best checkpoint to train/best_model.ckpt; the last one is kept instead.
rm -f "${RUN}/train/best_model.ckpt"
python slim_checkpoint.py "${RUN}/train/checkpoints/last.ckpt"
