#!/bin/bash
# Task generation and training for the scaling-law grid in design.md.
#
#   scaling_laws/run.sh tasks T DRAW
#       T random 30-day tasks for one draw, then train.parquet subsampled to every D.
#   scaling_laws/run.sh train CONFIG W D T DRAW
#       Train one run, then test it on the top document (K = 1) and, for retrieval,
#       on a random document.
#
#   CONFIG  patient_only | retrieval_learned_query | retrieval_qwen3_query   (#1, #2, #3)
#   W       encoder width, or WIDTHxLAYERSxHEADS (e.g. 128x2x4); default 1 layer, 1 head
#   D       training patients: 1000 3000 10000 30000 100000 153220
#   T       tasks: 5 25 125 625 2500
#   DRAW    1-5 (task seed 101 * DRAW, training seed DRAW)
#
# Paths (environment, defaults shown):
#   COHORT_DIR=data/cohort          medrap-preprocess output (intermediate/ + tensorized/)
#   RETRIEVAL_DB=data/retrieval_db  hf_dataset retrieval DB, including the null document
#   OUTPUT_DIR=outputs
# Needs MedRAP at or after 4a6f4ff (qwen3_token_embeddings, head-bias init). On SLURM, wrap a call, e.g.
#   sbatch --gres=gpu:1 --wrap "scaling_laws/run.sh train patient_only 16 10000 625 1"

set -euo pipefail

COHORT_DIR="${COHORT_DIR:-data/cohort}"
RETRIEVAL_DB="${RETRIEVAL_DB:-data/retrieval_db}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
D_VALUES=(1000 3000 10000 30000 100000 153220)

if [ "$1" = tasks ]; then
    T=$2 DRAW=$3
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
    # Same D patients for every T in a draw; tuning/held_out are copied unchanged.
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

[ "$1" = train ] || { echo "usage: $0 tasks T DRAW | train CONFIG W D T DRAW" >&2; exit 1; }
CONFIG=$2 W=$3 D=$4 T=$5 DRAW=$6
IFS=x read -r WIDTH LAYERS HEADS <<<"${W}"
LABELS="data/tasks${T}_D${D}/draw${DRAW}/tasks"
RUN="${OUTPUT_DIR}/${CONFIG}_W${W}_D${D}_T${T}/draw${DRAW}"

ARGS=(
    encoder=rope
    "encoder.embedding_dim=${WIDTH}"
    "encoder.num_layers=${LAYERS:-1}"
    "encoder.num_heads=${HEADS:-1}"
    "encoder.ff_dim=$((2 * WIDTH))"
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
    training.datamodule.batch_size=32
)
case "${CONFIG}" in
    patient_only)
        ARGS+=(fusion=passthrough "query_projector.in_dim=${WIDTH}" "head.in_dim=${WIDTH}"
            training/loss=multitask_binary_bce)
        ;;
    retrieval_learned_query | retrieval_qwen3_query)
        if [ "${CONFIG}" = retrieval_learned_query ]; then
            ARGS+=(query_projector=sequence_mean_1024 "query_projector.in_dim=${WIDTH}")
        else
            ARGS+=(query_projector=qwen3_text query_projector.device=cuda
                "query_projector.code_metadata_path=${COHORT_DIR}/tensorized/metadata/codes.parquet")
        fi
        ARGS+=(
            retriever=hf_dataset
            "retriever.dataset_path=${RETRIEVAL_DB}"
            retriever.doc_ids_column=null
            retrieval_encoder=qwen3_token_embeddings
            fusion=cross_attention_perdoc_medium
            marginalized_retrieval=true
            marginalized_output_mode=binary
            head.in_dim=256
            training/loss=multitask_binary_bce_marginalized
            "training.loss.num_tasks=${T}"
        )
        ;;
    *) echo "unknown CONFIG ${CONFIG}" >&2; exit 1 ;;
esac

case "${D}" in
    1000) EPOCHS=10 WARMUP=60 ;;
    3000) EPOCHS=7 WARMUP=100 ;;
    10000) EPOCHS=5 WARMUP=200 ;;
    30000) EPOCHS=4 WARMUP=200 ;;
    100000 | 153220) EPOCHS=3 WARMUP=200 ;;
    *) echo "no epoch schedule for D=${D}" >&2; exit 1 ;;
esac
TRAIN_K=()
[ "${CONFIG}" = patient_only ] || TRAIN_K=(retriever.k=4)

medrap-train "${ARGS[@]}" ${TRAIN_K[@]+"${TRAIN_K[@]}"} \
    "training.trainer.max_epochs=${EPOCHS}" \
    "training.module.warmup_steps=${WARMUP}" \
    training.module.lr=1e-3 \
    training.module.init_bias_from_base_rate=true \
    "seed=${DRAW}" \
    "output_dir=${RUN}" \
    do_overwrite=true

TEST_K=()
[ "${CONFIG}" = patient_only ] || TEST_K=(retriever.k=1)
medrap-eval "${ARGS[@]}" ${TEST_K[@]+"${TEST_K[@]}"} eval_mode=test \
    "checkpoint_path=${RUN}/checkpoints/last.ckpt" "output_dir=${RUN}_test" do_overwrite=true
if [ "${CONFIG}" != patient_only ]; then
    medrap-eval "${ARGS[@]}" ${TEST_K[@]+"${TEST_K[@]}"} retriever.ablation_mode=random_docs eval_mode=test \
        "checkpoint_path=${RUN}/checkpoints/last.ckpt" "output_dir=${RUN}_test_random_doc" do_overwrite=true
fi
