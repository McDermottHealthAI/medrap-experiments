#!/bin/bash
# Data preparation, training, prediction and evaluation for design.md.
#
#   everyquery/run.sh preprocess                 raw MEDS -> event shards + tensorized cohort
#   everyquery/run.sh labels DRAW                training (seed DRAW) and tuning labels
#   everyquery/run.sh eval_grid                  1,000 query specs on held-out patients (shared)
#   everyquery/run.sh encode_docs                document keys index + contextual memory (GPU)
#   everyquery/run.sh train CONFIG SIZE LR DRAW  train, predict on the grid, evaluate
#
#   CONFIG  patient_only  (retrieval configurations follow the objective decision, design.md §6)
#   SIZE    small | base | large   (design.md §7)
#   LR      learning rate, e.g. 2e-4
#   DRAW    1-3
#
# Paths (environment, defaults shown), relative to everyquery/:
#   MEDS_DIR=/groups/mm6677_gp/data/MIMIC_MEDS/MEDS_cohort   raw MEDS (data/ + metadata/)
#   RETRIEVAL_DB=../scaling_laws/data/retrieval_db          textbook passages + null document
#   N_TRAIN=5000000                                         training contexts per draw
# everyquery/.venv is used when it exists (uv sync).

set -euo pipefail
cd "$(dirname "$0")"
# shellcheck source=/dev/null
[ ! -f .venv/bin/activate ] || source .venv/bin/activate
export WANDB_MODE=offline

MEDS_DIR="${MEDS_DIR:-/groups/mm6677_gp/data/MIMIC_MEDS/MEDS_cohort}"
RETRIEVAL_DB="${RETRIEVAL_DB:-../scaling_laws/data/retrieval_db}"
N_TRAIN="${N_TRAIN:-5000000}"
EVENTS=data/events
COHORT=data/cohort

case "${1:-}" in
    preprocess)
        EQ_process_data "input_dir=${MEDS_DIR}" "intermediate_dir=${EVENTS}" "output_dir=${COHORT}"
        ;;
    labels)
        DRAW=$2
        for SPLIT in train tuning; do
            EQ_generate_multitask_sequences "data_dir=${EVENTS}" "out_dir=data/labels/draw${DRAW}" \
                "query_codes=${COHORT}" "split=${SPLIT}" "seed=${DRAW}" \
                "num_training_examples=${N_TRAIN}" overwrite=true
        done
        ;;
    eval_grid)
        EQ_generate_evaluation_query_sequences "data_dir=${EVENTS}" out_dir=data/eval_grid \
            "query_codes=${COHORT}" split=held_out seed=1 num_evaluation_sequences=1000 overwrite=true
        ;;
    encode_docs)
        python encode_docs.py "${RETRIEVAL_DB}" data/docs
        ;;
    train)
        CONFIG=$2 SIZE=$3 LR=$4 DRAW=$5
        case "${SIZE}" in
            small) LAYERS=4 HIDDEN=256 HEADS=4 FF=1024 ;;
            base) LAYERS=12 HIDDEN=384 HEADS=6 FF=1536 ;;
            large) LAYERS=12 HIDDEN=512 HEADS=8 FF=2048 ;;
            *) echo "unknown SIZE ${SIZE}" >&2; exit 1 ;;
        esac
        [ "${CONFIG}" = patient_only ] || { echo "CONFIG ${CONFIG} not implemented yet (design.md §6)" >&2; exit 1; }
        RUN="outputs/${CONFIG}_${SIZE}_lr${LR}/draw${DRAW}"
        mkdir -p "${RUN}"
        EQ_train --config-name=conditional_multitask_ar_config \
            "output_dir=${RUN}" "hydra.run.dir=${RUN}" do_overwrite=true "seed=${DRAW}" \
            "datamodule.config.tensorized_cohort_dir=${COHORT}" \
            "datamodule.config.task_labels_dir=data/labels/draw${DRAW}" \
            "lightning_module.model.config_overrides.num_hidden_layers=${LAYERS}" \
            "lightning_module.model.config_overrides.hidden_size=${HIDDEN}" \
            "lightning_module.model.config_overrides.num_attention_heads=${HEADS}" \
            "lightning_module.model.config_overrides.num_key_value_heads=${HEADS}" \
            "lightning_module.model.config_overrides.intermediate_size=${FF}" \
            "lightning_module.optimizer.lr=${LR}" \
            '~trainer.logger' \
            '+trainer.logger={_target_: lightning.pytorch.loggers.CSVLogger, save_dir: ${trainer.default_root_dir}/loggers}'
        EQ_predict_multitask "model_run_dir=${RUN}" tasks_dir=data/eval_grid/eval \
            "output_parquet=${RUN}/predictions.parquet" split=held_out enable_progress_bar=false overwrite=true
        EQ_evaluate_multitask "predictions_parquet=${RUN}/predictions.parquet" "metrics_stem=${RUN}/metrics"
        ;;
    *)
        echo "usage: $0 preprocess | labels DRAW | eval_grid | encode_docs | train CONFIG SIZE LR DRAW" >&2
        exit 1
        ;;
esac
