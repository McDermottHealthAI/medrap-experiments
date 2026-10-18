# Model + data overrides shared by train.sh (training and its test eval) and
# predict_probabilities.sh, so every command rebuilds exactly the same model.
# Sourced, not run. Expects ARM, DIM, NUM_TASKS, TENSORIZED_DIR, LABELS_DIR, RETRIEVAL_DB and
# REPO_DIR to be set; sets MODEL_ARGS (no trainer settings -- callers add their own).

MODEL_ARGS=(
    encoder=rope
    "encoder.embedding_dim=${DIM}"
    encoder.num_heads=1
    encoder.num_layers=1
    "encoder.ff_dim=$((2 * DIM))"
    pooling=masked_mean
    head=linear
    "head.out_dim=${NUM_TASKS}"
    training/task=multitask_binary
    "training.task.num_tasks=${NUM_TASKS}"
    training/datamodule=meds_multitask
    "training.datamodule.config.tensorized_cohort_dir=${TENSORIZED_DIR}"
    training.datamodule.config.max_seq_len=256
    "training.datamodule.config.task_labels_dir=${LABELS_DIR}"
    "training.datamodule.mt_labels_dir=${LABELS_DIR}"
    "training.datamodule.num_tasks=${NUM_TASKS}"
    training.datamodule.batch_size=32
    training.datamodule.config.seq_sampling_strategy=to_end
    training.module.validation_auroc_log_per_task=true
    "query_projector.in_dim=${DIM}"
)

case "${ARM}" in
    patient_only)
        MODEL_ARGS+=(
            fusion=passthrough
            "head.in_dim=${DIM}"
            training/loss=multitask_binary_bce
        )
        ;;
    retrieval | retrieval_random_docs | retrieval_null_docs)
        # Control arms: identical model, but the documents carry no knowledge -- random
        # corpus documents (retriever.ablation_mode=random_docs) or the content-free corpus
        # from scripts/make_null_retrieval_db.py -- at train AND test time.
        ABLATION_MODE=none
        if [ "${ARM}" = retrieval_random_docs ]; then
            ABLATION_MODE=random_docs
        elif [ "${ARM}" = retrieval_null_docs ]; then
            RETRIEVAL_DB="${REPO_DIR}/data/retrieval_db_null"
        fi
        if [ ! -d "${RETRIEVAL_DB}" ]; then
            echo "ERROR: ${RETRIEVAL_DB} not found. See README.md for the symlink." >&2
            exit 1
        fi
        MODEL_ARGS+=(
            query_projector=sequence_mean_1024
            retriever=hf_dataset
            "retriever.dataset_path=${RETRIEVAL_DB}"
            retriever.doc_ids_column=null
            retriever.k=4
            "retriever.ablation_mode=${ABLATION_MODE}"
            retrieval_encoder=token_feature
            retrieval_encoder.vocab_size=151936
            retrieval_encoder.embedding_dim=64
            fusion=cross_attention_perdoc_medium
            marginalized_retrieval=true
            marginalized_output_mode=binary
            head.in_dim=256
            training/loss=multitask_binary_bce_marginalized
            "training.loss.num_tasks=${NUM_TASKS}"
        )
        ;;
    *)
        echo "ERROR: unknown ARM '${ARM}' (expected patient_only, retrieval, retrieval_random_docs or retrieval_null_docs)." >&2
        exit 1
        ;;
esac
