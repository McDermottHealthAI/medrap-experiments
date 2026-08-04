#!/bin/bash
# ============================================================
# SLURM array: cross-attention marginalized retrieval with
#              marginalized_output_mode=binary, 2 code selections x 5 model
#              seeds x N in {1, 2, 4, 8, 16, 32, 64, 128} -- column 2 of the
#              ablation ladder, and the shared baseline for columns 3 and 4.
# ------------------------------------------------------------
# Every Hydra override is copied verbatim from
# ../mimic_iv_sweep_frequent/scripts/sweep_marginalized_binary_seeds_n1248.sh
# -- including the full-tuning-split validation protocol
# (training.trainer.limit_val_batches=1.0, training.trainer.val_check_interval=0.5)
# and marginalized_score_similarity=cosine -- with exactly two changes: this
# array is 3-D (code selection is now an array dimension rather than a
# property of the directory), and training.datamodule.num_workers=8 is added.
#
# The checkpoints this script writes are scored FOUR ways: once here at
# training time, then three times by medrap-eval on held_out --
# eval_ladder_real_docs.sh (column 2), eval_ladder_random_docs.sh
# (column 3, ablation_mode=random_docs, 3 reps) and eval_ladder_null_docs.sh
# (column 4, the content-free corpus from build_null_retrieval_db.py). C2 and
# C3 are within-checkpoint comparisons: the weights are frozen and only the
# documents change, so they are far more tightly paired than C1.
#
# marginalized_score_similarity=cosine is load-bearing and easy to lose. It
# is a plain Python attribute on the model (model.py:145-147), NOT part of
# state_dict, so an eval run that omits it loads these weights cleanly and
# then scores documents under a different rule than they were trained with,
# silently. It must appear in this script and in all three marginalized eval
# scripts. Its purpose: the raw dot product left ||q|| acting as an implicit
# inverse temperature, so the softmax over the K retrieved documents
# saturated and effective_k_mean sat at 1.00-1.15 across all 32 runs of the
# previous sweep -- making the marginalization arithmetically a no-op and
# driving the gradient into the retrieval scores to exactly zero. cosine
# bounds the scores to [-1, 1] and removes that degree of freedom. Watch the
# first job's metrics.csv: if effective_k_mean is still ~1.0 the retrieval
# arm is collapsed and columns 2-4 are meaningless.
#
# num_workers=8 is a pure throughput change and cannot perturb any number.
# training.datamodule.config.seq_sampling_strategy=to_end resolves to
# `return seq_len - max_seq_len` in meds_torchdata/types.py -- there is no RNG
# anywhere in the subsequence path -- the label join is keyed on
# (subject_id, prediction_time), and lightning.seed_everything(cfg.seed,
# workers=True) seeds every DataLoader worker regardless. For this arm the
# retrieved documents are a deterministic function of the query encoder, so
# retrieval order is not a separate noise source either; it is downstream of
# the same seed.
#
# 80 array tasks = 2 selections x 5 seeds x 8 N values, flattened as
# IDX = sel_idx * 40 + seed_idx * 8 + n_idx
# (sel_idx in [0,1], seed_idx in [0,4], n_idx in [0,7]).
#
# Array index -> (code selection, model seed, N):
#   0-7   -> random,   seed 1001, N = 1, 2, 4, 8, 16, 32, 64, 128
#   8-15  -> random,   seed 2002, N = 1, 2, 4, 8, 16, 32, 64, 128
#   16-23 -> random,   seed 3003, N = 1, 2, 4, 8, 16, 32, 64, 128
#   24-31 -> random,   seed 4004, N = 1, 2, 4, 8, 16, 32, 64, 128
#   32-39 -> random,   seed 5005, N = 1, 2, 4, 8, 16, 32, 64, 128
#   40-47 -> frequent, seed 1001, N = 1, 2, 4, 8, 16, 32, 64, 128
#   48-55 -> frequent, seed 2002, N = 1, 2, 4, 8, 16, 32, 64, 128
#   56-63 -> frequent, seed 3003, N = 1, 2, 4, 8, 16, 32, 64, 128
#   64-71 -> frequent, seed 4004, N = 1, 2, 4, 8, 16, 32, 64, 128
#   72-79 -> frequent, seed 5005, N = 1, 2, 4, 8, 16, 32, 64, 128
#
# The array range IS the code selection: --array=0-39 runs `random` only and
# --array=40-79 runs `frequent` only, which is how run_all.sh --selection
# picks between them. Running one selection at a time keeps all 40 paired
# cells, all 5 seeds and all 8 N values -- it only defers the second table.
#
# MODEL_SEEDS=(1001 2002 3003 4004 5005) are deliberately distinct from the
# default 42 and from the label-draw seeds 101/202/303/404/505, matching this
# repo's rationale of keeping seed families disjoint.
#
# This is the more expensive of the two training arms
# (PerDocCrossAttentionFusion expands the batch to B*K internally at k=4), so
# a small-N-first submission is the cheap way to sanity-check the setup.
#
# Prerequisites:
#   sbatch scripts/prepare_retrieval.sh
#   sbatch scripts/generate_labels_random.sh
#   sbatch scripts/generate_labels_frequent.sh
#
# Usage:
#   cd mimic_iv_ablation_ladder
#   sbatch --array=0-39%12 scripts/sweep_marginalized_binary_ladder.sh   # random
#   sbatch --array=40-79%12 scripts/sweep_marginalized_binary_ladder.sh  # frequent
#   sbatch --array=0 scripts/sweep_marginalized_binary_ladder.sh         # smoke test
# ============================================================

#SBATCH --job-name=sweep-marg-binary-ladder-k128
#SBATCH --array=0-79
#SBATCH --partition=gpu
#SBATCH --account=mm6677_gp
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=16:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

IDX=${SLURM_ARRAY_TASK_ID}

SELECTIONS=(random frequent)
MODEL_SEEDS=(1001 2002 3003 4004 5005)
NS=(1 2 4 8 16 32 64 128)

SEL_IDX=$((IDX / 40))
REM=$((IDX % 40))
SEED_IDX=$((REM / 8))
N_IDX=$((REM % 8))

SELECTION=${SELECTIONS[$SEL_IDX]}
MODEL_SEED=${MODEL_SEEDS[$SEED_IDX]}
N=${NS[$N_IDX]}

REPO_DIR="${SLURM_SUBMIT_DIR}"
VENV="${REPO_DIR}/.venv/bin/activate"  # created by: cd mimic_iv_ablation_ladder && uv sync

RETRIEVAL_DB="${REPO_DIR}/data/retrieval_db"
TENSORIZED_DIR="/groups/mm6677_gp/data/MIMIC_MEDS/MEDS_cohort/processed"
LABELS_DIR="${REPO_DIR}/data/tasks_${SELECTION}/n${N}/tasks"
OUTPUT_DIR="${REPO_DIR}/outputs/marginalized_binary_ladder_k128/${SELECTION}/seed${MODEL_SEED}/n${N}"

cd "${REPO_DIR}"
# shellcheck source=/dev/null
source "${VENV}"
mkdir -p logs "${OUTPUT_DIR}"

if [ ! -d "${RETRIEVAL_DB}" ]; then
    echo "ERROR: ${RETRIEVAL_DB} not found. Run sbatch scripts/prepare_retrieval.sh first." >&2
    exit 1
fi

if [ ! -d "${LABELS_DIR}" ]; then
    echo "ERROR: ${LABELS_DIR} not found. Run sbatch scripts/generate_labels_${SELECTION}.sh first." >&2
    exit 1
fi

echo "=== Job info ==="
echo "  Array job                 : ${SLURM_ARRAY_JOB_ID:-local}[${IDX}]"
echo "  Node                      : ${SLURMD_NODENAME:-$(hostname)}"
echo "  GPU(s)                    : ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "  Code selection            : ${SELECTION}"
echo "  N (tasks)                 : ${N}"
echo "  Model seed                : ${MODEL_SEED} (replicate $((SEED_IDX + 1))/5; default would be 42)"
echo "  marginalized_output_mode  : binary (MedRAP#93)"
echo "  Score similarity          : cosine"
echo "  Started                   : $(date)"
echo ""

# Override set copied verbatim from
# ../mimic_iv_sweep_frequent/scripts/sweep_marginalized_binary_seeds_n1248.sh,
# plus training.datamodule.num_workers=8. Any eval run against these
# checkpoints must repeat marginalized_score_similarity=cosine -- see header.
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
    training.datamodule.num_workers=8 \
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
    retriever.k=128 \
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
    "seed=${MODEL_SEED}" \
    "output_dir=${OUTPUT_DIR}" \
    do_overwrite=true \
    "wandb_run_name=marginalized-binary-ladder-k128-${SELECTION}-seed${MODEL_SEED}-n${N}-${SLURM_ARRAY_JOB_ID:-local}_${IDX}" \
    "$@"

echo ""
echo "=== Done: $(date) ==="
