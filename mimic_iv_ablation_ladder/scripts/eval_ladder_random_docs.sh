#!/bin/bash
# ============================================================
# SLURM array: COLUMN 3 of the ablation ladder -- inference-time
#              random-document ablation on the trained marginalized (binary)
#              checkpoints. Model frozen, the real retrieved documents
#              replaced by random ones at eval time only
#              (retriever.ablation_mode=random_docs), over 2 code selections
#              x 5 model seeds x 8 N values x 3 reps.
# ------------------------------------------------------------
# Comparison C2: is the retriever SELECTING useful documents, or is the
# marginalized arm just winning on the extra machinery? Nothing is retrained
# here -- this loads the checkpoint written by
# sweep_marginalized_binary_ladder.sh and swaps only what the retriever hands
# the fusion module. Every number here is read against the same
# (selection, seed, N) number from eval_ladder_real_docs.sh, which is this
# exact harness with ablation_mode=none.
#
# C2 is a WITHIN-CHECKPOINT comparison: the same weights score both columns,
# so it is far more tightly paired than the between-arm C1, and nearly free
# because it reuses checkpoints C1 already paid for.
#
# WHY THREE REPS, AND WHY THEY ARE NOT SEEDS. The random draw is UNSEEDED:
# medrap-eval never calls seed_everything and _eval.yaml declares no `seed`
# field, so every run picks a different set of random documents and no run is
# reproducible. Rather than report one arbitrary draw, this runs 3
# independent repetitions per cell; report mean +/- sd of test/auroc across
# them. `rep` (inference-time document draw) and `seed` (training noise,
# baked into the checkpoint) measure different things and BOTH must be kept
# in the aggregation -- do not collapse one into the other. This is the only
# ladder column with a rep dimension; the other three are deterministic given
# their checkpoint.
#
# Differs from eval_ladder_real_docs.sh in exactly three ways:
#   1. retriever.ablation_mode=random_docs (that script runs no ablation).
#   2. The rep dimension and the extra /rep${REP} output_dir level.
#   3. output_dir and wandb_run_name.
# retriever.k stays at 4 -- the k the checkpoints were TRAINED with. Fewer
# documents would be a different ablation, and mixing it in would confound
# document count with document relevance, which is the only thing this
# column is trying to measure.
#
# random_docs is the WEAKER of the two content ablations, by design.
# ablation_mode=random_docs still returns genuine textbook prose (just the
# wrong passages), so the fusion stage still sees plausible medical language
# and the per-document scores still vary. eval_ladder_null_docs.sh is the
# strictly stronger version. Reading C2 and C3 together separates "the
# retriever picks the right passages" from "any medical prose at all would
# do".
#
# eval_mode=test runs the FULL MEDS held_out split (meds_torchdata's
# Lightning datamodule maps "held_out" -> Lightning's "test", "tuning" ->
# "val"). Do NOT add limit_val_batches: nothing limits test batches, and the
# flag would be a no-op that implies a limit exists. eval_mode=validate would
# instead silently score only the 6,400-row tuning prefix that
# training/trainer=lightning_wandb's limit_val_batches=200 imposes.
#
# training/trainer=lightning_wandb is REQUIRED, not cosmetic. The _eval.yaml
# default trainer is lightning_eval, which sets logger: false -- no
# metrics.csv is written, so aggregate_ladder.py sees no run at all and this
# column comes out silently empty.
#
# Every OTHER Hydra override is copied VERBATIM from
# sweep_marginalized_binary_ladder.sh. If the architecture args drift, either
# load_state_dict fails outright or -- worse -- the weights load into a
# silently different model. That includes
# training/loss=multitask_binary_bce_marginalized and
# training.loss.num_tasks=<N>; dropping the loss config was a real bug once
# (commit 88ad907). Only the training-only knobs are dropped: max_epochs,
# gradient_clip_val, log_every_n_steps, module.lr, module.warmup_steps,
# limit_val_batches, val_check_interval, datamodule.num_workers and seed.
# (seed is not merely unnecessary -- _eval.yaml declares no `seed` field, so
# passing it is a hard Hydra struct error.)
#
# marginalized_score_similarity=cosine IS set, and must be: it is a
# scoring-function override rather than a weight shape, so it has to MATCH
# the checkpoint being loaded. sweep_marginalized_binary_ladder.sh trains with
# cosine (rationale: NULL_RESULT_DIAGNOSIS.md). The flag is a plain Python
# attribute, not a state_dict entry (model.py:145-147), so omitting it would
# load cleanly and then silently score documents by a different rule than
# training used -- exactly the class of bug this ablation exists to detect,
# and it must not be introduced by the ablation itself. Results from this
# script are NOT comparable to pre-change (dot-trained) numbers.
#
# Requires medrap built from the commit pinned in pyproject.toml
# (feat/anchor-strategy-uniform-event @ 878c038), which carries
# McDermottHealthAI/MedRAP#94 (test/auroc/* logging under trainer.test()),
# #95 (the eval-config key declarations Hydra struct mode needs --
# marginalized_retrieval, marginalized_score_similarity,
# marginalized_output_mode, wandb_project, wandb_run_name), #97 (do_overwrite
# on medrap-eval), and a retriever config that declares ablation_mode --
# otherwise Hydra rejects the override outright.
#
# 240 array tasks = 2 code selections x 5 model seeds x 8 N values x 3 reps,
# flattened as IDX = sel_idx * 120 + seed_idx * 24 + n_idx * 3 + rep_idx.
# The array RANGE selects the code selection: --array=0-119 is `random`,
# --array=120-239 is `frequent`.
#
# Array index -> (code selection, model seed, N, rep):
#   0-2     -> random,   seed 1001, N=1,   reps 1-3
#   3-5     -> random,   seed 1001, N=2,   reps 1-3
#   6-8     -> random,   seed 1001, N=4,   reps 1-3
#   9-11    -> random,   seed 1001, N=8,   reps 1-3
#   12-14   -> random,   seed 1001, N=16,  reps 1-3
#   15-17   -> random,   seed 1001, N=32,  reps 1-3
#   18-20   -> random,   seed 1001, N=64,  reps 1-3
#   21-23   -> random,   seed 1001, N=128, reps 1-3
#   24-47   -> random,   seed 2002, N = 1..128, reps 1-3 (same layout)
#   48-71   -> random,   seed 3003, N = 1..128, reps 1-3
#   72-95   -> random,   seed 4004, N = 1..128, reps 1-3
#   96-119  -> random,   seed 5005, N = 1..128, reps 1-3
#   120-143 -> frequent, seed 1001, N = 1..128, reps 1-3
#   144-167 -> frequent, seed 2002, N = 1..128, reps 1-3
#   168-191 -> frequent, seed 3003, N = 1..128, reps 1-3
#   192-215 -> frequent, seed 4004, N = 1..128, reps 1-3
#   216-239 -> frequent, seed 5005, N = 1..128, reps 1-3
#
# Prerequisites:
#   sbatch scripts/prepare_retrieval.sh                   (builds data/retrieval_db)
#   sbatch scripts/generate_labels_random.sh              (or generate_labels_frequent.sh)
#   sbatch scripts/sweep_marginalized_binary_ladder.sh    (checkpoints must exist)
#   sbatch scripts/eval_ladder_real_docs.sh               (the paired control)
#
# Usage:
#   cd mimic_iv_ablation_ladder
#   sbatch --array=0-119%12 scripts/eval_ladder_random_docs.sh     # random selection
#   sbatch --array=120-239%12 scripts/eval_ladder_random_docs.sh   # frequent selection
#   sbatch --array=0-2 scripts/eval_ladder_random_docs.sh          # random, seed 1001, N=1, all 3 reps
#
# Aggregating: this column is the C2 treatment against the real-docs
# baseline. --split test is REQUIRED -- eval_mode=test runs log test/auroc/*
# and no val/* column at all, so the aggregator's default --split val would
# find nothing. The rep component regex must collapse BOTH the seed and rep
# directory levels: it is re.fullmatch'ed per path component, so 'rep\d+'
# alone would leave seed1001 in the cell key, this column sits one level
# deeper than the other three, and unmatched cells are silently ignored --
# getting it wrong yields a quietly empty table rather than an error:
#
#   python scripts/aggregate_ladder.py --split test --rep-component '(seed|rep)\d+'
# ============================================================

#SBATCH --job-name=eval-ladder-random-docs
#SBATCH --array=0-239
#SBATCH --partition=gpu
#SBATCH --account=mm6677_gp
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

IDX=${SLURM_ARRAY_TASK_ID}

SELECTIONS=(random frequent)
MODEL_SEEDS=(1001 2002 3003 4004 5005)
NS=(1 2 4 8 16 32 64 128)

SEL_IDX=$((IDX / 120))
REM=$((IDX % 120))
SEED_IDX=$((REM / 24))
REM2=$((REM % 24))
N_IDX=$((REM2 / 3))
REP_IDX=$((REM2 % 3))

SELECTION=${SELECTIONS[$SEL_IDX]}
MODEL_SEED=${MODEL_SEEDS[$SEED_IDX]}
N=${NS[$N_IDX]}
REP=$((REP_IDX + 1))

REPO_DIR="${SLURM_SUBMIT_DIR}"
VENV="${REPO_DIR}/.venv/bin/activate"  # created by: cd mimic_iv_ablation_ladder && uv sync

RETRIEVAL_DB="${REPO_DIR}/data/retrieval_db"
TENSORIZED_DIR="/groups/mm6677_gp/data/MIMIC_MEDS/MEDS_cohort/processed"
LABELS_DIR="${REPO_DIR}/data/tasks_${SELECTION}/n${N}/tasks"
TRAIN_OUTPUT_DIR="${REPO_DIR}/outputs/marginalized_binary_ladder/${SELECTION}/seed${MODEL_SEED}/n${N}"
CHECKPOINT_PATH="${TRAIN_OUTPUT_DIR}/checkpoints/last.ckpt"
EVAL_OUTPUT_DIR="${REPO_DIR}/outputs/eval_random_docs/${SELECTION}/seed${MODEL_SEED}/n${N}/rep${REP}"

cd "${REPO_DIR}"
# shellcheck source=/dev/null
source "${VENV}"
mkdir -p logs "${EVAL_OUTPUT_DIR}"

if [ ! -d "${RETRIEVAL_DB}" ]; then
    echo "ERROR: ${RETRIEVAL_DB} not found. Run sbatch scripts/prepare_retrieval.sh first." >&2
    exit 1
fi

if [ ! -d "${LABELS_DIR}" ]; then
    echo "ERROR: ${LABELS_DIR} not found. Run sbatch scripts/generate_labels_${SELECTION}.sh first." >&2
    exit 1
fi

if [ ! -f "${CHECKPOINT_PATH}" ]; then
    echo "ERROR: ${CHECKPOINT_PATH} not found. Run sbatch scripts/sweep_marginalized_binary_ladder.sh first." >&2
    exit 1
fi

echo "=== Job info ==="
echo "  Array job        : ${SLURM_ARRAY_JOB_ID:-local}[${IDX}]"
echo "  Node             : ${SLURMD_NODENAME:-$(hostname)}"
echo "  GPU(s)           : ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "  Ladder column    : 3/4 -- frozen random docs (C2 treatment)"
echo "  Code selection   : ${SELECTION}"
echo "  Model seed       : ${MODEL_SEED} (replicate $((SEED_IDX + 1))/5)"
echo "  N (tasks)        : ${N}"
echo "  Rep              : ${REP}/3 (UNSEEDED draw -- aggregate mean +/- sd)"
echo "  Retrieval corpus : ${RETRIEVAL_DB} (real corpus; docs drawn at random)"
echo "  ablation_mode    : random_docs (frozen model, random docs at inference)"
echo "  retriever.k      : 4 (same k the checkpoint was trained with)"
echo "  score similarity : cosine (MUST match the checkpoint)"
echo "  eval_mode        : test (MEDS held_out split, no batch limit)"
echo "  Checkpoint       : ${CHECKPOINT_PATH}"
echo "  W&B run name     : eval-random-docs-${SELECTION}-seed${MODEL_SEED}-n${N}-rep${REP}-${SLURM_ARRAY_JOB_ID:-local}_${IDX}"
echo "  Started          : $(date)"
echo ""

# Same architecture as sweep_marginalized_binary_ladder.sh (must match the
# checkpoint's trained architecture for state_dict loading to succeed), and
# identical to eval_ladder_real_docs.sh except
# retriever.ablation_mode=random_docs, output_dir and wandb_run_name.
medrap-eval \
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
    training/trainer=lightning_wandb \
    training.trainer.accelerator=gpu \
    training.trainer.devices=1 \
    training.module.validation_auroc_log_per_task=true \
    query_projector=sequence_mean_1024 \
    query_projector.in_dim=128 \
    retriever=hf_dataset \
    "retriever.dataset_path=${RETRIEVAL_DB}" \
    retriever.doc_ids_column=null \
    retriever.k=4 \
    retriever.ablation_mode=random_docs \
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
    "checkpoint_path=${CHECKPOINT_PATH}" \
    eval_mode=test \
    "output_dir=${EVAL_OUTPUT_DIR}" \
    do_overwrite=true \
    "wandb_run_name=eval-random-docs-${SELECTION}-seed${MODEL_SEED}-n${N}-rep${REP}-${SLURM_ARRAY_JOB_ID:-local}_${IDX}" \
    "$@"

echo ""
echo "=== Done: $(date) ==="
