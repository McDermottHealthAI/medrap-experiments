#!/bin/bash
# ============================================================
# SLURM array: COLUMN 2 of the ablation ladder -- evaluate every trained
#              marginalized (binary) checkpoint on the held-out test split
#              with the REAL retrieved documents
#              (retriever.ablation_mode=none), model frozen, over 2 code
#              selections x 5 model seeds x 8 N values.
# ------------------------------------------------------------
# This column is the pivot of the whole table. It is the TREATMENT of
# comparison C1 (against eval_ladder_patient_only.sh: does retrieval help at
# all?) and simultaneously the BASELINE of C2 (against
# eval_ladder_random_docs.sh: is retrieval SELECTING useful documents?) and
# C3 (against eval_ladder_null_docs.sh: do document CONTENTS contribute?).
# Because it is one side of three comparisons, every other script in the
# ladder is this harness with exactly one thing changed.
#
# Byte-for-byte the same harness as eval_ladder_random_docs.sh and
# eval_ladder_null_docs.sh apart from retriever.ablation_mode /
# retriever.dataset_path, the missing rep dimension, output_dir and
# wandb_run_name. That is deliberate: the ablation's effect is
# (this script's test/auroc) - (the ablated test/auroc), and that subtraction
# is only meaningful if literally nothing else differs.
#
# Do NOT substitute the training-time val/auroc from
# sweep_marginalized_binary_ladder.sh for this column. That number comes from
# a different split: tuning, not held_out. (Since that sweep sets
# training.trainer.limit_val_batches=1.0 it is at least the FULL tuning split
# rather than the old 200-batch/6,400-row prefix -- but full-tuning is still
# the wrong split.) held_out is what this ladder is measured on, for all four
# columns; mixing splits across columns would confound the ablation with a
# change of split and the deltas would be uninterpretable.
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
# No rep dimension here: with ablation_mode=none and the real corpus there is
# no random draw at inference, so this is deterministic and one run per
# (selection, seed, N) cell is exact. The random-docs column needs its own rep
# dimension because its draw is unseeded (medrap-eval never calls
# seed_everything and _eval.yaml declares no `seed` field); compare this
# single number against the mean +/- sd of those reps. The `seed` here is the
# TRAINING seed baked into the checkpoint, a different noise source entirely,
# and the aggregator must keep both -- see the --rep-component note below.
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
# cosine (rationale: NULL_RESULT_DIAGNOSIS.md -- the raw dot product left
# ||q|| acting as an implicit inverse temperature, saturating the softmax over
# the K retrieved documents so effective_k_mean sat at 1.00-1.15 and the
# marginalization was arithmetically a no-op). The flag is a plain Python
# attribute, not a state_dict entry (model.py:145-147), so omitting it would
# load cleanly and then silently score documents by a different rule than
# training used -- exactly the class of bug this ladder exists to detect, and
# it must not be introduced by the ladder itself. Results from this script are
# NOT comparable to pre-change (dot-trained) numbers.
#
# retriever.k=128 is the trained k. Holding K fixed across all three
# marginalized columns is what keeps C2 and C3 measuring document RELEVANCE
# and document CONTENT rather than document COUNT.
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
# 80 array tasks = 2 code selections x 5 model seeds x 8 N values, flattened
# as IDX = sel_idx * 40 + seed_idx * 8 + n_idx. The array RANGE selects the
# code selection: --array=0-39 is `random`, --array=40-79 is `frequent`.
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
# Prerequisites:
#   sbatch scripts/prepare_retrieval.sh                   (builds data/retrieval_db)
#   sbatch scripts/generate_labels_random.sh              (or generate_labels_frequent.sh)
#   sbatch scripts/sweep_marginalized_binary_ladder.sh    (checkpoints must exist)
#
# Usage:
#   cd mimic_iv_ablation_ladder
#   sbatch --array=0-39%12 scripts/eval_ladder_real_docs.sh    # random selection
#   sbatch --array=40-79%12 scripts/eval_ladder_real_docs.sh   # frequent selection
#   sbatch --array=0 scripts/eval_ladder_real_docs.sh          # random, seed 1001, N=1
#
# Aggregating: this column is the C1 treatment and the C2/C3 baseline.
# --split test is REQUIRED -- eval_mode=test runs log test/auroc/* and no
# val/* column at all, so the aggregator's default --split val would find
# nothing. The rep component regex must collapse BOTH the seed and rep
# directory levels ('rep\d+' alone would leave seed1001 in the cell key), and
# unmatched cells are silently ignored:
#
#   python scripts/aggregate_ladder.py --split test --rep-component '(seed|rep)\d+'
# ============================================================

#SBATCH --job-name=eval-ladder-real-docs-k128-k128
#SBATCH --array=0-79
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
TRAIN_OUTPUT_DIR="${REPO_DIR}/outputs/marginalized_binary_ladder_k128/${SELECTION}/seed${MODEL_SEED}/n${N}"
CHECKPOINT_PATH="${TRAIN_OUTPUT_DIR}/checkpoints/last.ckpt"
EVAL_OUTPUT_DIR="${REPO_DIR}/outputs/eval_real_docs_k128/${SELECTION}/seed${MODEL_SEED}/n${N}"

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
echo "  Ladder column    : 2/4 -- marginalized, real docs (C1 treatment; C2/C3 baseline)"
echo "  Code selection   : ${SELECTION}"
echo "  Model seed       : ${MODEL_SEED} (replicate $((SEED_IDX + 1))/5)"
echo "  N (tasks)        : ${N}"
echo "  Retrieval corpus : ${RETRIEVAL_DB} (real MedRAG/textbooks passages)"
echo "  ablation_mode    : none (control -- real retrieved docs, frozen model)"
echo "  retriever.k      : 4 (same k the checkpoint was trained with)"
echo "  score similarity : cosine (MUST match the checkpoint)"
echo "  eval_mode        : test (MEDS held_out split, no batch limit)"
echo "  Checkpoint       : ${CHECKPOINT_PATH}"
echo "  W&B run name     : eval-real-docs-${SELECTION}-seed${MODEL_SEED}-n${N}-${SLURM_ARRAY_JOB_ID:-local}_${IDX}"
echo "  Started          : $(date)"
echo ""

# Same architecture as sweep_marginalized_binary_ladder.sh (must match the
# checkpoint's trained architecture for state_dict loading to succeed), and
# identical to eval_ladder_random_docs.sh / eval_ladder_null_docs.sh except
# retriever.ablation_mode=none, retriever.dataset_path, output_dir and
# wandb_run_name.
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
    retriever.k=128 \
    retriever.ablation_mode=none \
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
    "wandb_run_name=eval-real-docs-${SELECTION}-seed${MODEL_SEED}-n${N}-${SLURM_ARRAY_JOB_ID:-local}_${IDX}" \
    "$@"

echo ""
echo "=== Done: $(date) ==="
