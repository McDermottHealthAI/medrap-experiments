#!/bin/bash
# ============================================================
# SLURM array: evaluate the trained marginalized (binary) checkpoints against
#              the held-out test split with retrieval content neutralized --
#              the frozen model is pointed at a hand-built content-free
#              ("null") corpus instead of the MedRAG/textbooks corpus it was
#              trained on, at the same retriever.k=4 used at training.
# ------------------------------------------------------------
# Same checkpoints as eval_marginalized_binary_top1_n1248.sh
# (outputs/marginalized_binary_n1248/n<N>/checkpoints/last.ckpt, written by
# sweep_marginalized_binary_n1248.sh) and the same medrap-eval eval_mode=test
# invocation, but that script varies K (k=1 vs. the trained k=4) while this
# one varies document *content* and holds K fixed. Nothing is retrained; the
# weights are frozen and only the corpus underneath them changes, so the gap
# between this script's test/auroc/* and the real-corpus test eval's is the
# part of the score that the retrieved textbook passages were actually
# carrying, as opposed to the patient encoder alone.
#
# The corpus IS the ablation, which is why retriever.ablation_mode stays
# none: every document in data/retrieval_db_null is a single placeholder
# token followed by padding, with an all-zeros key embedding, and every row
# is identical. FAISS search, payload materialization and the marginalization
# all run exactly as in a normal eval -- there is simply nothing in the
# documents to read. That is strictly stronger than ablation_mode=random_docs,
# which still returns genuine textbook prose (just the wrong passages), so the
# fusion stage still sees plausible medical language and the per-document
# scores still vary. Here the differentiable document scores are exactly tied,
# so marginalizing over K=4 degenerates to averaging K identical per-document
# predictions.
#
# retriever.k=4 is the trained k on purpose: holding K at its training value
# keeps the fused tensor shapes and the marginalization arity byte-identical
# to training, leaving document content as the only changed variable.
#
# eval_mode=test (the MEDS held_out split), not validate: medrap-eval
# eval_mode=validate runs Lightning's validation loop, which inherits
# limit_val_batches=200 from training/trainer=lightning_wandb and would score
# only the first 200 tuning batches. eval_mode=test runs the full held_out
# split and is directly comparable to eval_marginalized_binary_top1_n1248.sh.
#
# marginalized_score_similarity=cosine IS set, and must be: the flag lives
# in the config rather than the state_dict, so it has to MATCH the checkpoint
# being loaded. sweep_marginalized_binary_n1248.sh now trains with cosine
# (rationale: NULL_RESULT_DIAGNOSIS.md), so omitting it here would load
# cleanly and then silently score documents under a different similarity
# than training, with no error to warn you. Run this only against
# checkpoints produced by the post-change sweep. Every other override below
# is copied verbatim from sweep_marginalized_binary_n1248.sh for the same
# reason, including training/loss=multitask_binary_bce_marginalized and
# training.loss.num_tasks=${N}.
#
# cosine is safe on this corpus: with the null documents' all-zeros key
# embeddings, torch.nn.functional.normalize's default eps=1e-12 clamp maps
# them to zero rather than NaN (model/retrieval_scoring.py:76-79), so the
# per-document scores stay exactly tied under cosine just as they were under
# dot, and the marginalization over K=4 still degenerates to averaging K
# identical per-document predictions -- which is the property this ablation
# depends on.
#
# Note on append_null_doc: MedRAP branches origin/feat/win-rate-metric and
# origin/random-prediction-time carry a retriever flag by that name, which is
# a DIFFERENT mechanism -- it appends the corpus's last row as a K+1'th
# "abstain" option with score 0, rather than replacing the corpus. It is also
# broken for real corpora: it hardcodes row len(dataset)-1, which in
# ../mimic_iv_sweep/data/retrieval_db is Surgery_Schwartz_14348, an ordinary
# textbook passage and not a null document at all. So it is deliberately not
# used here; this script gets its null documents from the corpus instead.
#
# Requires the same MedRAP support as eval_marginalized_binary_top1_n1248.sh:
# McDermottHealthAI/MedRAP#94 (test/auroc/* logging for trainer.test()), #95
# (marginalized_retrieval/marginalized_score_similarity/
# marginalized_output_mode/wandb_project/wandb_run_name declared on
# _eval.yaml, without which Hydra struct mode rejects these overrides), and
# #97 (do_overwrite on medrap-eval, so this can be re-run without deleting
# the output dir by hand).
#
# Array index -> N:
#   0 -> N=1
#   1 -> N=2
#   2 -> N=4
#   3 -> N=8
#   4 -> N=16
#   5 -> N=32
#   6 -> N=64
#   7 -> N=128
#
# Prerequisites:
#   sweep_marginalized_binary_n1248.sh must have already finished for the N
#   values you want to evaluate (checkpoints must exist).
#   python scripts/build_null_retrieval_db.py --output-dir data/retrieval_db_null
#     (plain CPU python, runs in seconds -- no GPU and no Qwen embedder needed,
#      because the retriever never re-embeds at inference)
#
# Usage:
#   cd mimic_iv_sweep_frequent
#   sbatch scripts/eval_frozen_null_docs_n1248.sh
#   sbatch --array=0 scripts/eval_frozen_null_docs_n1248.sh   # N=1 only
#
# Aggregating (paired against the real-docs control). --split test is
# REQUIRED: eval_mode=test runs log test/auroc/* and no val/* column at all,
# so the aggregator's default --split val would find nothing.
#
#   python scripts/aggregate_results.py --split test \
#       --baseline  'outputs/frozen_real_docs_eval_n1248/n*' \
#       --baseline-label  'real docs' \
#       --treatment 'outputs/frozen_null_docs_eval_n1248/n*' \
#       --treatment-label 'null docs'
# ============================================================

#SBATCH --job-name=eval-frozen-null-docs-n1248-frequent
#SBATCH --array=0-7
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

NS=(1 2 4 8 16 32 64 128)
N=${NS[$IDX]}

REPO_DIR="${SLURM_SUBMIT_DIR}"
VENV="${REPO_DIR}/.venv/bin/activate"  # created by: cd mimic_iv_sweep_frequent && uv sync

# Local null corpus -- NOT the shared ../mimic_iv_sweep/data/retrieval_db.
RETRIEVAL_DB="${REPO_DIR}/data/retrieval_db_null"
TENSORIZED_DIR="/groups/mm6677_gp/data/MIMIC_MEDS/MEDS_cohort/processed"
LABELS_DIR="${REPO_DIR}/data/tasks/n${N}/tasks"
TRAIN_OUTPUT_DIR="${REPO_DIR}/outputs/marginalized_binary_n1248/n${N}"
CHECKPOINT_PATH="${TRAIN_OUTPUT_DIR}/checkpoints/last.ckpt"
EVAL_OUTPUT_DIR="${REPO_DIR}/outputs/frozen_null_docs_eval_n1248/n${N}"

cd "${REPO_DIR}"
# shellcheck source=/dev/null
source "${VENV}"
mkdir -p logs "${EVAL_OUTPUT_DIR}"

if [ ! -d "${RETRIEVAL_DB}" ]; then
    echo "ERROR: ${RETRIEVAL_DB} not found. Run python scripts/build_null_retrieval_db.py --output-dir data/retrieval_db_null first." >&2
    exit 1
fi

if [ ! -f "${CHECKPOINT_PATH}" ]; then
    echo "ERROR: ${CHECKPOINT_PATH} not found. Run sbatch scripts/sweep_marginalized_binary_n1248.sh for N=${N} first." >&2
    exit 1
fi

echo "=== Job info ==="
echo "  Array job         : ${SLURM_ARRAY_JOB_ID:-local}[${IDX}]"
echo "  Node              : ${SLURMD_NODENAME:-$(hostname)}"
echo "  GPU(s)            : ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "  N (tasks)         : ${N}"
echo "  Retrieval corpus  : ${RETRIEVAL_DB} (null docs -- content-free)"
echo "  retriever.k       : 4 (same as training; only doc content changes)"
echo "  ablation_mode     : none (the corpus is the ablation)"
echo "  eval_mode         : test (MEDS held_out split)"
echo "  Checkpoint        : ${CHECKPOINT_PATH}"
echo "  W&B run name      : frozen-null-docs-frequent-n${N}-${SLURM_ARRAY_JOB_ID:-local}_${IDX}"
echo "  Started           : $(date)"
echo ""

# Same architecture as sweep_marginalized_binary_n1248.sh (must match the
# checkpoint's trained architecture for state_dict loading to succeed),
# except retriever.dataset_path (the null corpus), eval_mode=test, and
# checkpoint_path/output_dir.
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
    "wandb_run_name=frozen-null-docs-frequent-n${N}-${SLURM_ARRAY_JOB_ID:-local}_${IDX}" \
    "$@"

echo ""
echo "=== Done: $(date) ==="
