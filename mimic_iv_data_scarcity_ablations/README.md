# mimic_iv_data_scarcity_ablations

Frozen-checkpoint random-doc and null-doc ablations run against the **exact
original checkpoints** behind
[`results/data_scarcity_retrieval/`](../results/data_scarcity_retrieval/README.md)
(Hayk's working copy at `/groups/mm6677_gp/hs3627/medrap-experiments/mimic_iv_sweep/`,
referenced read-only), following the same recipe zzw2102 already ran for the
capacity-starved line
([`mimic_iv_null_random_doc_ablations/`](https://github.com/McDermottHealthAI/medrap-experiments/tree/main/mimic_iv_null_random_doc_ablations))
-- **that line is not re-run here**, only the data-scarcity checkpoints,
which had no ablation coverage yet.

The tensorized MIMIC cohort these checkpoints trained on was deleted from
`/groups/mm6677_gp/data/MIMIC_MEDS/MEDS_cohort/` (confirmed gone as of this
writing); evals use the same byte-for-byte snapshot-restored copy zzw2102's
line already validated: `/groups/mm6677_gp/zzw2102/data/MIMIC_MEDS_restored/processed`.

Every script in `scripts/` is a mechanical clone of the original
`mimic_iv_sweep/scripts/` data-scarcity sweep scripts with only the
tensorized-cohort path, checkpoint/label paths, ablation flags, and
job/wandb names changed. No new model code: `random_docs` is MedRAP's
built-in `retriever.ablation_mode`; the null corpus is the same pre-existing
content-free artifact zzw2102's line uses
(`data/retrieval_db_null` -> symlink to
`/groups/mm6677_gp/zzw2102/medrap-experiments/mimic_iv_ablation_ladder/data/retrieval_db_null`).

Scope: `learned-linear` query projector only (the only variant trained in
the data-scarcity line), all 4 fractions (50/20/10/5%). `qwen3_text` was
never trained under data scarcity, so there's nothing to ablate there.
`patient_only` has no retrieval, so only its test-split reproduction is
needed (as the floor these ablations compare against) -- no random/null
ablation applies to it.

## Run order

```bash
cd mimic_iv_data_scarcity_ablations
uv sync                                                              # once
ln -s /groups/mm6677_gp/hs3627/medrap-experiments/mimic_iv_sweep/data/retrieval_db data/retrieval_db
ln -s /groups/mm6677_gp/zzw2102/medrap-experiments/mimic_iv_ablation_ladder/data/retrieval_db_null data/retrieval_db_null

# reproduction gate (must match published test-direction numbers before ablations mean anything)
for f in 50 20 10 5; do
  sbatch scripts/eval_test_patient_only_data_scarcity_train${f}pct_repro.sh
  sbatch scripts/eval_test_marginalized_binary_learned_linear_data_scarcity_train${f}pct_repro.sh
done

# ablations (after the gate passes)
for f in 50 20 10 5; do
  sbatch scripts/eval_test_marginalized_binary_learned_linear_data_scarcity_train${f}pct_random_docs.sh
  sbatch scripts/eval_test_marginalized_binary_learned_linear_data_scarcity_train${f}pct_null_docs.sh
done
```

Results: [`results/random_doc_null_doc_ablations_data_scarcity/`](../results/random_doc_null_doc_ablations_data_scarcity/README.md)

## Cleanup

Once results are aggregated and merged into the results README, the large
generated artifacts in this directory (`outputs/`, `.venv/`, `data/` symlink
targets are read-only elsewhere so the symlinks themselves are tiny) are
deleted from the cluster -- the numbers live in the merged README and W&B,
not in local eval output copies. Only `scripts/`, `README.md`,
`pyproject.toml`, and `uv.lock` are meant to persist.
