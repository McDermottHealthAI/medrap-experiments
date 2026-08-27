# mimic_iv_null_random_doc_ablations

Frozen-checkpoint random-doc and null-doc ablations run against the **exact original
checkpoints** behind [`results/capacity_starved_retrieval/`](../results/capacity_starved_retrieval/README.md)
(Hayk's working copy at `/groups/mm6677_gp/hs3627/medrap-experiments/mimic_iv_sweep/`,
referenced read-only), plus a full reproduction of that experiment's published val and
test tables as the setup-validation gate.

The tensorized MIMIC cohort those checkpoints trained on had been deleted from
`/groups/mm6677_gp/data/MIMIC_MEDS/MEDS_cohort/`; it was recovered byte-for-byte from
the filesystem snapshot `/groups/.snapshots/@GMT-2026.08.25-12.06.07/` into
`/groups/mm6677_gp/zzw2102/data/MIMIC_MEDS_restored/processed`.

Every script in `scripts/` is a mechanical clone of the original
`mimic_iv_sweep/scripts/` eval scripts with only the tensorized-cohort path
(→ the snapshot-restored copy), checkpoint/label paths (→ hs3627's originals),
ablation flags, and job/wandb names changed. No new model code: `random_docs` is MedRAP's built-in
`retriever.ablation_mode`; the null corpus is the pre-existing content-free artifact
(`data/retrieval_db_null` → symlink to `mimic_iv_ablation_ladder/data/retrieval_db_null`).

## Run order

```bash
cd mimic_iv_null_random_doc_ablations
uv sync                                                              # once
# reproduction gate (must match published tables before ablations mean anything)
sbatch scripts/eval_test_patient_only_capacity_starved_n25_30d_repro.sh
sbatch scripts/eval_test_marginalized_binary_learned_linear_capacity_starved_n25_30d_repro.sh
sbatch scripts/eval_test_marginalized_binary_qwen3_text_capacity_starved_n25_30d_repro.sh
sbatch scripts/eval_val_*_repro200.sh                                # val, 200-batch protocol
# ablations
sbatch scripts/eval_test_marginalized_binary_learned_linear_capacity_starved_n25_30d_random_docs.sh
sbatch scripts/eval_test_marginalized_binary_learned_linear_capacity_starved_n25_30d_null_docs.sh
sbatch scripts/eval_test_marginalized_binary_qwen3_text_capacity_starved_n25_30d_random_docs.sh
sbatch scripts/eval_test_marginalized_binary_qwen3_text_capacity_starved_n25_30d_null_docs.sh
```

(`eval_val_*_repro.sh` without the `200` suffix evaluate the FULL tuning split — a
bonus metric; the published val table used the 200-batch prefix protocol.)

Results: [`results/random_doc_null_doc_ablations_mimic_iv/`](../results/random_doc_null_doc_ablations_mimic_iv/README.md)
