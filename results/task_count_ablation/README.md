# Task-count ablation (N=1..128 task codes), extreme capacity

Status: complete. 8 task-count levels x 2 architectures, single job each
(no draw array -- see methodology caveat below).

Scripts: `mimic_iv_sweep/scripts/sweep_{patient_only,marginalized_binary_learned_linear}_task_count_ablation_n{1,2,4,8,16,32,64,128}.sh`
and matching `eval_test_*` scripts, all in `mimic_iv_sweep/scripts/`.

## Methodology caveat (read before interpreting results)

This experiment is **not** apples-to-apples with every other result in this
repo, for reasons that could not be resolved without a risky, unverified
cluster operation:

1. **Old label-generation methodology.** These labels reuse the pre-existing
   set at `data/tasks/n{N}/tasks/` from an earlier line of this project:
   `horizon_days=7.0` (not the 30-day window used everywhere else),
   generated before the anchor-sampling fix (not `anchor_strategy=uniform_event`,
   MedRAP#100).
2. **Single draw, not 5.** These label sets were only ever generated once
   per N (no repeated random task-code draws), so there is no draw-to-draw
   variance estimate here, unlike every other experiment in this repo.
3. **`intermediate/` is unrecoverable.** Generating fresh labels at the
   current methodology requires `medrap-preprocess` reading from
   `MEDS_cohort/intermediate/` (the filtered-population MEDS data). This
   directory was deleted alongside the original tensorized cohort
   (~2026-08-25) and, unlike `MEDS_cohort/processed/` (restored from a
   snapshot for the other experiments in this repo), no available snapshot
   -- checked back to the earliest retained one -- contains `intermediate/`.
   Regenerating it from raw `MEDS_cohort/data/` requires filtering
   parameters (`min_subjects_per_code`, `min_events_per_subject`) that
   were not confirmed, so it was not attempted -- doing so with guessed
   parameters risks silently producing label data on a different filtered
   population than the rest of this project, which would be worse than not
   running the experiment at all.
4. **Row count is constant across all N** (200,773 rows) -- this axis only
   varies the number of task columns, not training-set size.
5. **N only spans 1-128**, not the requested 5-2500. Extending the range
   requires the `intermediate/` regeneration described above.

Given these constraints, this experiment reuses the existing N=1-128 label
sets as the most useful thing that could be run without a risky,
unverified preprocessing step, and reports results with the above caveats
attached rather than silently presenting them as equivalent to the rest of
this repo's results.

**Vocab compatibility check**: canary runs at N=1 and N=2 produced val
AUROC (0.9991, 0.9814) matching known historical values from when these
exact label sets were originally used against the pre-deletion tensorized
cohort, confirming the restored cohort (`/groups/mm6677_gp/zzw2102/data/MIMIC_MEDS_restored/processed`)
is vocab-compatible with these older labels.

## Config

Same extreme-starved capacity cut as
[`results/extreme_capacity_starved_retrieval/`](../extreme_capacity_starved_retrieval/README.md)'s
Experiments 2/3:

| Param | Value |
| --- | --- |
| `encoder.embedding_dim` | 4 |
| `encoder.num_heads` | 1 |
| `encoder.num_layers` | 1 |
| `encoder.ff_dim` | 8 |
| `max_seq_len` | 8 |

`patient_only` (`fusion=passthrough`): masked-mean pool → `Linear(4, N)`.
`marginalized`: `query_projector=sequence_mean_1024` (`in_dim=4`) →
`HFDatasetRetriever` (K=4, full 125k-passage `MedRAG/textbooks` corpus) →
`TokenFeatureRetrievalEncoder` → `PerDocCrossAttentionFusion` (`d_model=256`)
→ `Linear(256, N)` → `marginalized_output_mode=binary`. `max_epochs=3`,
`warmup_steps=200` (fixed, since row count doesn't vary across N here).
Tensorized cohort: `/groups/mm6677_gp/zzw2102/data/MIMIC_MEDS_restored/processed`.

Test-split `marginalized` numbers use top-1-only retrieval
(`retriever.k=1, ablation_mode=none`), matching the convention established
in the other results file.

## Results

Val: `val/auroc/mean` / `val/loss`, computed once at end of fit. Test:
`test/auroc/mean` / `test/loss`, `eval_mode=test` scoring
`checkpoints/last.ckpt` against the held-out split.

| N | patient_only (val AUROC) | patient_only (val loss) | patient_only (test AUROC) | patient_only (test loss) | marginalized (val AUROC) | marginalized (val loss) | marginalized (test AUROC) | marginalized (test loss) | Δ (test AUROC) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.9991 | 0.0012 | 0.9649 | 0.0011 | 0.9926 | 0.0020 | 0.9716 | 0.0014 | +0.0068 |
| 2 | 0.9814 | 0.0021 | 0.7704 | 0.0007 | 0.9700 | 0.0025 | 0.8172 | 0.0006 | +0.0468 |
| 4 | 0.9801 | 0.0015 | 0.6715 | 0.0005 | 0.9958 | 0.0012 | 0.4288 | 0.0005 | -0.2426 |
| 8 | 0.9821 | 0.0015 | 0.8571 | 0.0010 | 0.9693 | 0.0016 | 0.8585 | 0.0010 | +0.0014 |
| 16 | 0.9850 | 0.0012 | 0.8871 | 0.0011 | 0.9733 | 0.0012 | 0.8669 | 0.0010 | -0.0202 |
| 32 | 0.9885 | 0.0013 | 0.8992 | 0.0011 | 0.9800 | 0.0014 | 0.8989 | 0.0011 | -0.0004 |
| 64 | 0.9885 | 0.0016 | 0.8634 | 0.0012 | 0.9805 | 0.0017 | 0.8752 | 0.0013 | +0.0118 |
| 128 | 0.9909 | 0.0014 | 0.8798 | 0.0010 | 0.9718 | 0.0016 | 0.8714 | 0.0011 | -0.0084 |

No consistent direction across N -- Δ ranges from -0.2426 (N=4) to +0.0468
(N=2), crossing zero repeatedly. N=4 is a striking single-draw outlier
(`marginalized` test AUROC 0.4288, below-chance, vs. `patient_only`'s
0.6715) -- with only one draw per level, this cannot be distinguished from
draw-level noise in a specific held-out split without repeated draws,
which these label sets don't have (see methodology caveat).
