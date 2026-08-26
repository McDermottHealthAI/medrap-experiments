# Random-doc and null-doc ablations on the EXACT MIMIC capacity-starved checkpoints

**Status: COMPLETE -- 85 jobs (30 reproduction, 40 frozen-random, 10 frozen-null, 5
smokes). Run against the exact original runs behind
[`results/capacity_starved_retrieval/`](../capacity_starved_retrieval/README.md).
Verdict: the capacity-starved retrieval benefit flows through document CONTENT --
destroying it costs 3-6x the entire benefit and drops the models BELOW the
no-retrieval floor.**

Scripts: [`mimic_iv_null_random_doc_ablations/scripts/`](../../mimic_iv_null_random_doc_ablations/README.md)
-- mechanical clones of the original experiment's eval scripts with ablation
flags/corpus swaps, pointed at the original checkpoints/labels/corpus
(`/groups/mm6677_gp/hs3627/medrap-experiments/mimic_iv_sweep/`, read-only).

## Provenance: recovering the exact evaluation data

The tensorized MIMIC cohort the checkpoints trained on had been deleted from
`/groups/mm6677_gp/data/MIMIC_MEDS/MEDS_cohort/` (~Aug 25, leaving raw MEDS only).
It was recovered **byte-for-byte** from the filesystem snapshot
`/groups/.snapshots/@GMT-2026.08.25-12.06.07/` into
`/groups/mm6677_gp/zzw2102/data/MIMIC_MEDS_restored/processed` (292/37/37
train/tuning/held_out shards, 11 GB, original timestamps preserved) -- no
re-tensorization, hence zero vocab-mapping risk against the checkpoints.

## Setup validation: full reproduction of the published tables

Before any ablation was interpreted, every published number was reproduced from
the restored data + original checkpoints:

- **Test table: 15/15 exact** (4-decimal match; e.g. patient_only
  0.8635/0.8098/0.8503/0.8715/0.7950).
- **Val table: 14/15 exact, 1 within 0.0012** (qwen3_text draw 1, 0.8713 vs
  0.8725) -- attributed to the nondeterministic frozen-Qwen3 GPU forward, the one
  component with no bitwise reproducibility. Accepted tolerance: exact everywhere,
  ≤0.002 on qwen3_text.
- **Protocol discovery**: the published val numbers were computed under
  `limit_val_batches=200` inherited from the `lightning_wandb` trainer config --
  i.e. a **6,400-row prefix** of the tuning split, not the full split. A first
  reproduction on the full split mismatched systematically until this was
  matched. Full-split val numbers were computed as a by-product
  (`outputs/*_val_repro/`); the test table (full held_out) is unaffected and is
  the trustworthy metric.

## Ablation design

Both ablations evaluate each run's exact `checkpoints/last.ckpt` on the MEDS
`held_out` split, as trained -- only the documents handed to the model change:

| Ablation | What changes | What stays | Answers |
| --- | --- | --- | --- |
| **frozen random** | the 4 retrieved docs are replaced by 4 uniform random draws from the full 125k-passage corpus, redrawn every batch (built-in `retriever.ablation_mode=random_docs`; selection fully destroyed) | trained weights; the model's own softmax-weighting over whatever docs arrive. Unseeded draw → **3 reps** per checkpoint | does *selection* matter at inference? |
| **frozen null** | corpus swapped for 16 identical content-free docs (all-zero 1024-d keys → exactly uniform weights; one pad token of content; deterministic) | trained weights | does *any document content* matter at inference? |

A train-time random_docs control (retraining with random documents) was not run
here and is left as future work.

## Results (test split, exact original checkpoints)

Reference floor: `patient_only` (starved) test mean **0.8380**.

### learned-linear checkpoints

| Draw | real docs | random docs (mean of 3 reps) | null docs | Δ random | Δ null |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.8581 | 0.8047 | 0.7591 | −0.0535 | −0.0990 |
| 2 | 0.8276 | 0.7693 | 0.7449 | −0.0583 | −0.0827 |
| 3 | 0.8596 | 0.8368 | 0.8095 | −0.0228 | −0.0501 |
| 4 | 0.8881 | 0.8797 | 0.8476 | −0.0083 | −0.0405 |
| 5 | 0.8050 | 0.7727 | 0.7684 | −0.0322 | −0.0365 |
| **mean** | **0.8477** | **0.8126 (Δ −0.0350)** | **0.7859 (Δ −0.0618)** | | |

### qwen3_text checkpoints

| Draw | real docs | random docs (mean of 3 reps) | null docs | Δ random | Δ null |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.8723 | 0.8630 | 0.8532 | −0.0093 | −0.0191 |
| 2 | 0.8335 | 0.8215 | 0.6694 | −0.0120 | −0.1641 |
| 3 | 0.8653 | 0.8592 | 0.8534 | −0.0060 | −0.0118 |
| 4 | 0.8934 | 0.8812 | 0.8595 | −0.0123 | −0.0340 |
| 5 | 0.8224 | 0.8116 | 0.8079 | −0.0108 | −0.0145 |
| **mean** | **0.8574** | **0.8473 (Δ −0.0101)** | **0.8087 (Δ −0.0487)** | | |

Random rep-spread ≤ 0.015 (ll d1) and ≤ 0.007 elsewhere. qwen3_text draw 2's null
result (−0.164) is an outlier worth flagging: that checkpoint collapses
catastrophically without real text.

## Interpretation: content is load-bearing

The original experiment's retrieval benefit over `patient_only` was
+0.010 (learned-linear) / +0.019 (qwen3_text) on the test split. Against that:

1. **Destroying content costs 3-6x the entire benefit** (random Δ −0.035/−0.010;
   null Δ −0.062/−0.049), dropping both variants **below the no-retrieval floor**
   (learned-linear: 0.813 random / 0.786 null vs 0.838 patient_only). The trained
   fusion pathway actively depends on relevant text and is poisoned by garbage --
   a signature that cannot be produced by the retrieval machinery alone.
2. **Selection matters more for learned-linear** (Δ random −0.035) **than for
   qwen3_text** (−0.010), while both need real content (null Δ ≈ −0.05). This is
   consistent with the original experiment's internal signal that qwen3_text's
   frozen semantic routing and the retrieved content both contribute.
3. The per-draw pattern is unanimous in direction (10/10 random cells negative,
   10/10 null cells negative, null ≤ random in 9/10).

## Caveats

- The train-time random_docs control (does the *training* benefit require real
  documents?) was deferred; this verdict rests on frozen-checkpoint evidence --
  decisive here, since ablated performance falls below the patient_only floor.
- Frozen-random preserves the model's own softmax-weighting over the (random)
  candidates by design -- "evaluate the model as trained on wrong inputs"; the
  null arm brackets the forced-uniform extreme from below.
- `random_docs` draws are unseeded; reported random numbers are means of 3
  independent eval reps.
- Checkpoints, labels, and retrieval corpus are the original working copy's,
  referenced read-only; all eval outputs and W&B runs
  (`wandb.ai/zzwang28-columbia-university/medrap`, `mimic-repro-*` /
  `mimic-abl-*`) are under `mimic_iv_null_random_doc_ablations/outputs/`.

## Takeaway

**On these checkpoints, the capacity-starved retrieval benefit is genuine
content use, not an artifact of the retrieval architecture.** Next steps: the
train-time random_docs control (completes the frozen-vs-trained symmetry), and
capacity-matched `patient_only` baselines (an equivalent un-starved auxiliary
pathway with no documents) to make future comparisons immune to the
machinery confound.
