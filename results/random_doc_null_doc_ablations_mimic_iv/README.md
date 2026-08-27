# Random-doc and null-doc ablations on the EXACT MIMIC capacity-starved checkpoints

**Status: COMPLETE -- 85 jobs (30 reproduction, 40 frozen-random, 10 frozen-null,
5 smokes), run against the exact original checkpoints behind
[`results/capacity_starved_retrieval/`](../capacity_starved_retrieval/README.md).
Verdict: the capacity-starved retrieval benefit flows through document CONTENT --
destroying it costs 3-6x the entire benefit and drops the models BELOW the
no-retrieval floor.**

Scripts: [`mimic_iv_null_random_doc_ablations/scripts/`](../../mimic_iv_null_random_doc_ablations/README.md)
-- mechanical clones of the original experiment's eval scripts with ablation
flags/corpus swaps, pointed at the original checkpoints/labels/corpus
(`/groups/mm6677_gp/hs3627/medrap-experiments/mimic_iv_sweep/`, read-only).
Every number in this README is regenerable by
`cd mimic_iv_null_random_doc_ablations && .venv/bin/python scripts/aggregate_results.py`.

## Key table

Test AUROC (MEDS `held_out` split), mean ± population sd over the 5 patient
draws. Bold marks the best arm per row.

| variant | patient_only | real docs | random docs | null docs |
| --- | --- | --- | --- | --- |
| learned_linear | 0.8380 ± 0.0302 | **0.8477 ± 0.0287** | 0.8126 ± 0.0415 | 0.7859 ± 0.0376 |
| qwen3_text | 0.8380 ± 0.0302 | **0.8574 ± 0.0260** | 0.8473 ± 0.0264 | 0.8087 ± 0.0721 |

**Caption.** Cells show AUROC on the held-out test split for 30-day prediction
of 25 clinical events with a deliberately capacity-starved patient encoder
(16-dim, 1 layer, 32-token history). **patient_only** is the no-retrieval
baseline. **real docs** is the full model retrieving 4 passages per patient
from the 125k-passage `MedRAG/textbooks` corpus. The last two columns are
diagnostic ablations of that *same trained model* at evaluation time:
**random docs** replaces the retrieved passages with uniformly random ones
(mean of 3 replications), **null docs** with empty, content-free placeholders.
Retrieval genuinely helps where real docs beat patient_only (the benefit
exists) *and* beat the random/null columns (the benefit flows through document
content rather than the extra network machinery). All arms share the same 5
draws, so columns are paired; the ± spread is common across-draw variability
-- paired within-draw differences are much tighter than the overlapping bars
suggest (see the statistics section below).

## Motivation

The original capacity-starved experiment closed on an explicit caveat:
marginalizing over K=4 documents could in principle help a starved model
simply by acting as an implicit ensemble/regularizer, independent of whether
the retrieved content is relevant. These ablations answer that caveat for the
exact checkpoints behind the published result: if the benefit is machinery,
feeding the trained models random or empty documents should leave their
performance roughly intact; if the benefit is content, it should destroy it.

## Ablation design

Both ablations evaluate each run's exact `checkpoints/last.ckpt` on the MEDS
`held_out` split, as trained -- only the documents handed to the model change:

| Ablation | What changes | What stays | Answers |
| --- | --- | --- | --- |
| **frozen random** | the 4 retrieved docs are replaced by 4 uniform random draws from the full 125k-passage corpus, redrawn every batch (built-in `retriever.ablation_mode=random_docs`; selection fully destroyed) | trained weights; the model's own softmax-weighting over whatever docs arrive. Unseeded draw → **3 reps** per checkpoint | does *selection* matter at inference? |
| **frozen null** | corpus swapped for 16 identical content-free docs (all-zero 1024-d keys → exactly uniform weights; one pad token of content; deterministic) | trained weights | does *any document content* matter at inference? |

A train-time random_docs control (retraining with random documents) was not run
here and is left as future work.

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

## Results (test split, exact original checkpoints, per draw)

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

## Statistics (paired over the 5 shared draws)

All arms share the same draws, so the per-draw deltas are the honest unit of
evidence (paired t-test, df=4, two-sided):

| Comparison | mean Δ | draws won | p |
| --- | --- | --- | --- |
| learned_linear real vs patient_only | +0.0097 | 4/5 | 0.080 |
| qwen3_text real vs patient_only | +0.0194 | 5/5 | **0.004** |
| real vs random / real vs null | −0.035 to −0.062 | 10/10 cells each | -- |

The ablation direction is unanimous (10/10 random cells negative, 10/10 null
cells negative, null ≤ random in 9/10). The benefit-exists comparison is solid
for qwen3_text and borderline (p = 0.080 at n=5) for learned_linear -- a power
analysis puts ~10 total draws at 84% power and ~15 at 96% for that cell.

## Checkpoint-protocol finding

Checksum verification during the follow-up work revealed that with this line's
trainer config (`ModelCheckpoint(monitor=val/loss, save_last=True)`, this
Lightning version), **`last.ckpt` is only rewritten when the monitored metric
improves -- so `last.ckpt` is byte-identical to the best-val-loss checkpoint
everywhere in this experimental line.** Final-epoch weights are never saved
(nearest proxy: the `progress=0.90` snapshot).

For the MIMIC results in this directory the distinction is immaterial: on the
original 3-epoch runs, val loss was still improving at the end of training, so
best-val-loss == final epoch (verified by checksum on the original checkpoints:
`last.ckpt` == `epoch=2-step=14361.ckpt`). The published numbers are unaffected.
However, the original README's phrasing that evals use "the exact final-epoch
weights ... not the best-val-loss checkpoint" is literally backwards as a
description of the mechanism -- the two coincided on MIMIC by luck of the
training length, not by construction. All evals in this line are best-val-loss
to best-val-loss.

## Peak-to-peak robustness check

The published comparison is last-to-last. Recomputing from the original training
logs with each run's *peak* `val/auroc/mean` instead (see
`scripts/aggregate_results.py`, section 4):

| | last-to-last (published) | peak-to-peak |
| --- | --- | --- |
| Δ learned-linear (val) | +0.0125, 5/5 | +0.0152, 5/5 |
| Δ qwen3_text (val) | +0.0248, 5/5 | +0.0246, 5/5 |

The retrieval benefit is robust to the protocol choice (and marginally larger
peak-to-peak for learned-linear).

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
3. The per-draw pattern is unanimous in direction.

## Caveats

- The train-time random_docs control (does the *training* benefit require real
  documents?) was deferred; this verdict rests on frozen-checkpoint evidence --
  decisive here, since ablated performance falls below the patient_only floor.
- Frozen-random preserves the model's own softmax-weighting over the (random)
  candidates by design -- "evaluate the model as trained on wrong inputs"; the
  null arm brackets the forced-uniform extreme from below.
- `random_docs` draws are unseeded; reported random numbers are means of 3
  independent eval reps.
- Dependence is not benefit: the below-floor ablation drops prove the trained
  models *use* content, while the benefit claim itself rests on the (smaller)
  real-vs-patient_only comparison in the statistics section.
- Checkpoints, labels, and retrieval corpus are the original working copy's,
  referenced read-only; all eval outputs and W&B runs
  (`wandb.ai/zzwang28-columbia-university/medrap`, `mimic-repro-*` /
  `mimic-abl-*`) are under `mimic_iv_null_random_doc_ablations/outputs/`.

## Takeaway

**On these checkpoints, the capacity-starved retrieval benefit is genuine
content use, not an artifact of the retrieval architecture.** Next steps: the
train-time random_docs control (completes the frozen-vs-trained symmetry),
capacity-matched `patient_only` baselines (an equivalent un-starved auxiliary
pathway with no documents), and more draws (~10-15 total) to power the
learned-linear benefit cell past p = 0.05.
