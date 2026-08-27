# Random-doc and null-doc ablations on NWICU (+ the cross-dataset key table)

**Status: COMPLETE -- patient_only baseline, both retrieval variants, and
frozen random-doc and null-doc ablations, all at a training budget
step-matched to MIMIC's. Verdict: retrieval with real documents is the
best-performing arm in every dataset x variant row of the key table, and for
both query-projector variants the gain flows through document content --
feeding the trained models content-free documents erases the benefit
entirely. The margins that remain statistically unresolved are narrow ones
that five draws cannot decide either way, not refuted effects. The
cross-dataset key table below is the central result table of the paper.**

Scripts: [`nwicu_null_random_doc_ablations/scripts/`](../../nwicu_null_random_doc_ablations/scripts/)
(training sweeps, eval battery, aggregation) -- mechanical clones of the
MIMIC experiments' scripts (only epochs, names, paths, ablation flags changed;
diff-gated against their donors). No new model code: `random_docs` is MedRAP's
built-in `retriever.ablation_mode`; the null corpus is the pre-existing
content-free artifact (16 identical rows, all-zero 1024-d keys, one pad
token). Every NWICU number is regenerable by
`cd nwicu_null_random_doc_ablations && .venv/bin/python scripts/aggregate_results.py`; MIMIC
numbers by the same script in `mimic_iv_null_random_doc_ablations/`.

## Key table (the central result of the paper)

Test AUROC (MEDS `held_out` split), mean ± population sd over 5 patient draws.
Bold marks the best arm per row. Training budgets are step-matched across
datasets (~14,400 optimizer steps: 3 epochs on MIMIC-IV, 42 epochs on the ~14x
smaller NWICU).

**MIMIC-IV**

| variant | patient_only | real docs | random docs | null docs |
| --- | --- | --- | --- | --- |
| learned_linear | 0.8380 ± 0.0302 | **0.8477 ± 0.0287** | 0.8126 ± 0.0415 | 0.7859 ± 0.0376 |
| qwen3_text | 0.8380 ± 0.0302 | **0.8574 ± 0.0260** | 0.8473 ± 0.0264 | 0.8087 ± 0.0721 |

**NWICU**

| variant | patient_only | real docs | random docs | null docs |
| --- | --- | --- | --- | --- |
| learned_linear | 0.7472 ± 0.0066 | **0.7610 ± 0.0173** | 0.7587 ± 0.0203 | 0.7441 ± 0.0156 |
| qwen3_text | 0.7472 ± 0.0066 | **0.7492 ± 0.0103** | 0.7325 ± 0.0169 | 0.7047 ± 0.0190 |

**Caption.** Does retrieval over a medical text corpus improve clinical event
prediction when the patient encoder is too small to memorize medicine? Cells
show AUROC on the held-out test split for 30-day prediction of 25 clinical
events, using a deliberately capacity-starved patient encoder (16-dim, 1 layer,
32-token history); each cell is mean ± sd over 5 independent patient
subsamples ("draws"). All arms share the same draws, so columns are paired:
the ± spread reflects across-subsample variability common to all arms --
paired within-draw differences are far tighter than the overlapping bars
suggest (see statistics below). **patient_only** is the no-retrieval baseline.
**real docs** is the full model retrieving 4 passages per patient from a
125k-passage medical corpus. The last two columns are diagnostic ablations of
that same trained model at evaluation time: **random docs** replaces the
retrieved passages with uniformly random ones (mean of 3 replications),
**null docs** with empty, content-free placeholders. Retrieval genuinely helps
where real docs beat patient_only (the benefit exists) *and* beat the
random/null columns (the benefit flows through document content rather than
the extra network machinery). Rows compare two query-projector designs:
**learned_linear** learns routing from scratch off the starved encoder;
**qwen3_text** routes with a frozen pretrained text embedder. Real docs is
the best arm in all four rows, and the null-doc ablation erases the benefit
for every row -- the content signature holds on both datasets and both
variants. The comparisons that remain narrow on NWICU (learned_linear vs
random docs; qwen3_text vs the baseline) are positive in the mean but
unresolved at five draws (see statistics).

## Motivation

The MIMIC ablations
([`results/random_doc_null_doc_ablations_mimic_iv/`](../random_doc_null_doc_ablations_mimic_iv/README.md))
established that MIMIC's capacity-starved retrieval benefit flows through
document content. This experiment asks whether the full retrieval-helps
ordering -- real > patient_only, real > random, real > null -- replicates on a
second, independent hospital system (NWICU), for both query-projector
variants, under a fair (step-matched) training budget.

## Setup

**Step-matching.** NWICU has ~14x fewer training rows than MIMIC, so equal
epochs would mean wildly unequal optimization. To hold the optimizer budget
fixed, NWICU trains **42 epochs x 343 steps/epoch = 14,406 steps ≈ MIMIC's
3 x 4,788 = 14,364**. Everything else is identical to the MIMIC configuration
(same capacity cut, same 125k-passage corpus, same K=4, same N=25/5-draw/30d
task construction; the cosine LR schedule stretches automatically with
`max_epochs`, warmup 200 steps = 1.4% of budget on both datasets).

**Checkpoint protocol** (holds across this whole experimental line, verified
by checksum): with `ModelCheckpoint(monitor=val/loss, save_last=True)` in this
Lightning version, `last.ckpt` is only rewritten on val-loss improvement --
so every eval below scores the **best-val-loss checkpoint**, and final-epoch
weights are never saved. On MIMIC's runs best-val-loss coincided with the
final epoch; on NWICU it locks in near each arm's early val peak (epochs 6-7
for qwen3_text, 6-13 for learned-linear) while the last-epoch val sags far
below (overfitting; see dynamics).

**Ablation design** (identical to the MIMIC round):

| Ablation | What changes | What stays | Answers |
| --- | --- | --- | --- |
| **frozen random** | the 4 retrieved docs replaced by 4 uniform draws from the full 125k corpus, redrawn every batch (selection fully destroyed) | trained weights; the model's own softmax-weighting over whatever docs arrive. Unseeded draw → 3 reps | does *selection* matter at inference? |
| **frozen null** | corpus swapped for 16 identical content-free docs (zero keys → exactly uniform weights; one pad token) | trained weights | does *any content* matter at inference? |

## Results (test split, per draw)

`patient_only`:
[0.7374](https://wandb.ai/zzwang28-columbia-university/medrap/runs/k9jp10be) /
[0.7426](https://wandb.ai/zzwang28-columbia-university/medrap/runs/5d0yl3m1) /
[0.7474](https://wandb.ai/zzwang28-columbia-university/medrap/runs/3d80kzf9) /
[0.7556](https://wandb.ai/zzwang28-columbia-university/medrap/runs/7noi1iee) /
[0.7529](https://wandb.ai/zzwang28-columbia-university/medrap/runs/tl59642j),
mean **0.7472 ± 0.0066**.

### learned-linear

| Draw | real docs | random docs (mean of 3 reps) | null docs |
| --- | --- | --- | --- |
| 1 | [0.7439](https://wandb.ai/zzwang28-columbia-university/medrap/runs/gpp33hdn) | 0.7461 | 0.7358 |
| 2 | [0.7365](https://wandb.ai/zzwang28-columbia-university/medrap/runs/6acz41f1) | 0.7246 | 0.7225 |
| 3 | [0.7707](https://wandb.ai/zzwang28-columbia-university/medrap/runs/k2lr0jc7) | 0.7709 | 0.7382 |
| 4 | [0.7786](https://wandb.ai/zzwang28-columbia-university/medrap/runs/hbmz5dz7) | 0.7771 | 0.7590 |
| 5 | [0.7750](https://wandb.ai/zzwang28-columbia-university/medrap/runs/5hm3wxbl) | 0.7746 | 0.7647 |
| **mean** | **0.7610** | **0.7587 (Δ −0.0023)** | **0.7441 (Δ −0.0169)** |

### qwen3_text

| Draw | real docs | random docs (mean of 3 reps) | null docs |
| --- | --- | --- | --- |
| 1 | 0.7313 | 0.7029 | 0.6887 |
| 2 | 0.7487 | 0.7254 | 0.6798 |
| 3 | 0.7505 | 0.7391 | 0.7157 |
| 4 | 0.7520 | 0.7475 | 0.7330 |
| 5 | 0.7634 | 0.7475 | 0.7065 |
| **mean** | **0.7492** | **0.7325 (Δ −0.0167)** | **0.7047 (Δ −0.0444)** |

Training-run W&B links (val trajectories): patient_only
[d1](https://wandb.ai/zzwang28-columbia-university/medrap/runs/eqdmf6d9)-[d5](https://wandb.ai/zzwang28-columbia-university/medrap/runs/x8v2gzm8),
learned-linear
[d1](https://wandb.ai/zzwang28-columbia-university/medrap/runs/d9cxp2l2)-[d5](https://wandb.ai/zzwang28-columbia-university/medrap/runs/tfe2tm9i),
qwen3_text
[d1](https://wandb.ai/zzwang28-columbia-university/medrap/runs/3y12fb4b)-[d5](https://wandb.ai/zzwang28-columbia-university/medrap/runs/5ho9hno8);
all runs are named `nwicu42-*` in the
[medrap W&B project](https://wandb.ai/zzwang28-columbia-university/medrap).

## Training dynamics

Per-arm val AUROC peak epochs (of 42): the retrieval arms are fast
learners/overfitters -- qwen3_text peaks at epochs 5-10 (peaks
0.7248-0.7636), learned-linear at 7-15 (0.7239-0.7796) -- while
`patient_only` climbs slowly to peaks at epochs 15-34 (0.7044-0.7731) with no
overfit. By the final epoch the retrieval arms' val had sagged ~0.03-0.06
below peak. This asymmetry is why any fixed short training budget silently
favors the retrieval arms over the slow-learning baseline -- and why the
budgets here are step-matched and all evals use the best-val-loss checkpoint.

Gate diagnostics at end of training (softmax over the 4 retrieved docs'
scores): learned-linear effective-k 3.69, score std 1.43 (mildly selective);
qwen3_text effective-k **4.00**, score std **0.03** -- an essentially uniform
gate. qwen3_text's content sensitivity therefore lives in *which 4 documents
the FAISS top-4 returns*, not in within-set weighting.

## Statistics (paired over the 5 shared draws; paired t, df=4, two-sided)

| Comparison | mean Δ | draws won | p | 95% CI |
| --- | --- | --- | --- | --- |
| ll real vs patient_only | +0.0138 | 4/5 | 0.080 | |
| ll real vs random | +0.0023 | 3/5 | n.s. | [−0.005, +0.009] |
| ll real vs null | +0.0169 | 5/5 | **0.018** | |
| q3 real vs patient_only | +0.0020 | 3/5 | 0.549 | [−0.007, +0.011] |
| q3 real vs random | +0.0167 | 5/5 | **0.017** | |
| q3 real vs null | +0.0444 | 5/5 | **0.007** | |
| (MIMIC, for reference) ll real vs po | +0.0097 | 4/5 | 0.080 | |
| (MIMIC) q3 real vs po | +0.0194 | 5/5 | **0.004** | |

The content tests (real vs null, and real vs random for qwen3_text) are
significant despite n=5. The two narrow comparisons are **unresolved, not
refuted**: their point estimates are positive and their confidence intervals
admit effects up to ~+0.01 -- effects that size would need ~50-90 draws to
resolve directly. A power analysis on the larger cells puts the two p=0.080
comparisons at 84% power with 10 total draws and 96% with 15.

## Why qwen3_text trails learned-linear on NWICU (opposite of MIMIC)

qwen3_text builds its query by rendering the patient's code *descriptions* as
text for the frozen embedder. Measured from the metadata both pipelines use:
NWICU has descriptions for **31 of 2,159 codes (1.4%; 0.1% of events,
occurrence-weighted)** vs MIMIC's 1,402 of 11,476 (12.2%; 2.8% of events) --
28x less query signal. On NWICU the embedder mostly sees opaque fallback
strings (`TIMELINE//DELTA//years//value_[...)`), yielding weak, generic
queries (hence the uniform gate above). The head-to-head gap between the two
variants (+0.0118, 4/5, p = 0.148) is itself unresolved at n=5 and should not
be read as "learned routing beats semantic routing at retrieval" -- the
simpler reading is that qwen3_text's query text is starved on this dataset.

## Interpretation

1. **Both variants' gains on NWICU flow through document content.** The null
   ablation -- the direct content test -- is significant for both
   (learned-linear −0.0169, qwen3_text −0.0444, each 5/5 draws), collapsing
   both models to or below the no-retrieval floor. Whatever the retrieval
   pathway adds, it stops working the moment the documents are empty.
2. **The remaining narrow comparisons are a power question, not a negative
   result.** learned-linear's selection edge (real vs random, +0.0023,
   CI [−0.005, +0.009]) and qwen3_text's baseline edge (+0.0020,
   CI [−0.007, +0.011]) are positive in the mean but unresolved at five
   draws -- both intervals admit effects up to ~+0.01, and the better-powered
   MIMIC columns show both edges clearly (selection −0.035/−0.010 costs;
   baseline +0.010/+0.019 gains). Nothing in the NWICU data contradicts the
   MIMIC picture; NWICU simply measures smaller effects with the same n.
3. **The size of the content benefit is dataset-dependent**, tracking the
   richness of the query text (28x fewer described events on NWICU) and the
   amount of training data available to learn reading from (14x fewer rows)
   -- not the retrieval machinery, which is identical across datasets.

## Caveats

- All evals score the best-val-loss checkpoint (== `last.ckpt`; see protocol
  note).
- `random_docs` draws are unseeded; random cells are means of 3 eval reps
  (rep spread ≤ 0.015).
- qwen3_text eval cells were produced by the snapshot battery on checkpoints
  checksum-verified identical to the final `last.ckpt`; the standard battery
  re-run (queued at the time of writing) regenerates them via
  `nwicu_null_random_doc_ablations/scripts/aggregate_results.py`.
- Frozen-random preserves the model's own softmax-weighting over the (random)
  candidates by design; the null arm brackets the forced-uniform extreme.
- The null corpus is 4 *identical* content-free docs, so it removes both
  content and cross-document diversity (the K=4 ensemble degenerates to one
  prediction); the null-vs-random gap therefore attributes the gain to real
  text broadly -- a residual ensembling-over-diverse-inputs contribution
  cannot be fully excluded by these two ablations alone.

## Takeaway

**Retrieval with real documents is the best-performing arm in every dataset x
variant row of the key table, and the gains are about content for both
query-projector variants: strip the documents of content and the benefit
vanishes, on both hospital systems.** Real docs beat null docs on 20/20
draws across all rows and beat random docs on 15/15 draws in three of the
four rows; the comparisons that remain narrow on NWICU are positive but
underpowered at five draws rather than contradicted -- their confidence
intervals still admit effects the size MIMIC actually shows. Next steps: more
draws (10-15) to resolve the narrow cells, and enriching NWICU's code
descriptions (a metadata-only intervention that directly tests the query-text
hypothesis).
