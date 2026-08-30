# Random-doc and null-doc ablations on NWICU (+ cross-dataset key table)

Status: complete. `patient_only` baseline, both retrieval variants, and
frozen random-doc/null-doc ablations, training budget step-matched to
MIMIC's.

Scripts: [`nwicu_null_random_doc_ablations/scripts/`](../../nwicu_null_random_doc_ablations/scripts/)
(training sweeps, eval battery, aggregation) -- mechanical clones of the
MIMIC experiments' scripts (epochs, names, paths, ablation flags changed).
`random_docs` is `retriever.ablation_mode`; null corpus is the same
content-free artifact used on MIMIC (16 identical rows, all-zero 1024-d
keys, one pad token). Regenerable via
`cd nwicu_null_random_doc_ablations && .venv/bin/python scripts/aggregate_results.py`.

## Setup

**Step-matching**: NWICU has ~14x fewer training rows than MIMIC. NWICU
trains 42 epochs x 343 steps/epoch = 14,406 steps ≈ MIMIC's 3 x 4,788 =
14,364. Same capacity cut, same 125k-passage corpus, K=4, N=25/5-draw/30d
task construction as MIMIC; cosine LR schedule stretches with `max_epochs`,
warmup 200 steps (1.4% of budget) on both datasets.

**Checkpoint protocol** (verified by checksum, holds across this line):
`ModelCheckpoint(monitor=val/loss, save_last=True)` only rewrites
`last.ckpt` on val-loss improvement, so every eval below scores the
best-val-loss checkpoint. On NWICU this locks in near each arm's early val
peak (epochs 6-7 qwen3_text, 6-13 learned-linear) while the last-epoch val
sags below peak.

## Ablation design

Identical to the MIMIC round:

| Ablation | What changes | What stays |
| --- | --- | --- |
| frozen random | 4 retrieved docs replaced by 4 uniform draws from the full 125k corpus, redrawn every batch. Unseeded → 3 reps | trained weights; softmax-weighting over whatever docs arrive |
| frozen null | corpus swapped for 16 identical content-free docs (zero keys → uniform weights, one pad token) | trained weights |

## Key table (test AUROC, `held_out` split, mean ± population sd over 5 draws)

Training budgets step-matched across datasets (~14,400 optimizer steps: 3
epochs on MIMIC-IV, 42 epochs on NWICU).

**MIMIC-IV**

| variant | patient_only | real docs | random docs | null docs |
| --- | --- | --- | --- | --- |
| learned_linear | 0.8380 ± 0.0302 | 0.8477 ± 0.0287 | 0.8126 ± 0.0415 | 0.7859 ± 0.0376 |
| qwen3_text | 0.8380 ± 0.0302 | 0.8574 ± 0.0260 | 0.8473 ± 0.0264 | 0.8087 ± 0.0721 |

**NWICU**

| variant | patient_only | real docs | random docs | null docs |
| --- | --- | --- | --- | --- |
| learned_linear | 0.7472 ± 0.0066 | 0.7610 ± 0.0173 | 0.7587 ± 0.0203 | 0.7441 ± 0.0156 |
| qwen3_text | 0.7472 ± 0.0066 | 0.7492 ± 0.0103 | 0.7325 ± 0.0169 | 0.7047 ± 0.0190 |

## Results, per draw (test split, NWICU)

`patient_only`: 0.7374 / 0.7426 / 0.7474 / 0.7556 / 0.7529, mean 0.7472 ± 0.0066.

### learned-linear

| Draw | real docs | random docs (mean of 3 reps) | null docs |
| --- | --- | --- | --- |
| 1 | 0.7439 | 0.7461 | 0.7358 |
| 2 | 0.7365 | 0.7246 | 0.7225 |
| 3 | 0.7707 | 0.7709 | 0.7382 |
| 4 | 0.7786 | 0.7771 | 0.7590 |
| 5 | 0.7750 | 0.7746 | 0.7647 |
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

W&B runs named `nwicu42-*` at `wandb.ai/zzwang28-columbia-university/medrap`.

## Training dynamics

Val AUROC peak epochs (of 42): qwen3_text peaks epochs 5-10 (0.7248-0.7636),
learned-linear epochs 7-15 (0.7239-0.7796); `patient_only` peaks epochs
15-34 (0.7044-0.7731), no overfit. Retrieval arms' val sags ~0.03-0.06
below peak by the final epoch.

Gate diagnostics at end of training (softmax over the 4 retrieved docs'
scores): learned-linear effective-k 3.69, score std 1.43. qwen3_text
effective-k 4.00, score std 0.03 (near-uniform gate).

## Statistics (paired over the 5 shared draws; paired t, df=4, two-sided)

| Comparison | mean Δ | draws won | p | 95% CI |
| --- | --- | --- | --- | --- |
| ll real vs patient_only | +0.0138 | 4/5 | 0.080 | |
| ll real vs random | +0.0023 | 3/5 | n.s. | [−0.005, +0.009] |
| ll real vs null | +0.0169 | 5/5 | 0.018 | |
| q3 real vs patient_only | +0.0020 | 3/5 | 0.549 | [−0.007, +0.011] |
| q3 real vs random | +0.0167 | 5/5 | 0.017 | |
| q3 real vs null | +0.0444 | 5/5 | 0.007 | |
| (MIMIC, reference) ll real vs po | +0.0097 | 4/5 | 0.080 | |
| (MIMIC) q3 real vs po | +0.0194 | 5/5 | 0.004 | |

Power analysis: the two p=0.080 comparisons reach 84% power at 10 total
draws, 96% at 15.

## Query-text availability (learned-linear vs qwen3_text on NWICU)

NWICU has code descriptions for 31 of 2,159 codes (1.4%; 0.1% of events,
occurrence-weighted) vs. MIMIC's 1,402 of 11,476 (12.2%; 2.8% of events) --
28x less query signal for `qwen3_text`'s text-rendering path on NWICU.

## Caveats

- All evals score the best-val-loss checkpoint (== `last.ckpt`).
- `random_docs` draws are unseeded; random cells are means of 3 eval reps
  (rep spread ≤ 0.015).
- The null corpus is 4 identical content-free docs, removing both content
  and cross-document diversity.
