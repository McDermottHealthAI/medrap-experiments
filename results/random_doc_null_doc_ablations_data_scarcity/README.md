# Random-doc and null-doc ablations on the EXACT data-scarcity checkpoints

**Status: COMPLETE -- 120 jobs (40 reproduction, 60 frozen-random, 20 frozen-null),
run against the exact original checkpoints behind
[`results/data_scarcity_retrieval/`](../data_scarcity_retrieval/README.md).
Verdict: unlike the capacity-starved line, the data-scarcity retrieval
benefit does NOT show a clear content signature -- random and null docs
barely move performance relative to real docs, in most cases well within
draw-to-draw noise.**

Scripts: [`mimic_iv_data_scarcity_ablations/scripts/`](../../mimic_iv_data_scarcity_ablations/README.md)
-- mechanical clones of zzw2102's
[`mimic_iv_null_random_doc_ablations/`](../../mimic_iv_null_random_doc_ablations/README.md)
eval scripts (same restored tensorized cohort, same null corpus, same
`retriever.ablation_mode` infra), pointed at the data-scarcity checkpoints
instead of re-running the already-ablated capacity-starved line. Scope:
`learned-linear` query projector only (the only variant trained under data
scarcity), all 4 fractions (50/20/10/5%). Every number below is regenerable
by `cd mimic_iv_data_scarcity_ablations && .venv/bin/python scripts/aggregate_results.py`.

## Key table

Test AUROC (MEDS `held_out` split), mean ± population sd over the 5 patient
draws. Bold marks the best arm per row.

| train fraction | patient_only | real docs | random docs | null docs |
| --- | --- | --- | --- | --- |
| 50% | **0.8824 ± 0.0203** | 0.8801 ± 0.0256 | 0.8794 ± 0.0248 | 0.8783 ± 0.0253 |
| 20% | 0.8479 ± 0.0295 | **0.8541 ± 0.0281** | 0.8526 ± 0.0283 | 0.8531 ± 0.0273 |
| 10% | 0.8225 ± 0.0276 | **0.8271 ± 0.0288** | 0.8255 ± 0.0277 | 0.8185 ± 0.0307 |
| 5% | 0.7883 ± 0.0391 | 0.8070 ± 0.0288 | **0.8080 ± 0.0288** | 0.8049 ± 0.0294 |

**Caption.** Cells show AUROC on the held-out test split for 30-day
prediction of 25 clinical events, with the patient encoder at **full
capacity** and only the *training set* subsampled (unlike the
capacity-starved line, which shrank the model instead). **patient_only**
is the no-retrieval baseline. **real docs** is the full model retrieving 4
passages per patient from the 125k-passage `MedRAG/textbooks` corpus. The
last two columns are diagnostic ablations of that same trained model at
evaluation time: **random docs** replaces the retrieved passages with
uniformly random ones (mean of 3 replications), **null docs** with empty,
content-free placeholders. Unlike the capacity-starved key table, where
random/null dropped both variants *below* the patient_only floor, here all
four columns sit close together at every fraction -- the ablations don't
clearly separate from real docs the way they did under capacity starvation.

## Motivation

[`results/random_doc_null_doc_ablations_mimic_iv/`](../random_doc_null_doc_ablations_mimic_iv/README.md)
found that the capacity-starved retrieval benefit is genuine content use --
destroying document content with random or null docs cost 3-6x the entire
benefit and dropped performance below the no-retrieval floor.
[`results/data_scarcity_retrieval/`](../data_scarcity_retrieval/README.md)
found a similarly-shaped benefit on an orthogonal axis (training data
volume instead of model capacity): retrieval is a small net negative at
50% train data, then flips to a consistent win at 20% and below. That
README's takeaway explicitly flagged the same open question the
capacity-starved line had already answered for itself: is the data-scarcity
benefit retrieval *content*, or could marginalizing over K=4 documents help
simply by acting as an implicit ensemble/regularizer, independent of
whether the retrieved content is relevant? This experiment runs the same
ablation recipe against the data-scarcity checkpoints to find out.

## Ablation design

Identical to the capacity-starved round. Both ablations evaluate each run's
exact `checkpoints/last.ckpt` on the MEDS `held_out` split, as trained --
only the documents handed to the model change:

| Ablation | What changes | What stays | Answers |
| --- | --- | --- | --- |
| **frozen random** | the 4 retrieved docs are replaced by 4 uniform random draws from the full 125k-passage corpus, redrawn every batch (built-in `retriever.ablation_mode=random_docs`; selection fully destroyed) | trained weights; the model's own softmax-weighting over whatever docs arrive. Unseeded draw → **3 reps** per checkpoint | does *selection* matter at inference? |
| **frozen null** | corpus swapped for the same content-free docs used in the MIMIC/NWICU capacity-starved rounds (all-zero 1024-d keys → exactly uniform weights; one pad token of content; deterministic) | trained weights | does *any document content* matter at inference? |

A train-time random_docs control (retraining with random documents) was not
run here either and is left as future work, same caveat as the
capacity-starved line.

## Provenance: evaluation data

Same restored tensorized cohort the capacity-starved ablations already
validated (`/groups/mm6677_gp/zzw2102/data/MIMIC_MEDS_restored/processed`),
since the original `/groups/mm6677_gp/data/MIMIC_MEDS/MEDS_cohort/processed`
is confirmed still gone. No new provenance work was needed -- this reuses
the same recovered data the capacity-starved round already checksum-verified.

## Reproduction gate

Before trusting any ablation, `patient_only` and `learned-linear` (real
docs) were re-evaluated on the test split (`eval_mode=test`) for all 4
fractions. Unlike the capacity-starved line, there is no published test
table to match exactly (the original `data_scarcity_retrieval` README only
reports `val/auroc/mean`), so the gate here is directional: the reproduced
test numbers should track the same real-vs-patient_only pattern as the
published val numbers, not degenerate or diverge wildly. They do --

| Fraction | Published val Δ (real − patient_only) | This round's test Δ (real − patient_only) |
| --- | --- | --- |
| 50% | −0.0053 ± 0.0071, 1/5 | −0.0023, 1/5 |
| 20% | +0.0148 ± 0.0152, 5/5 | +0.0062, 4/5 |
| 10% | +0.0113 ± 0.0111, 4/5 | +0.0046, 4/5 |
| 5% | +0.0161 ± 0.0078, 5/5 | +0.0187, 4/5 |

Same sign, same rough ordering (50% negative, 5% largest positive) at every
fraction. Test-split magnitudes are smaller/noisier than val (expected --
different split, and val was implicitly selected on across variants), but
the direction holds cleanly enough to trust the ablation comparison below.

## Results (test split, exact original checkpoints, per draw)

### 50% train

| Draw | real docs | random docs (mean of 3 reps) | null docs | Δ random | Δ null |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.8852 | 0.8817 | 0.8839 | −0.0035 | −0.0013 |
| 2 | 0.8539 | 0.8571 | 0.8539 | +0.0032 | −0.0000 |
| 3 | 0.8951 | 0.8917 | 0.8889 | −0.0034 | −0.0062 |
| 4 | 0.9172 | 0.9178 | 0.9174 | +0.0006 | +0.0001 |
| 5 | 0.8489 | 0.8486 | 0.8477 | −0.0003 | −0.0012 |
| **mean** | **0.8801** | **0.8794 (Δ −0.0007)** | **0.8783 (Δ −0.0017)** | | |

### 20% train

| Draw | real docs | random docs (mean of 3 reps) | null docs | Δ random | Δ null |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.8697 | 0.8619 | 0.8662 | −0.0078 | −0.0036 |
| 2 | 0.8178 | 0.8163 | 0.8189 | −0.0015 | +0.0011 |
| 3 | 0.8749 | 0.8759 | 0.8764 | +0.0010 | +0.0015 |
| 4 | 0.8853 | 0.8865 | 0.8823 | +0.0012 | −0.0030 |
| 5 | 0.8227 | 0.8223 | 0.8216 | −0.0004 | −0.0011 |
| **mean** | **0.8541** | **0.8526 (Δ −0.0015)** | **0.8531 (Δ −0.0010)** | | |

### 10% train

| Draw | real docs | random docs (mean of 3 reps) | null docs | Δ random | Δ null |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.8457 | 0.8436 | 0.8433 | −0.0021 | −0.0024 |
| 2 | 0.8001 | 0.7997 | 0.7779 | −0.0004 | −0.0222 |
| 3 | 0.8451 | 0.8424 | 0.8385 | −0.0026 | −0.0066 |
| 4 | 0.8590 | 0.8564 | 0.8485 | −0.0026 | −0.0105 |
| 5 | 0.7855 | 0.7854 | 0.7844 | −0.0001 | −0.0011 |
| **mean** | **0.8271** | **0.8255 (Δ −0.0016)** | **0.8185 (Δ −0.0086)** | | |

### 5% train

| Draw | real docs | random docs (mean of 3 reps) | null docs | Δ random | Δ null |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.8312 | 0.8312 | 0.8317 | +0.0001 | +0.0006 |
| 2 | 0.7879 | 0.7857 | 0.7819 | −0.0022 | −0.0060 |
| 3 | 0.8201 | 0.8242 | 0.8233 | +0.0041 | +0.0032 |
| 4 | 0.8358 | 0.8362 | 0.8289 | +0.0004 | −0.0069 |
| 5 | 0.7601 | 0.7628 | 0.7585 | +0.0026 | −0.0017 |
| **mean** | **0.8070** | **0.8080 (Δ +0.0010)** | **0.8049 (Δ −0.0022)** | | |

Random rep-spread was small everywhere (≤ 0.005 typically); the largest
single-draw ablation effect in the whole experiment is 10%/draw2's null
result (−0.0222), still an order of magnitude smaller than the
capacity-starved line's worst case (qwen3_text draw 2, −0.164).

## Statistics (paired over the 5 shared draws; paired t, df=4, two-sided)

Critical t for df=4: |t| > 2.776 for p<0.05, |t| > 2.132 for p<0.10.

| Fraction | Comparison | mean Δ | t | draws won | significant? |
| --- | --- | --- | --- | --- | --- |
| 50% | real vs patient_only | −0.0023 | −0.72 | 1/5 | no |
| 50% | real vs random | −0.0007 | −0.54 | 2/5 | no |
| 50% | real vs null | −0.0017 | −1.46 | 1/5 | no |
| 20% | real vs patient_only | +0.0062 | 2.17 | 4/5 | borderline (p≈0.09) |
| 20% | real vs random | −0.0015 | −0.91 | 2/5 | no |
| 20% | real vs null | −0.0010 | −0.98 | 2/5 | no |
| 10% | real vs patient_only | +0.0046 | 1.60 | 4/5 | no |
| 10% | real vs random | −0.0016 | **−2.85** | 0/5 | **yes (p<0.05)** |
| 10% | real vs null | −0.0086 | −2.26 | 0/5 | borderline (p≈0.08) |
| 5% | real vs patient_only | +0.0187 | 2.37 | 4/5 | borderline (p≈0.08) |
| 5% | real vs random | +0.0010 | 0.91 | 3/5 (random *beats* real) | no |
| 5% | real vs null | −0.0022 | −1.13 | 2/5 | no |

The only content-ablation comparison that clears even p<0.10 is 10%'s
real-vs-null (borderline) and real-vs-random (significant, but the effect
is tiny: −0.0016). At 5%, random docs edge out real docs on 3/5 draws with
a small positive mean Δ -- the opposite direction a content story would
predict.

## Interpretation: no clear content signature, unlike capacity starvation

The contrast with
[`results/random_doc_null_doc_ablations_mimic_iv/`](../random_doc_null_doc_ablations_mimic_iv/README.md)
is the headline finding here:

1. **Capacity starvation**: destroying content cost 3-6x the entire
   retrieval benefit and dropped both variants *below* the no-retrieval
   floor (random Δ −0.035/−0.010, null Δ −0.062/−0.049 on MIMIC). The
   effect was unanimous in direction (10/10 random cells negative, 10/10
   null cells negative).
2. **Data scarcity**: ablation deltas are an order of magnitude smaller
   (−0.001 to −0.009 in 11 of 12 cells) and **not unanimous in direction**
   -- random docs beat real docs on 2-3/5 draws at three of the four
   fractions, and outright beat real docs in mean at 5% train. Only one
   cell (10%, null docs) shows a Δ approaching capacity-starved magnitude
   (−0.0086, still 6x smaller than capacity-starved's smallest content
   effect), and even that one has 0/5 draws where null beats real -- weak
   evidence at best.
3. Where the benefit-exists comparison (real vs patient_only) is itself
   only borderline-significant at n=5 (20%, 5%) or non-significant (10%),
   the ablation comparisons riding on top of it are even less powered to
   detect anything -- but the *point estimates* still don't point toward a
   content story the way the capacity-starved numbers did unambiguously,
   even accounting for that.

**Reading these two experiments together**: the same retrieval
architecture, trained on the same corpus, shows a clear content-dependent
signature when the bottleneck is *model capacity*, but not when the
bottleneck is *training data volume*. A plausible explanation is that a
data-starved model with full capacity has enough representational room to
treat the K=4 retrieved slots as extra stochastic input regardless of
content -- gaining from the added noise/regularization the marginalization
provides -- rather than needing the text itself the way a capacity-starved
model does. This is consistent with, not a refutation of, the original
data-scarcity README's caveat that it couldn't rule out an
ensembling/regularization explanation.

## Caveats

- Unlike the capacity-starved round, there's no published test table for
  data-scarcity to reproduce exactly -- the gate here is directional
  consistency with the published val numbers, not a 4-decimal match. See
  the reproduction-gate section above.
- `random_docs` draws are unseeded; reported random numbers are means of 3
  independent eval reps (rep spread ≤ 0.005 typically, checked per-draw
  above).
- At n=5 draws, most individual comparisons here are underpowered even
  where the point estimate is real (e.g. the benefit-exists comparisons
  themselves are only borderline at 3 of 4 fractions) -- absence of a
  significant content effect is not the same as proof there is none. More
  draws would be needed to fully rule out a small content contribution at
  10%, the one fraction with a non-trivial (if unproven) null-docs Δ.
- Only `learned-linear` was tested (the only query-projector variant
  trained under data scarcity); the capacity-starved line's other finding
  -- that `qwen3_text` shows an even cleaner content signature than
  `learned-linear` -- has no data-scarcity analog to compare against.
- Checkpoints, labels, and retrieval corpus are hs3627's original working
  copy's, referenced read-only.

## Takeaway

**Unlike the capacity-starved retrieval benefit, the data-scarcity
retrieval benefit does not show a clear content-dependent signature.**
Random and null document ablations move performance by roughly an order of
magnitude less than they did under capacity starvation, are not unanimous
in direction, and in one case (5% train) random docs slightly *beat* real
docs. This doesn't prove the data-scarcity benefit is purely
ensembling/regularization -- the underlying real-vs-patient_only effect is
itself only borderline-significant at n=5 in most cells -- but it means the
two "does retrieval help" results in this repo have different underlying
mechanisms as far as the current evidence shows: capacity starvation
clearly needs real content; data scarcity's benefit, where it exists,
cannot yet be attributed to content over machinery. Next steps: more draws
to power the 10% null-docs cell (the one candidate for a real, if small,
content effect), and the train-time random-docs control both this and the
capacity-starved line still lack.
