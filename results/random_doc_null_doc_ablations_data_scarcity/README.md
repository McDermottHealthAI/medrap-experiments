# Random-doc and null-doc ablations on the EXACT data-scarcity checkpoints

Status: complete. 120 jobs (40 reproduction, 60 frozen-random, 20
frozen-null) plus 20 top1-only jobs, run against the exact original
checkpoints behind [`results/data_scarcity_retrieval/`](../data_scarcity_retrieval/README.md).

Scripts: [`mimic_iv_data_scarcity_ablations/scripts/`](../../mimic_iv_data_scarcity_ablations/README.md)
-- mechanical clones of the [`mimic_iv_null_random_doc_ablations/`](../../mimic_iv_null_random_doc_ablations/README.md)
eval scripts (same restored tensorized cohort, same null corpus, same
`retriever.ablation_mode` infra), pointed at the data-scarcity checkpoints
instead of re-running the already-ablated capacity-starved line. Scope:
`learned-linear` query projector only (the only variant trained under data
scarcity), all 4 fractions (50/20/10/5%). Regenerable via
`cd mimic_iv_data_scarcity_ablations && .venv/bin/python scripts/aggregate_results.py`.

## Ablation design

Identical to the capacity-starved round. Both ablations evaluate each
run's exact `checkpoints/last.ckpt` on the MEDS `held_out` split, as
trained -- only the documents handed to the model change:

| Ablation | What changes | What stays |
| --- | --- | --- |
| frozen random | 4 retrieved docs replaced by 4 uniform random draws from the full 125k-passage corpus, redrawn every batch (`retriever.ablation_mode=random_docs`), 3 reps per checkpoint | trained weights; softmax-weighting over whatever docs arrive |
| frozen null | corpus swapped for the same content-free docs used in the MIMIC/NWICU capacity-starved rounds (all-zero 1024-d keys, one pad token) | trained weights |

## Provenance

Same restored tensorized cohort as the capacity-starved ablations
(`/groups/mm6677_gp/zzw2102/data/MIMIC_MEDS_restored/processed`); original
`/groups/mm6677_gp/data/MIMIC_MEDS/MEDS_cohort/processed` confirmed still
gone.

## Reproduction gate

No published test table exists for data-scarcity (only `val/auroc/mean`
was published), so the gate is directional: reproduced test Δ tracks the
same sign/ordering as the published val Δ.

| Fraction | Published val Δ (real − patient_only) | Test Δ (real − patient_only) |
| --- | --- | --- |
| 50% | −0.0053 ± 0.0071, 1/5 | −0.0023, 1/5 |
| 20% | +0.0148 ± 0.0152, 5/5 | +0.0062, 4/5 |
| 10% | +0.0113 ± 0.0111, 4/5 | +0.0046, 4/5 |
| 5% | +0.0161 ± 0.0078, 5/5 | +0.0187, 4/5 |

## Key table

Test AUROC (`held_out`), mean ± population sd over the 5 draws.

| train fraction | patient_only | real docs | random docs | null docs |
| --- | --- | --- | --- | --- |
| 50% | 0.8824 ± 0.0203 | 0.8801 ± 0.0256 | 0.8794 ± 0.0248 | 0.8783 ± 0.0253 |
| 20% | 0.8479 ± 0.0295 | 0.8541 ± 0.0281 | 0.8526 ± 0.0283 | 0.8531 ± 0.0273 |
| 10% | 0.8225 ± 0.0276 | 0.8271 ± 0.0288 | 0.8255 ± 0.0277 | 0.8185 ± 0.0307 |
| 5% | 0.7883 ± 0.0391 | 0.8070 ± 0.0288 | 0.8080 ± 0.0288 | 0.8049 ± 0.0294 |

## Results, per draw

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

Random rep-spread ≤ 0.005 typically.

## Results, top1-only inference (test split, `retriever.k=1`, `ablation_mode=none`)

| Fraction | real docs (k=4) | top1 (k=1) | Δ |
| --- | --- | --- | --- |
| 50% | 0.8801 | 0.8800 | −0.0001 |
| 20% | 0.8541 | 0.8539 | −0.0002 |
| 10% | 0.8271 | 0.8273 | +0.0002 |
| 5% | 0.8070 | 0.8067 | −0.0003 |

## Statistics (paired over the 5 shared draws; paired t, df=4, two-sided)

Critical t for df=4: |t| > 2.776 for p<0.05, |t| > 2.132 for p<0.10.

| Fraction | Comparison | mean Δ | t | draws won | p |
| --- | --- | --- | --- | --- | --- |
| 50% | real vs patient_only | −0.0023 | −0.72 | 1/5 | n.s. |
| 50% | real vs random | −0.0007 | −0.54 | 2/5 | n.s. |
| 50% | real vs null | −0.0017 | −1.46 | 1/5 | n.s. |
| 20% | real vs patient_only | +0.0062 | 2.17 | 4/5 | ~0.09 |
| 20% | real vs random | −0.0015 | −0.91 | 2/5 | n.s. |
| 20% | real vs null | −0.0010 | −0.98 | 2/5 | n.s. |
| 10% | real vs patient_only | +0.0046 | 1.60 | 4/5 | n.s. |
| 10% | real vs random | −0.0016 | −2.85 | 0/5 | <0.05 |
| 10% | real vs null | −0.0086 | −2.26 | 0/5 | ~0.08 |
| 5% | real vs patient_only | +0.0187 | 2.37 | 4/5 | ~0.08 |
| 5% | real vs random | +0.0010 | 0.91 | 3/5 | n.s. |
| 5% | real vs null | −0.0022 | −1.13 | 2/5 | n.s. |

## Caveats

- No published test table for data-scarcity to reproduce exactly (gate is
  directional consistency with published val, see reproduction-gate table
  above).
- `random_docs` draws are unseeded; reported random numbers are means of 3
  independent eval reps.
- Only `learned-linear` tested (only variant trained under data scarcity).
- Checkpoints, labels, and retrieval corpus are hs3627's original working
  copy's, referenced read-only.
