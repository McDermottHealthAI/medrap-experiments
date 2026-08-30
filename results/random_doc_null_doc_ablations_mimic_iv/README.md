# Random-doc and null-doc ablations on the EXACT MIMIC capacity-starved checkpoints

Status: complete. 85 jobs (30 reproduction, 40 frozen-random, 10
frozen-null, 5 smokes), run against the exact original checkpoints behind
[`results/capacity_starved_retrieval/`](../capacity_starved_retrieval/README.md).

Scripts: [`mimic_iv_null_random_doc_ablations/scripts/`](../../mimic_iv_null_random_doc_ablations/README.md)
-- mechanical clones of the original experiment's eval scripts with
ablation flags/corpus swaps, pointed at the original
checkpoints/labels/corpus (`/groups/mm6677_gp/hs3627/medrap-experiments/mimic_iv_sweep/`,
read-only). Regenerable via `cd mimic_iv_null_random_doc_ablations && .venv/bin/python scripts/aggregate_results.py`.

## Ablation design

| Ablation | What changes | What stays |
| --- | --- | --- |
| frozen random | the 4 retrieved docs are replaced by 4 uniform random draws from the full 125k-passage corpus, redrawn every batch (`retriever.ablation_mode=random_docs`), 3 reps per checkpoint | trained weights; the model's own softmax-weighting over whatever docs arrive |
| frozen null | corpus swapped for 16 identical content-free docs (all-zero 1024-d keys, one pad token of content, deterministic) | trained weights |

Both score `checkpoints/last.ckpt` on the MEDS `held_out` split.

## Provenance

Tensorized MIMIC cohort restored from `/groups/.snapshots/@GMT-2026.08.25-12.06.07/`
into `/groups/mm6677_gp/zzw2102/data/MIMIC_MEDS_restored/processed`
(292/37/37 train/tuning/held_out shards, 11 GB, original timestamps
preserved; original at `/groups/mm6677_gp/data/MIMIC_MEDS/MEDS_cohort/`
was deleted ~Aug 25).

## Reproduction gate

- Test table: 15/15 exact (4-decimal match).
- Val table (200-batch protocol): 14/15 exact, 1 within 0.0012.
- Published val numbers were computed under `limit_val_batches=200`
  (inherited from `lightning_wandb` trainer config), a 6,400-row prefix of
  the tuning split, not the full split. Test table (full `held_out`) is
  unaffected.

## Key table

Test AUROC (`held_out`), mean ± population sd over 5 draws.

| variant | patient_only | real docs | random docs | null docs |
| --- | --- | --- | --- | --- |
| learned_linear | 0.8380 ± 0.0302 | 0.8477 ± 0.0287 | 0.8126 ± 0.0415 | 0.7859 ± 0.0376 |
| qwen3_text | 0.8380 ± 0.0302 | 0.8574 ± 0.0260 | 0.8473 ± 0.0264 | 0.8087 ± 0.0721 |

## Results, per draw

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

Random rep-spread: ≤ 0.015 (learned-linear draw 1), ≤ 0.007 elsewhere.

## Statistics (paired over the 5 shared draws, paired t-test, df=4, two-sided)

| Comparison | mean Δ | draws won | p |
| --- | --- | --- | --- |
| learned_linear real vs patient_only | +0.0097 | 4/5 | 0.080 |
| qwen3_text real vs patient_only | +0.0194 | 5/5 | 0.004 |
| real vs random / real vs null | −0.035 to −0.062 | 10/10 cells each | -- |

random cells: 10/10 negative. null cells: 10/10 negative, null ≤ random in
9/10.

## Checkpoint-protocol note

With this line's trainer config (`ModelCheckpoint(monitor=val/loss, save_last=True)`,
this Lightning version), `last.ckpt` is only rewritten when the monitored
metric improves -- `last.ckpt` is byte-identical to the best-val-loss
checkpoint everywhere in this line. On the original 3-epoch MIMIC runs, val
loss was still improving at the end of training, so best-val-loss ==
final epoch (checksum-verified: `last.ckpt` == `epoch=2-step=14361.ckpt`).

## Peak-to-peak check (val split, original training logs)

| | last-to-last (published) | peak-to-peak |
| --- | --- | --- |
| Δ learned-linear (val) | +0.0125, 5/5 | +0.0152, 5/5 |
| Δ qwen3_text (val) | +0.0248, 5/5 | +0.0246, 5/5 |

## Caveats

- Train-time random_docs control (retraining with random documents, not
  just evaluating with them) was not run.
- Frozen-random preserves the model's own softmax-weighting over the
  (random) candidates; the null arm forces uniform weights.
- `random_docs` draws are unseeded; reported random numbers are means of 3
  independent eval reps.
- Checkpoints, labels, and retrieval corpus are the original working
  copy's, referenced read-only. W&B runs at
  `wandb.ai/zzwang28-columbia-university/medrap` (`mimic-repro-*` /
  `mimic-abl-*`).
