# mimic_iv_sweep_frequent

Compares two architectures on task labels selected by
**most-frequent code** (highest distinct-subject count in the train
split), across N ∈ {1, 2, 4, 8, 16, 32, 64, 128} simultaneous binary
tasks:

- **`patient_only`** — no retrieval, predicts directly from the
  patient's own EHR sequence.
- **`marginalized` (binary mode)** — retrieval-augmented, predicts per
  retrieved document and marginalizes independently per task
  (`marginalized_output_mode=binary`,
  [MedRAP PR #93](https://github.com/McDermottHealthAI/MedRAP/pull/93)).

Requires `medrap` built from
[`McDermottHealthAI/MedRAP@164c2ef`](https://github.com/McDermottHealthAI/MedRAP/commit/164c2ef3290c13090e647d2f73a0b058ccbd38a5)
(merges `main` -- which now has
[PR #93](https://github.com/McDermottHealthAI/MedRAP/pull/93)
`marginalized_output_mode`,
[PR #94](https://github.com/McDermottHealthAI/MedRAP/pull/94) test-split
AUROC, and
[PR #95](https://github.com/McDermottHealthAI/MedRAP/pull/95) eval-config
fixes -- with the still-unmerged `feat/task-gen-most-frequent-codes`
(`code_selection`),
[PR #96](https://github.com/McDermottHealthAI/MedRAP/pull/96)
(`duration_distribution`, random per-task occurrence windows), and
[PR #97](https://github.com/McDermottHealthAI/MedRAP/pull/97)
(`do_overwrite` support for `medrap-eval`)), pinned in `pyproject.toml`.

## Run instructions

```bash
cd mimic_iv_sweep_frequent
uv sync
sbatch ../mimic_iv_sweep/scripts/prepare_retrieval.sh   # once, if data/retrieval_db doesn't exist yet
sbatch scripts/generate_labels_n1248_frequent.sh        # task labels, N=1..128
sbatch scripts/sweep_patient_only_n1248.sh              # patient_only, N=1..128
sbatch scripts/sweep_marginalized_binary_n1248.sh       # marginalized (binary), N=1..128
sbatch scripts/eval_marginalized_binary_top1_n1248.sh   # held-out test AUROC, top-1 doc only (after the above finishes)
```

Results land in W&B (`medrap` project) as `patient-only-frequent-n{N}-*`,
`marginalized-binary-frequent-n{N}-*`, and (for the held-out top-1-doc eval)
`marginalized-binary-top1-test-frequent-n{N}-*`. See
[Results](#held-out-test-auroc-top-retrieved-document-only) below.

Three further variants of `sweep_marginalized_binary_n1248.sh`, each
isolating one change against that same baseline -- see
[Further variants](#further-variants-longer-training-more-retrieved-docs-random-task-durations)
below for what each tests and why:

```bash
sbatch scripts/sweep_marginalized_binary_longer_n1248.sh     # max_epochs=6 (base uses 3)
sbatch scripts/sweep_marginalized_binary_k10_n1248.sh         # retriever.k=10 (base uses 4)
sbatch scripts/generate_labels_duration_n1248_frequent.sh     # random per-task durations (base: fixed 7-day)
sbatch scripts/sweep_marginalized_binary_duration_n1248.sh    # trains on those duration-labels
```

## Architecture & hyperparameters

Both arms share: RoPE patient encoder, 3 training epochs, `lr=1e-3`,
`warmup_steps=200`, batch size 32, `max_seq_len=256`,
`seq_sampling_strategy=to_end`, `gradient_clip_val=1.0`.

| Stage | `patient_only` | `marginalized` (binary) |
| --- | --- | --- |
| Encoder | `TimeDeltaRoPEPatientEncoder` — vocab 65536, embed dim 128, 4 heads, 2 layers, ff dim 256 | same |
| Query projector | unused (`fusion=passthrough` discards it) | `SequenceMeanQueryProjector` — in 128, out 1024 |
| Retriever | none | `hf_dataset` (FAISS) — `k=4`, corpus = `MedRAG/textbooks` |
| Retrieval encoder | none | `TokenFeatureRetrievalEncoder` — vocab 151936, embed dim 64 |
| Fusion | `PassthroughFusion` (no retrieval) | `PerDocCrossAttentionFusion` — d_model 256, 8 heads, ff dim 512, 2 layers, dropout 0.1 |
| Head | `LinearHead`, in 128, out N | `LinearHead`, in 256, out N |
| Loss | `MultiTaskBCELoss` | `MultiTaskBCEMarginalizedLoss` (marginalizes per-task sigmoid over the 4 retrieved docs) |

## Results

**What "valid task" means:** a sampled task counts as valid only if its
held-out validation split contains at least one positive *and* one
negative example — AUROC is undefined otherwise, so those tasks are
excluded from the mean rather than scored as 0.

### Average AUROC

| N | patient_only | marginalized (binary) | Δ (binary − patient_only) | valid tasks |
| --- | --- | --- | --- | --- |
| 1 | [0.9866](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/b252lcbs) | [0.9823](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/b084s53o) | -0.0043 | 1/1 |
| 2 | [0.8894](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/lsgx9cwn) | [0.8709](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/r73ezhcr) | -0.0185 | 2/2 |
| 4 | [0.9808](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/uglkrb02) | [0.9457](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/pf0g1pop) | -0.0351 | 4/4 |
| 8 | [0.8873](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/kesr4bob) | [0.8630](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/fruenovy) | -0.0243 | 8/8 |
| 16 | [0.9107](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/rcg52tcf) | [0.7198](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/nppi07vm) | -0.1909 | 16/16 |
| 32 | [0.9855](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/or2dgh80) | [0.9867](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/0gwrqwfn) | +0.0012 | 28/32 |
| 64 | [0.8987](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/7am95jcz) | [0.9233](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/3xhox0y2) | +0.0246 | 63/64 |
| 128 | [0.9454](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/k0socaoo) | [0.9310](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/2rrgfx15) | -0.0144 | 124/128 |

Values link to the W&B run. `val/auroc/mean`. "valid tasks" is
identical between the two architectures at a given N since both train
on the same label file (task validity depends only on the labels, not
the model).

### Per-task win rate

Of the valid tasks at each N, the fraction where `marginalized (binary)`
beat `patient_only` on that same task:

| N | marginalized (binary) wins | valid tasks | win rate |
| --- | --- | --- | --- |
| 1 | 0 | 1 | 0.0% |
| 2 | 0 | 2 | 0.0% |
| 4 | 2 | 4 | 50.0% |
| 8 | 3 | 8 | 37.5% |
| 16 | 5 | 16 | 31.2% |
| 32 | 12 | 28 | 42.9% |
| 64 | 38 | 63 | 60.3% |
| 128 | 16 | 124 | 12.9% |

No clear trend with N. `patient_only` wins on average AUROC at 6 of 8
N values (only loses at N=32 and N=64); at the per-task level
`marginalized (binary)`'s win rate ranges from 0% (N=1, N=2) to 60.3%
(N=64). N=16 is the largest gap (0.9107 vs. 0.7198, an 11/16-task
deficit for marginalized) and worth a closer look.

### Held-out test AUROC, top-retrieved-document only

Everything above is `val/auroc/mean`, computed on the MEDS **tuning**
split (Lightning's `val_dataloader`), and marginalizes over all `k=4`
retrieved documents -- identically at train and eval time (no
train/inference discrepancy in how documents are combined; see
`RetrievalAugmentedModel.forward()`, which has no train/eval branching).

`eval_marginalized_binary_top1_n1248.sh` answers a different question:
how does each trained `marginalized (binary)` checkpoint perform on the
true **held-out** split, using only the single top-retrieved document
(`retriever.k=1`) instead of the 4 it was trained with? This is possible
without retraining because `PerDocCrossAttentionFusion` and the binary
marginalization are both K-agnostic -- no weight is sized by `K`, and
marginalizing over `K=1` is mathematically a no-op (the "marginal"
prediction becomes exactly that one document's prediction). Requires
[MedRAP PR #94](https://github.com/McDermottHealthAI/MedRAP/pull/94)
(`test/auroc/*` logging -- without it, `medrap-eval eval_mode=test` only
reports loss/accuracy on `held_out`, no AUROC) and
[MedRAP PR #95](https://github.com/McDermottHealthAI/MedRAP/pull/95)
(`_eval.yaml` didn't declare `marginalized_retrieval`/
`marginalized_output_mode`/`wandb_*`, so neither the checkpoint's
architecture nor W&B logging could be configured for eval before this).

Results land in W&B as `marginalized-binary-top1-test-frequent-n{N}-*`.

| N | test AUROC (top-1 doc) | val AUROC (tuning, k=4, same checkpoint) | valid tasks |
| --- | --- | --- | --- |
| 1 | [0.9224](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/2cgfj565) | 0.9823 | 1/1 |
| 2 | [0.9011](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/icwwnup9) | 0.8709 | 2/2 |
| 4 | [0.8684](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/ok058nl8) | 0.9456 | 4/4 |
| 8 | [0.9191](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/96b79rrc) | 0.8630 | 8/8 |
| 16 | [0.8936](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/kvrgrmjw) | 0.7197 | 16/16 |
| 32 | [0.9384](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/k5wtl7yi) | 0.9866 | 32/32 |
| 64 | [0.9139](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/4vuskccw) | 0.9233 | 64/64 |
| 128 | [0.9323](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/irw8l4j9) | 0.9310 | 128/128 |

Unlike the `val/auroc` numbers throughout the rest of this README, every
task is valid here at every N (`valid tasks` = `N` across the board) --
`held_out` is a different (and here, apparently more balanced) split than
`tuning`. Top-1-doc held-out AUROC tracks the full-k=4 tuning-split AUROC
reasonably closely and even exceeds it at N=2, 8, and 16 (N=16 most
notably: 0.72 -> 0.89) -- the model doesn't appear to depend heavily on
marginalizing across multiple retrieved documents for its discrimination;
much of the signal seems to already be present in the single
best-retrieved document. N=4 is the one case where top-1 clearly
underperforms (0.87 vs. 0.95).

## Further variants: longer training, more retrieved docs, random task durations

Three more scripts, each changing exactly one thing against the
`sweep_marginalized_binary_n1248.sh` baseline above (same task labels
otherwise, same architecture otherwise) -- run and check each in isolation
before combining any of them.

### Longer training (`sweep_marginalized_binary_longer_n1248.sh`)

The baseline sweep (`max_epochs=3`) finished in ~1.5h wall-clock per run.
This variant only changes `training.trainer.max_epochs=6`, targeting
roughly double that (~3h), to check whether the model is still improving
past 3 epochs or has already converged. Results land in W&B as
`marginalized-binary-longer-frequent-n{N}-*`; checkpoints at
`outputs/marginalized_binary_longer_n1248/n{N}/`.

```bash
sbatch scripts/sweep_marginalized_binary_longer_n1248.sh
```

**Results:** all 8/8 finished.

| N | baseline (3 epochs) | longer (6 epochs) | Δ |
| --- | --- | --- | --- |
| 1 | [0.9823](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/b084s53o) | [0.9826](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/5kr42czb) | +0.0003 |
| 2 | [0.8709](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/r73ezhcr) | [0.8715](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/npld0h2z) | +0.0006 |
| 4 | [0.9457](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/pf0g1pop) | [0.9780](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/wjewdf3m) | +0.0323 |
| 8 | [0.8630](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/fruenovy) | [0.8873](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/i9s99zug) | +0.0243 |
| 16 | [0.7198](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/nppi07vm) | [0.9342](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/h8lhsk2o) | +0.2145 |
| 32 | [0.9867](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/0gwrqwfn) | [0.9831](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/hzb7c5uh) | -0.0035 |
| 64 | [0.9233](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/3xhox0y2) | [0.9055](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/mcgqay1f) | -0.0179 |
| 128 | [0.9310](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/2rrgfx15) | [0.9258](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/8sjg8hh7) | -0.0052 |

Longer training wins or ties at N<=16 (N=16's 0.72 -> 0.93 is the single
biggest gain anywhere in this experiment -- looks like undertraining, not
an architectural limit), and is roughly flat-to-slightly-worse at N>=32.

### More retrieved documents at training time (`sweep_marginalized_binary_k10_n1248.sh`)

The baseline sweep retrieves `k=4` documents per prediction, both for the
marginalized loss and the reported logits/AUROC (same `k` at train and
eval time throughout this sweep -- see
[`eval_marginalized_binary_top1_n1248.sh`](#held-out-test-auroc-top-retrieved-document-only)
above for the deliberately-*mismatched*-`k` ablation in the other
direction). This variant only changes `retriever.k=10`, to check whether
marginalizing over more retrieved documents helps. Bumped `--mem` (96G vs.
64G) since more documents per batch means more compute/memory per step,
most visible in `PerDocCrossAttentionFusion`'s effective `B*K` batch.
Results land in W&B as `marginalized-binary-k10-frequent-n{N}-*`;
checkpoints at `outputs/marginalized_binary_k10_n1248/n{N}/`.

```bash
sbatch scripts/sweep_marginalized_binary_k10_n1248.sh
```

**Results:** all 8/8 finished.

| N | baseline (k=4) | k10 (k=10) | Δ |
| --- | --- | --- | --- |
| 1 | [0.9823](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/b084s53o) | [0.9850](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/9hgm1e7k) | +0.0026 |
| 2 | [0.8709](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/r73ezhcr) | [0.8809](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/p4egsc8j) | +0.0100 |
| 4 | [0.9457](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/pf0g1pop) | [0.9797](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/5voogldq) | +0.0341 |
| 8 | [0.8630](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/fruenovy) | [0.8891](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/lo6nop8v) | +0.0261 |
| 16 | [0.7198](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/nppi07vm) | [0.8780](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/5rz1r9hq) | +0.1582 |
| 32 | [0.9867](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/0gwrqwfn) | [0.9831](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/es5xsbey) | -0.0036 |
| 64 | [0.9233](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/3xhox0y2) | [0.9119](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/wxkulvvs) | -0.0114 |
| 128 | [0.9310](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/2rrgfx15) | [0.9239](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/kzo63wb4) | -0.0071 |

Same pattern as longer training: k=10 wins clearly at N<=16 (again N=16
is the standout: 0.72 -> 0.88), roughly flat-to-slightly-worse at N>=32.
Whether this is "more documents help" or simply "the extra retrieval
compute happens to correlate with the same undertraining fix as more
epochs" isn't disentangled by this run alone.

### Random per-task occurrence-window durations (`generate_labels_duration_n1248_frequent.sh` + `sweep_marginalized_binary_duration_n1248.sh`)

Every label set in this README so far uses one fixed 7-day occurrence
window shared by every task (`duration_distribution=fixed`, the
`medrap-preprocess` default). This variant instead samples each task's own
occurrence-window duration independently, log-uniformly over `[1, 90]`
days (`duration_distribution=log-uniform`,
[MedRAP PR #96](https://github.com/McDermottHealthAI/MedRAP/pull/96)) --
same most-frequent code selection, same everything else. Ported from
[EveryQuery](https://github.com/payalchandak/EveryQuery)'s duration-sampling
formula (see PR #96 for why only the formula, not the package, was
adopted). Labels land in a *separate* directory
(`data/tasks_duration/n<N>/`, distinct from `data/tasks/n<N>/`) so this
never overwrites the fixed-duration labels every other script here reads.

```bash
sbatch scripts/generate_labels_duration_n1248_frequent.sh   # task labels, N=1..128
sbatch scripts/sweep_marginalized_binary_duration_n1248.sh  # training, N=1..128
```

Results land in W&B as `marginalized-binary-duration-frequent-n{N}-*`;
checkpoints at `outputs/marginalized_binary_duration_n1248/n{N}/`. Compare
against `marginalized-binary-frequent-n{N}-*` (the fixed-7-day baseline) to
see whether random durations change AUROC -- note the comparison isn't
perfectly apples-to-apples (different label files, so different sampled
positives/negatives per task, same as the frequent-vs-random-code-selection
comparison earlier in this experiment), but it's the same task-selection
strategy and architecture otherwise.

**Results:** all 8/8 finished.

| N | baseline (fixed 7d) | duration (log-uniform, 1-90d) | Δ | duration's valid tasks |
| --- | --- | --- | --- | --- |
| 1 | [0.9823](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/b084s53o) | [0.9576](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/6q5b2zgu) | -0.0247 | 1/1 |
| 2 | [0.8709](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/r73ezhcr) | [0.9596](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/mx5kpr8b) | +0.0887 | 2/2 |
| 4 | [0.9457](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/pf0g1pop) | [0.9277](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/p7xf22t7) | -0.0180 | 4/4 |
| 8 | [0.8630](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/fruenovy) | [0.9236](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/1qjrwp20) | +0.0606 | 7/8 |
| 16 | [0.7198](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/nppi07vm) | [0.7841](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/z38oywhv) | +0.0643 | 16/16 |
| 32 | [0.9867](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/0gwrqwfn) | [0.9414](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/cwfef8p4) | -0.0453 | 26/32 |
| 64 | [0.9233](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/3xhox0y2) | [0.9608](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/5owdril0) | +0.0375 | 54/64 |
| 128 | [0.9310](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/2rrgfx15) | [0.9434](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/b5drlxi7) | +0.0124 | 98/128 |

Mixed: duration wins at N=2, 8, 16, 64, 128 and loses at N=1, 4, 32, with
no clean pattern by N. Note random durations produce a real drop in valid
tasks at higher N (e.g. 98/128 vs. 124/128 for the fixed-window labels) --
some log-uniform-sampled durations land on windows too short (or, more
rarely, positioned such that the anchor's trailing room doesn't help) for
the code to show both classes in that split, so this comparison is
somewhat less apples-to-apples than `longer`/`k10` above (same labels,
architecture only) -- here both labels *and* the metric's task-count
denominator differ.

## Task codes used

The exact `num_tasks` codes selected by `code_selection=most_frequent`
at each N (from `data/tasks/n<N>/tasks/code_index.json`; same codes for
both architectures at a given N). Selection is nested — the top-K codes
at a given N are a subset of the top-(2K) codes at the next N, as
expected for a deterministic "most frequent" ranking:

<details><summary><code>N=1</code> (1 code)</summary>

```
0: LAB//50920//UNK
```

</details>

<details><summary><code>N=2</code> (2 codes)</summary>

```
0: LAB//50920//UNK
1: TRANSFER_TO//ED//Emergency Department
```

</details>

<details><summary><code>N=4</code> (4 codes)</summary>

```
0: LAB//50920//UNK
1: TRANSFER_TO//ED//Emergency Department
2: TRANSFER_TO//discharge//UNKNOWN
3: LAB//51478//mg/dL
```

</details>

<details><summary><code>N=8</code> (8 codes)</summary>

```
0: LAB//50920//UNK
1: TRANSFER_TO//ED//Emergency Department
2: TRANSFER_TO//discharge//UNKNOWN
3: LAB//51478//mg/dL
4: LAB//51514//mg/dL
5: MEDICATION//START//Sodium Chloride 0.9%  Flush
6: MEDICATION//STOP//Sodium Chloride 0.9%  Flush
7: LAB//51484//mg/dL
```

</details>

<details><summary><code>N=16</code> (16 codes)</summary>

```
0: LAB//50920//UNK
1: TRANSFER_TO//ED//Emergency Department
2: TRANSFER_TO//discharge//UNKNOWN
3: LAB//51478//mg/dL
4: LAB//51514//mg/dL
5: MEDICATION//START//Sodium Chloride 0.9%  Flush
6: MEDICATION//STOP//Sodium Chloride 0.9%  Flush
7: LAB//51484//mg/dL
8: ED_OUT
9: ED_REGISTRATION
10: LAB//51087//UNK
11: MEDICATION//START//Acetaminophen
12: MEDICATION//STOP//Acetaminophen
13: LAB//51492//mg/dL
14: MEDICATION//START//UNK
15: MEDICATION//STOP//UNK
```

</details>

<details><summary><code>N=32</code> (32 codes)</summary>

```
0: LAB//50920//UNK
1: TRANSFER_TO//ED//Emergency Department
2: TRANSFER_TO//discharge//UNKNOWN
3: LAB//51478//mg/dL
4: LAB//51514//mg/dL
5: MEDICATION//START//Sodium Chloride 0.9%  Flush
6: MEDICATION//STOP//Sodium Chloride 0.9%  Flush
7: LAB//51484//mg/dL
8: ED_OUT
9: ED_REGISTRATION
10: LAB//51087//UNK
11: MEDICATION//START//Acetaminophen
12: MEDICATION//STOP//Acetaminophen
13: LAB//51492//mg/dL
14: MEDICATION//START//UNK
15: MEDICATION//STOP//UNK
16: LAB//50933//UNK
17: Weight (Lbs)
18: Blood Pressure
19: LAB//50971//mEq/L//value_[4.1,4.3)
20: LAB//50902//mEq/L//value_[104.0,106.0)
21: BMI (kg/m2)
22: Height (Inches)
23: LAB//51466//UNK
24: LAB//51486//UNK
25: LAB//51487//UNK
26: LAB//51506//UNK
27: LAB//51508//UNK
28: LAB//51464//mg/dL
29: LAB//50887//UNK
30: MEDICATION//START//Heparin
31: MEDICATION//STOP//Heparin
```

</details>

<details><summary><code>N=64</code> (64 codes)</summary>

```
0: LAB//50920//UNK
1: TRANSFER_TO//ED//Emergency Department
2: TRANSFER_TO//discharge//UNKNOWN
3: LAB//51478//mg/dL
4: LAB//51514//mg/dL
5: MEDICATION//START//Sodium Chloride 0.9%  Flush
6: MEDICATION//STOP//Sodium Chloride 0.9%  Flush
7: LAB//51484//mg/dL
8: ED_OUT
9: ED_REGISTRATION
10: LAB//51087//UNK
11: MEDICATION//START//Acetaminophen
12: MEDICATION//STOP//Acetaminophen
13: LAB//51492//mg/dL
14: MEDICATION//START//UNK
15: MEDICATION//STOP//UNK
16: LAB//50933//UNK
17: Weight (Lbs)
18: Blood Pressure
19: LAB//50971//mEq/L//value_[4.1,4.3)
20: LAB//50902//mEq/L//value_[104.0,106.0)
21: BMI (kg/m2)
22: Height (Inches)
23: LAB//51466//UNK
24: LAB//51486//UNK
25: LAB//51487//UNK
26: LAB//51506//UNK
27: LAB//51508//UNK
28: LAB//51464//mg/dL
29: LAB//50887//UNK
30: MEDICATION//START//Heparin
31: MEDICATION//STOP//Heparin
32: LAB//51006//mg/dL//value_[13.0,16.0)
33: LAB//50955//UNK
34: LAB//50912//mg/dL//value_[0.9,1.1)
35: LAB//50868//mEq/L//value_[13.0,14.0)
36: LAB//50868//mEq/L//value_[14.0,15.0)
37: LAB//50868//mEq/L//value_[16.0,18.0)
38: MEDICATION//START//Docusate Sodium
39: MEDICATION//STOP//Docusate Sodium
40: LAB//50983//mEq/L//value_[139.0,140.0)
41: LAB//50983//mEq/L//value_[140.0,141.0)
42: LAB//51221//%//value_[37.2,39.4)
43: LAB//51301//K/uL//value_[7.6,8.6)
44: LAB//50971//mEq/L//value_[3.7,3.9)
45: LAB//51221//%//value_[39.4,42.1)
46: LAB//50947//UNK//value_[1.0,2.0)
47: LAB//51301//K/uL//value_[8.6,9.8)
48: LAB//51222//g/dL//value_[12.3,13.1)
49: LAB//50882//mEq/L//value_[25.0,26.0)
50: LAB//50868//mEq/L//value_[12.0,13.0)
51: MEDICATION//START//Senna
52: MEDICATION//STOP//Senna
53: LAB//50868//mEq/L//value_[15.0,16.0)
54: LAB//50971//mEq/L//value_[4.4,4.6)
55: LAB//50882//mEq/L//value_[26.0,27.0)
56: LAB//50983//mEq/L//value_[138.0,139.0)
57: LAB//51279//m/uL//value_[4.16,4.41)
58: LAB//50931//mg/dL//value_[103.0,111.0)
59: LAB//51279//m/uL//value_[4.41,4.74)
60: LAB//51301//K/uL//value_[6.8,7.6)
61: LAB//51222//g/dL//value_[13.1,14.0)
62: LAB//50882//mEq/L//value_[28.0,30.0)
63: LAB//50902//mEq/L//value_[103.0,104.0)
```

</details>

<details><summary><code>N=128</code> (128 codes)</summary>

```
0: LAB//50920//UNK
1: TRANSFER_TO//ED//Emergency Department
2: TRANSFER_TO//discharge//UNKNOWN
3: LAB//51478//mg/dL
4: LAB//51514//mg/dL
5: MEDICATION//START//Sodium Chloride 0.9%  Flush
6: MEDICATION//STOP//Sodium Chloride 0.9%  Flush
7: LAB//51484//mg/dL
8: ED_OUT
9: ED_REGISTRATION
10: LAB//51087//UNK
11: MEDICATION//START//Acetaminophen
12: MEDICATION//STOP//Acetaminophen
13: LAB//51492//mg/dL
14: MEDICATION//START//UNK
15: MEDICATION//STOP//UNK
16: LAB//50933//UNK
17: Weight (Lbs)
18: Blood Pressure
19: LAB//50971//mEq/L//value_[4.1,4.3)
20: LAB//50902//mEq/L//value_[104.0,106.0)
21: BMI (kg/m2)
22: Height (Inches)
23: LAB//51466//UNK
24: LAB//51486//UNK
25: LAB//51487//UNK
26: LAB//51506//UNK
27: LAB//51508//UNK
28: LAB//51464//mg/dL
29: LAB//50887//UNK
30: MEDICATION//START//Heparin
31: MEDICATION//STOP//Heparin
32: LAB//51006//mg/dL//value_[13.0,16.0)
33: LAB//50955//UNK
34: LAB//50912//mg/dL//value_[0.9,1.1)
35: LAB//50868//mEq/L//value_[13.0,14.0)
36: LAB//50868//mEq/L//value_[14.0,15.0)
37: LAB//50868//mEq/L//value_[16.0,18.0)
38: MEDICATION//START//Docusate Sodium
39: MEDICATION//STOP//Docusate Sodium
40: LAB//50983//mEq/L//value_[139.0,140.0)
41: LAB//50983//mEq/L//value_[140.0,141.0)
42: LAB//51221//%//value_[37.2,39.4)
43: LAB//51301//K/uL//value_[7.6,8.6)
44: LAB//50971//mEq/L//value_[3.7,3.9)
45: LAB//51221//%//value_[39.4,42.1)
46: LAB//50947//UNK//value_[1.0,2.0)
47: LAB//51301//K/uL//value_[8.6,9.8)
48: LAB//51222//g/dL//value_[12.3,13.1)
49: LAB//50882//mEq/L//value_[25.0,26.0)
50: LAB//50868//mEq/L//value_[12.0,13.0)
51: MEDICATION//START//Senna
52: MEDICATION//STOP//Senna
53: LAB//50868//mEq/L//value_[15.0,16.0)
54: LAB//50971//mEq/L//value_[4.4,4.6)
55: LAB//50882//mEq/L//value_[26.0,27.0)
56: LAB//50983//mEq/L//value_[138.0,139.0)
57: LAB//51279//m/uL//value_[4.16,4.41)
58: LAB//50931//mg/dL//value_[103.0,111.0)
59: LAB//51279//m/uL//value_[4.41,4.74)
60: LAB//51301//K/uL//value_[6.8,7.6)
61: LAB//51222//g/dL//value_[13.1,14.0)
62: LAB//50882//mEq/L//value_[28.0,30.0)
63: LAB//50902//mEq/L//value_[103.0,104.0)
64: LAB//50983//mEq/L//value_[141.0,142.0)
65: LAB//51221//%//value_[35.2,37.2)
66: LAB//51265//K/uL//value_[218.0,243.0)
67: LAB//51265//K/uL//value_[243.0,272.0)
68: LAB//50931//mg/dL//value_[83.0,91.0)
69: LAB//50931//mg/dL//value_[91.0,97.0)
70: LAB//50912//mg/dL//value_[0.8,0.9)
71: LAB//50902//mEq/L//value_[102.0,103.0)
72: MEDICATION//Acetaminophen//Administered
73: LAB//50882//mEq/L//value_[24.0,25.0)
74: LAB//51301//K/uL//value_[9.8,11.4)
75: LAB//51279//m/uL//value_[3.92,4.16)
76: LAB//50931//mg/dL//value_[97.0,103.0)
77: LAB//51265//K/uL//value_[193.0,218.0)
78: LAB//51301//K/uL//value_[6.0,6.8)
79: LAB//50971//mEq/L//value_[4.0,4.1)
80: LAB//50882//mEq/L//value_[27.0,28.0)
81: LAB//51222//g/dL//value_[11.6,12.3)
82: MEDICATION//Sodium Chloride 0.9%  Flush//Flushed
83: MEDICATION//UNK//Started
84: LAB//51519//UNK
85: LAB//51463//UNK
86: HOSPITAL_DISCHARGE//HOME
87: LAB//51006//mg/dL//value_[11.0,13.0)
88: LAB//51221//%//value_[33.1,35.2)
89: LAB//50960//mg/dL//value_[2.0,2.1)
90: LAB//51265//K/uL//value_[272.0,310.0)
91: LAB//50931//mg/dL//value_[111.0,120.0)
92: LAB//50902//mEq/L//value_[101.0,102.0)
93: LAB//51301//K/uL//value_[11.4,14.5)
94: LAB//51222//g/dL//value_[14.0,inf)
95: LAB//50902//mEq/L//value_[106.0,108.0)
96: LAB//51250//fL//value_[89.0,91.0)
97: LAB//50971//mEq/L//value_[3.9,4.0)
98: LAB//50912//mg/dL//value_[0.7,0.8)
99: LAB//50983//mEq/L//value_[135.0,137.0)
100: LAB//51277//%//value_[13.2,13.6)
101: LAB//51221//%//value_[42.1,inf)
102: LAB//51250//fL//value_[91.0,93.0)
103: LAB//51006//mg/dL//value_[16.0,18.0)
104: LAB//50931//mg/dL//value_[120.0,133.0)
105: LAB//51006//mg/dL//value_[18.0,21.0)
106: LAB//51277//%//value_[12.7,13.2)
107: LAB//50971//mEq/L//value_[4.6,4.9)
108: LAB//51279//m/uL//value_[3.68,3.92)
109: LAB//50983//mEq/L//value_[137.0,138.0)
110: LAB//50868//mEq/L//value_[11.0,12.0)
111: MEDICATION//START//Ondansetron
112: LAB//50960//mg/dL//value_[2.2,2.4)
113: MEDICATION//STOP//Ondansetron
114: LAB//50960//mg/dL//value_[2.1,2.2)
115: LAB//50882//mEq/L//value_[23.0,24.0)
116: LAB//50970//mg/dL//value_[3.4,3.7)
117: LAB//50971//mEq/L//value_[4.3,4.4)
118: LAB//51277//%//value_[13.6,14.1)
119: LAB//51222//g/dL//value_[10.9,11.6)
120: LAB//51237//UNK//value_[1.1,1.2)
121: LAB//50960//mg/dL//value_[1.9,2.0)
122: LAB//51265//K/uL//value_[167.0,193.0)
123: LAB//50902//mEq/L//value_[98.0,100.0)
124: LAB//51301//K/uL//value_[5.1,6.0)
125: LAB//50983//mEq/L//value_[142.0,143.0)
126: LAB//50983//mEq/L//value_[143.0,inf)
127: LAB//51491//units//value_[6.0,6.5)
```

</details>

