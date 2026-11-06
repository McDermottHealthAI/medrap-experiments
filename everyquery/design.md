# EveryQuery with retrieval: design

Retrieval-augmented [EveryQuery](https://github.com/payalchandak/EveryQuery) (EQ) on MIMIC-IV, with
the medical-textbook corpus. This replaces the next round of [`scaling_laws/`](../scaling_laws/design.md).

Status: design for review. Data preparation, the patient-only stage and the offline document
encoding are runnable now ([`README.md`](README.md)); the retrieval model (one token per passage,
section 2) is next.

## 1. EveryQuery in brief

EQ trains one model that answers questions of the form

> Given the record up to time *t*, will code *c* occur inside the window (*s*, *e*)?

for **every code in the vocabulary at once**.

### Training examples

A **context** is a (patient, prediction time). For each context the label generator draws a fixed
sequence of 5 **windows**. A window opens at the prediction time, after a delay, or at the next
occurrence of a start code, and closes after a number of days or at the next occurrence of a bound
code. Each window is labelled for every code in the vocabulary: did it occur in the window (1) or
not (0).

| Item | Default |
|---|---|
| Training contexts | 5,000,000 sampled across the training split |
| Windows per context | 5 |
| Window length | 1 to 1,826 days, log-uniform |
| Labels per window | one bit per vocabulary code (`V` bits) |

### The model

One causal Llama backbone reads a single stream:

```
[ patient events (≤ 256) | W0 | C0 A0 | W1 | C1 A1 | W2 | C2 A2 | W3 | C3 A3 | W4 ]
```

- **Patient events**: the last ≤ 256 codes of the record before the prediction time; positions are
  elapsed time (rotary), not token order.
- **W*i***: one token describing window *i* (start rule, end rule).
- **C*i* A*i***: a code from window *i* and its true answer, teacher-forced, so later windows are
  conditioned on what happened earlier ("given the record did not end within 30 days, …").
- **Readout**: the hidden state at each W token is multiplied by the input-embedding table, giving
  one logit per code. The same table embeds codes and scores them.

| Shape | Default |
|---|---|
| Stream | B × (S + 13) tokens, S ≤ 256 |
| Backbone | 12 layers, hidden 384, 6 heads, feed-forward 1,536 |
| Window hidden states | B × 5 × 384 |
| Logits | B × 5 × V |
| Loss | binary cross-entropy over every (window, code), padding codes excluded |

### Evaluation

A separate grid fixes **1,000 query specifications** (query code, window, up to 2 conditioning
queries) and labels them at one prediction time per held-out patient. The model scores the last
query of each row. The evaluator reports one AUROC per specification with a 1,000-resample
bootstrap. We additionally compute **test loss** per row from the predictions (section 7).

### Commands

| Step | Command | Output |
|---|---|---|
| Preprocess raw MEDS | `EQ_process_data` | event shards + tensorized cohort |
| Training labels | `EQ_generate_multitask_sequences` (train, tuning) | contexts + packed label bits |
| Evaluation grid | `EQ_generate_evaluation_query_sequences` (held_out) | 1,000 specs × patients |
| Train | `EQ_train --config-name=conditional_multitask_ar_config` | checkpoints |
| Predict | `EQ_predict_multitask` | one probability per grid row |
| Evaluate | `EQ_evaluate_multitask` | AUROC per spec |

## 2. What retrieval adds

```
patient events ──► backbone ──► h_last (hidden state at the last patient event)
                                  │
                                  ▼ query q: linear 384 → 1024
                     FAISS top-K over textbook keys, scores s_k = q · key_k / √1024
                                  │
                                  ▼ passage k: token_k = linear(key_k) + s_k · u        (1024 → 384)
[ patient events | K passage tokens | W0 C0 A0 … W4 ]  ──► EveryQuery loss, unchanged
```

The simplest version: retrieval from the patient embedding before any query, one token per
passage, and EveryQuery's own loss.

1. The backbone runs over the patient events alone; the hidden state at the last event, `h_last`,
   becomes the query through one linear layer.
2. FAISS returns the top K passages by inner product with their keys.
3. Each passage becomes one token: its key embedding (the passage's 1,024-d Qwen3-Embedding
   vector, already in the database) projected to 384, plus `s_k · u`.
4. The backbone runs over the full stream; the window tokens attend to the patient events and to
   every passage token. Readout and loss are those of #1.

### Training the query

Nothing flows back through the FAISS lookup, so each passage's score is added to its token,
`s_k · u` with `u` a learned 384-d vector. If a passage helps, the gradient raises its score, and
that trains the query. Without this term the query would stay at its random initialisation.

| Piece | Shape | Trained |
|---|---|---|
| Passage keys | 125,847 × 1,024, inner-product index | no |
| Query projection | 384 → 1,024 | yes |
| Key projection | 1,024 → 384 | yes |
| Score vector `u` | 384 | yes |

About 0.8M new parameters. No null document: when no passage helps, the window tokens can attend
to the patient events instead. Passage tokens have their own token type and the prediction time as
position.

### Accepted trade-offs

- **Two passes per batch** (patient alone for the query, then the full stream): about 2× the cost
  of #1 per step. Reusing the first pass's keys and values can come later.
- **One vector per passage** is a thin description. If #2 does not beat #1, a richer encoding is
  next: `encode_docs.py` has already stored 16 contextual Qwen3 vectors per passage.
- **Retrieval starts random**, until the score term has trained the query.
- **FAISS on L40S only**: the GPU wheels have no H100 kernels, and CPU search on every step would
  add hours per run.

### Cost

| K | Stream length | Cost vs patient-only (269 tokens) |
|---|---|---|
| 8 | 277 | ≈ 2× (two passes) |
| 128 | 397 | ≈ 2.5× |
| 512 | 781 | ≈ 4× (attention ≈ 8×) |

## 3. Configurations

| # | Configuration | Windows attend to | Query |
|---|---|---|---|
| 1 | Patient only | patient events | – |
| 2 | Retrieval + patient | patient events + K passage tokens | learned from `h_last` |

Retrieval-only and random-document configurations come later.

## 4. Shapes, end to end (#2)

| Step | Shape |
|---|---|
| Patient events | B × S |
| Backbone over the patient events | B × S × 384 |
| Query from the last patient event | B × 1,024 |
| Retrieve | B × K passage ids, keys and scores |
| Passage tokens | B × K × 384 |
| Full stream | B × (S + K + 13) × 384 |
| Window hidden states | B × 5 × 384 |
| Logits | B × 5 × V |
| Loss | EveryQuery's, as in #1 |

## 5. Why passages in context

The passages are not combined by a separate objective; attention reads them all. This is the
smallest change to EveryQuery: EveryQuery's loss and only K to set. If attention does not learn to
use them, the next variants are a richer encoding per passage (several vectors each) and an exact
marginal over passages per context, `log Σ_k w_k p(future | passage k)`.

## 6. Stages and run counts (3 draws each)

A draw is one training-label sample (seed = draw) and training seed = draw. The tuning labels and
the evaluation grid are shared by every run, so all runs are paired on the same test rows.

| Stage | What | Runs | Status |
|---|---|---|---|
| 0 | Patient-only configuration: 3 sizes × 2 learning rates | 18 | runnable |
| 1 | K sweep at the Stage 0 configuration: #2 at K ∈ {8, 32, 128, 512} (#1 from Stage 0) | 12 | after #2 is implemented |
| 2 | Scaling around the chosen configuration: 3 sizes × 3 training-set sizes, #1 and #2 | 54 | after Stage 1 |
| 3 | PubMed | ~10 | later |

Stage 0 sizes:

| Name | Layers | Hidden | Heads | Feed-forward |
|---|---|---|---|---|
| small | 4 | 256 | 4 | 1,024 |
| base (EQ default) | 12 | 384 | 6 | 1,536 |
| large | 12 | 512 | 8 | 2,048 |

Learning rates 2e-4 (EQ default) and 5e-4. One epoch over 5M contexts, batch 96, bf16; validation
every 5% of the epoch on 100 batches; the best checkpoint by tuning loss is tested. A pilot (base,
2e-4, draw 1) times the run before the rest is submitted.

## 7. Measurement and analysis

- **Test loss per row**: binary cross-entropy of each grid row's probability against its label,
  averaged over rows. AUROC per specification is kept from `EQ_evaluate_multitask` but scaling fits
  use loss only.
- **Repeated evaluations**: per run, 1,000 bootstrap resamples of the evaluation rows with fixed
  resample ids, shared across runs (`test_set_id`). Checkpoints are kept, so this can be redone.
- **Paired analysis**: for each (setting, draw), the difference #1 − #2 on the same rows and
  resamples is one number; statistics are over those differences.
- **Laws**: simple-scaling-laws, fitted on loss; separable-power plus the other forms it offers,
  compared by its diagnostics; a fit flagged `lack_of_fit` or `ill_conditioned_fit`, or with low
  R², is reported but not used.
- **Compute and parameters**: one training step under `torch.utils.flop_counter.FlopCounterMode`
  gives FLOPs per step (forward + backward, including the patient pass used for the query); times steps run. Wall time per
  run from the trainer. Parameters reported as trainable and frozen (backbone, embeddings,
  projections; passage keys frozen). FAISS search counted
  separately.

## 8. Environment

EQ requires polars ≥ 1.35, meds-torch-data 0.9 and meds-transforms ≥ 0.6.7; MedRAP pins older
versions, so the two cannot share an environment. This directory has its own environment
([`pyproject.toml`](pyproject.toml): EQ pinned to a commit, FAISS, datasets) and reads the textbook
retrieval database (`scaling_laws/data/retrieval_db`) directly with `datasets` and FAISS, without
importing MedRAP.

## 9. Open decisions

1. K values for Stage 1 (8, 32, 128, 512 above).
2. Training-set sizes for Stage 2 (subsets of patients, shared across configurations).
