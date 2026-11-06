# EveryQuery with retrieval: design

Retrieval-augmented [EveryQuery](https://github.com/payalchandak/EveryQuery) (EQ) on MIMIC-IV, with
the medical-textbook corpus. This replaces the next round of [`scaling_laws/`](../scaling_laws/design.md).

Status: design for review. Data preparation, the patient-only stage and the offline document
encoding are runnable now ([`README.md`](README.md)); the retrieval model (documents in context,
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
                                  ▼ document k: M memory vectors (precomputed, 1024-d) → linear → 384, + s_k · u
[ patient events | doc 1 … doc K memory (K·M tokens) | W0 C0 A0 … W4 ]  ──► EveryQuery loss, unchanged
```

Retrieval from the patient embedding before any query (option (a)). The backbone is causal, so
`h_last` does not depend on what follows it: the patient events are run once, their keys and values
are reused for the rest of the stream.

All K documents go into the stream at once, ordered by score. Every window token attends to the
patient events and to every document; attention decides which documents matter. There is no
combination step and no new loss: #2 is trained and evaluated exactly like #1.

### Training the query

Nothing flows back through the FAISS lookup itself, so each document's score is added to its memory
tokens, `s_k · u` with `u` a learned 384-d vector. If a document helps, the gradient raises its score,
and that trains the query (and the scores order the documents).

### Document encoding

Each passage is read once, offline, by the full Qwen3-Embedding-0.6B model (not only its token
table), and its 256 contextual token states are pooled into 16 memory vectors (mean over 16
consecutive tokens). Stored as `float16`: 125,848 × 16 × 1,024 ≈ 4.1 GB. This replaces the
context-free token embeddings of `scaling_laws/`, where a document was a bag of word pieces with no
order. A run that uses M < 16 averages them down to M when loading (M ∈ {1, 2, 4, 8, 16}), so one
encoding serves every setting. [`encode_docs.py`](encode_docs.py) builds it.

| Piece | Shape | Trained |
|---|---|---|
| Document keys (retrieval) | 125,847 × 1,024, inner-product index | no |
| Document memory | 125,848 × 16 × 1,024 | no |
| Query projection | 384 → 1,024 | yes |
| Memory projection | 1,024 → 384 | yes |
| Score vector `u` | 384 | yes |

No null document: when no passage helps, the window tokens can attend to the patient events instead.

### Injection

Memory tokens sit between the patient events and W0, with their own token type and the prediction
time as position. Compared with `scaling_laws/`:

- no fusion block between the patient and the head: the patient keeps its direct path;
- no unmasked average over padding;
- the added parameters are two projections and `u` (~0.8M) instead of a fixed ~1.85M fusion, and
  the backbone's own attention does the reading.

### Cost

Cost is set by the number of memory tokens, K × M.

| K | M | Memory tokens | Stream length | Cost vs patient-only (269 tokens) |
|---|---|---|---|---|
| – | – | 0 | 269 | 1× |
| 32 | 16 | 512 | 781 | ~3× (attention ~8×) |
| 128 | 4 | 512 | 781 | same |
| 512 | 1 | 512 | 781 | same |
| 128 | 16 | 2,048 | 2,317 | ~9× (attention ~70×) |

## 3. Configurations

| # | Configuration | Windows attend to | Query |
|---|---|---|---|
| 1 | Patient only | patient events | – |
| 2 | Retrieval + patient | patient events + document memory | learned from `h_last` |

Retrieval-only and random-document configurations come later.

## 4. Shapes, end to end (#2)

| Step | Shape |
|---|---|
| Patient events | B × S |
| Backbone over the patient events | B × S × 384 (keys and values reused below) |
| Query from the last patient event | B × 1,024 |
| Retrieve | B × K document ids and scores |
| Memory tokens | B × (K·M) × 384 |
| Rest of the stream: memory + window tokens | B × (K·M + 13) tokens attending to the patient events |
| Window hidden states | B × 5 × 384 |
| Logits | B × 5 × V |
| Loss | EveryQuery's, as in #1 |

## 5. Why documents in context

The documents are not combined by a separate objective; attention reads them all. This is the
smallest change to EveryQuery: one forward pass, EveryQuery's loss, and only K and M to set. It gets
"more documents" through attention, not through a posterior over documents. If attention does not
learn to use them, the next variant is an exact marginal over documents per context,
`log Σ_k w_k p(future | doc k)`, which gives that posterior without new hyperparameters.

## 6. Stages and run counts (3 draws each)

A draw is one training-label sample (seed = draw) and training seed = draw. The tuning labels and
the evaluation grid are shared by every run, so all runs are paired on the same test rows.

| Stage | What | Runs | Status |
|---|---|---|---|
| 0 | Patient-only configuration: 3 sizes × 2 learning rates | 18 | runnable |
| 1 | Documents at the Stage 0 configuration: #2 at (K, M) ∈ {(8, 16), (32, 16), (128, 4), (512, 1)} (#1 from Stage 0) | 12 | after #2 is implemented |
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
  projections; Qwen3 and the memory table frozen). Document encoding and FAISS search counted
  separately.

## 8. Environment

EQ requires polars ≥ 1.35, meds-torch-data 0.9 and meds-transforms ≥ 0.6.7; MedRAP pins older
versions, so the two cannot share an environment. This directory has its own environment
([`pyproject.toml`](pyproject.toml): EQ pinned to a commit, FAISS, datasets) and reads the textbook
retrieval database (`scaling_laws/data/retrieval_db`) directly with `datasets` and FAISS, without
importing MedRAP.

## 9. Open decisions

1. Memory-token budget K × M for Stage 1 (512 above).
2. Training-set sizes for Stage 2 (subsets of patients, shared across configurations).
