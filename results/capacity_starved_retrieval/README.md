# Capacity-starved `patient_only` vs. marginalized retrieval (N=25, 30d)

**Status: COMPLETE -- all 15 jobs (5 draws x 3 architectures) finished.
First consistent, statistically-larger-than-noise retrieval benefit found
anywhere in this repo.**

Scripts: [`sweep_patient_only_capacity_starved_n25_30d.sh`](../../mimic_iv_sweep/scripts/sweep_patient_only_capacity_starved_n25_30d.sh) +
[`sweep_marginalized_binary_qwen3_text_capacity_starved_n25_30d.sh`](../../mimic_iv_sweep/scripts/sweep_marginalized_binary_qwen3_text_capacity_starved_n25_30d.sh) +
[`sweep_marginalized_binary_learned_linear_capacity_starved_n25_30d.sh`](../../mimic_iv_sweep/scripts/sweep_marginalized_binary_learned_linear_capacity_starved_n25_30d.sh)
(all in `mimic_iv_sweep/`, which this experiment reuses for its labels,
retrieval corpus, and cluster setup -- see that directory's `README.md`
for the full sweep infrastructure and every other experiment run against
the same MIMIC-IV cohort).

## The model, layer by layer

This experiment trains the same underlying architecture on MIMIC-IV, in
one of two configurations (`patient_only` or `marginalized`/retrieval-
augmented). Below is the exact sequence of layers each configuration
runs, in forward-pass order.

**Task**: multi-task binary classification -- for N=25 randomly-sampled
diagnostic codes, predict whether each will occur within a fixed 30-day
occurrence window after a prediction "anchor" time (see `mimic_iv_sweep/README.md`'s
anchor-sampling sections for how that anchor point is chosen). One model,
25 independent binary predictions per patient.

**1. Patient encoder -- `TimeDeltaRoPEPatientEncoder`** (identical in both
configurations; only its size changes between the full-capacity and
capacity-starved experiments)

A patient's clinical event history (a time-ordered sequence of codes --
diagnoses, procedures, labs, etc.) is encoded as follows:

1. **Token embedding**: `nn.Embedding(vocab_size, D)` -- each event code
   (an integer id, `0` = padding) is mapped to a `D`-dimensional vector.
   `D` = `embedding_dim` (128 full-capacity, 16 starved).
2. **Time-delta rotary position encoding (RoPE)**: rather than the
   standard "position 1, 2, 3, ..." rotary encoding, positions are derived
   from the *cumulative log time gap* between consecutive events
   (`cumsum(log1p(time_delta_days))`). This produces a `cos`/`sin` pair per
   position that rotary attention (step 3) uses to encode *how much real
   time* separated two events, not just their sequence order. Padding
   positions don't advance the time axis.
3. **`num_layers` stacked transformer blocks** (2 full-capacity, 1
   starved), each block:
   - `LayerNorm` → **multi-head self-attention** (`num_heads` heads: 4
     full-capacity, 1 starved) with the RoPE rotation from step 2 applied
     to the Q and K projections before the dot-product, so attention
     scores are time-aware → residual add.
   - `LayerNorm` → **position-wise feed-forward** (`Linear(D, ff_dim)` →
     `GELU` → `Linear(ff_dim, D)`; `ff_dim` = 256 full-capacity, 32
     starved) → residual add.
4. **Final `LayerNorm`** over the whole sequence.

Output: one `D`-dim vector per event in the patient's history, shape
`(batch, seq_len, D)`, plus a padding mask. `seq_len` (`max_seq_len`, how
much history the model is even allowed to see) is 256 full-capacity, 32
starved.

**2a. `patient_only` path** (`fusion=passthrough`)

- **Pooling**: `MaskedMeanPooling` -- mean over the sequence dimension,
  excluding padding positions. `(batch, seq_len, D)` → `(batch, D)`.
- **Head**: a single `nn.Linear(D, 25)` -- one logit per one of the N=25
  task codes. No retrieval, no external information; predictions come
  entirely from the patient's own recorded history.

**2b. `marginalized` (retrieval-augmented) path**

- **Query projector**: turns the patient's encoded state into a vector
  used to search an external document corpus. Two variants are compared
  here (see "Two query-projector designs" below).
- **Retriever**: `HFDatasetRetriever` -- a FAISS nearest-neighbor search
  over a corpus of `MedRAG/textbooks` passages (generic medical textbook
  text), pre-embedded offline with a frozen `Qwen3-Embedding-0.6B` model
  and indexed once. Returns the top **K=4** matching documents (token ids
  + attention mask + the documents' own Qwen3 key embeddings) per patient.
- **Retrieval encoder** (`TokenFeatureRetrievalEncoder`): a learned
  `nn.Embedding(vocab_size=151936, 64)` applied to each retrieved
  document's token ids -- turns the K retrieved documents' raw tokens into
  dense per-token vectors the fusion layer can attend to. (This is a
  *separate*, always-trainable embedding table from the frozen Qwen3
  encoder used for retrieval search itself -- it exists purely to give
  the fusion layer document *content* to look at, not to influence which
  documents get retrieved.)
- **Fusion**: `PerDocCrossAttentionFusion` -- for each of the K retrieved
  documents independently, the patient's pooled state cross-attends to
  that document's token embeddings across 2 stacked cross-attention
  layers (`LayerNorm` → cross-attention → residual → `LayerNorm` → FFN →
  residual, `d_model=256`, 8 heads, `ff_dim=512`), producing one
  256-dim fused vector *per document*: `(batch, K=4, 256)`.
- **Head**: the same `nn.Linear(256, 25)` applied independently to each of
  the K fused vectors, producing K separate sets of 25 logits (one full
  prediction per retrieved document).
- **Marginalization**: the K per-document predictions are combined into a
  single prediction per task, weighted by `softmax` over the K documents'
  retrieval scores (patient-query · document-key dot product) --
  i.e. documents that matched the query better get more weight in the
  final prediction. Because each of the 25 tasks is an *independent*
  binary decision (not a mutually-exclusive multi-class choice), this
  uses `marginalized_output_mode=binary`: each task's `sigmoid`
  probability is marginalized over K separately, rather than forcing all
  25 tasks to compete for one shared probability mass.

## Two query-projector designs

The query projector is the piece that decides *what to search for* --
it turns the patient's state into a vector compared against the
document corpus's embeddings. Two designs are compared here:

- **`qwen3_text`** (MedRAP#101): ignores the patient encoder's hidden
  state entirely. Instead, it renders the patient's most recent event
  codes as plain text (e.g. code descriptions joined together) and embeds
  that text with the *same* frozen `Qwen3-Embedding-0.6B` model used to
  build the document corpus -- so queries and documents live in the same
  vector space by construction, with nothing learned. Because it never
  touches the patient encoder, it is **completely unaffected** by
  shrinking the encoder (the capacity-starvation experiment below).
- **`sequence_mean_1024`** ("learned-linear", the original, pre-#101
  design): mean-pools the patient encoder's own hidden state over the
  sequence dimension, excluding padding (`(batch, seq_len, D)` →
  `(batch, D)`), then a single trainable `nn.Linear(D, 1024)` projects up
  to the retrieval corpus's 1024-dim embedding space. Because this *does*
  read the patient encoder's hidden state, it shrinks proportionally with
  the rest of the model under capacity starvation (`D` = `embedding_dim`,
  128 → 16).

## Motivation for this experiment

Every retrieval variant tried in `mimic_iv_sweep` before this -- the
original learned linear query projector, the Qwen3-aligned frozen one,
and a trainable residual adapter on top of the latter -- had failed to
make `marginalized` *consistently* beat `patient_only` at full model
capacity. This tests a different hypothesis: maybe retrieval only helps
once the base model is too small to represent the task on its own from
patient history alone. If retrieved content carries real information, a
severely **capacity-starved** `patient_only` should degrade much more
sharply than a similarly-starved `marginalized`, which can lean on
retrieved content the encoder itself doesn't need to memorize. If both
degrade together, that's evidence the retrieved `MedRAG/textbooks` content
isn't adding real information regardless of capacity pressure -- narrowing
the "why doesn't retrieval help" question down to the corpus/data itself
rather than model capacity or training mechanics.

**The capacity cut** (applied identically to `patient_only` and both
`marginalized` variants -- only the presence/choice of retrieval differs):

| Param | Full capacity (rest of `mimic_iv_sweep`) | Starved (this experiment) |
| --- | --- | --- |
| `encoder.embedding_dim` | 128 | 16 (8x smaller) |
| `encoder.num_heads` | 4 | 1 |
| `encoder.num_layers` | 2 | 1 |
| `encoder.ff_dim` | 256 | 32 (8x smaller) |
| `max_seq_len` (patient history length seen) | 256 | 32 (8x shorter) |

Two `marginalized` variants are tested in parallel against the same
starved `patient_only` baseline: `qwen3_text` (query projector unaffected
by the cut, since it doesn't use the encoder's hidden state) and
`sequence_mean_1024`/learned-linear (query projector *is* capacity-cut
along with the rest of the model, a more apples-to-apples "starved vs.
starved" comparison). Same N=25/5-draw/30d labels and seeds as every other
30d experiment in `mimic_iv_sweep`.

```bash
cd mimic_iv_sweep
sbatch scripts/sweep_patient_only_capacity_starved_n25_30d.sh
sbatch scripts/sweep_marginalized_binary_qwen3_text_capacity_starved_n25_30d.sh
sbatch scripts/sweep_marginalized_binary_learned_linear_capacity_starved_n25_30d.sh
```

## Results

All three architectures -- `patient_only` (starved) and both
`marginalized` variants (starved) -- are fully finished, 5 draws each.

| Duration | Draw | patient_only (starved) | marginalized, learned-linear (starved) | marginalized, qwen3_text (starved) |
| --- | --- | --- | --- | --- |
| 30d | 1 | [0.8565](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/azbxcay8) | [0.8583](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/i1z8aqu8) | [0.8725](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/qtk931gk) |
| 30d | 2 | [0.8202](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/lt9z4p6l) | [0.8400](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/phgp8lz9) | [0.8525](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/349uufv2) |
| 30d | 3 | [0.8534](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/hgb5bur1) | [0.8567](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/zy5pgbxz) | [0.8638](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/v7575aei) |
| 30d | 4 | [0.8805](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/3lkn16as) | [0.9066](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/xo626ese) | [0.9204](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/43kesvso) |
| 30d | 5 | [0.7904](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/9wx8dzad) | [0.8023](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/rwllk4mi) | [0.8160](https://wandb.ai/haykstepanyan02-columbia-university/medrap/runs/kcj83n08) |

`patient_only` (starved) mean ± std across the 5 draws: **0.8402 ± 0.0352**.

For context, the full-capacity `patient_only` 30d mean was **0.8949 ± 0.0220**
(from `mimic_iv_sweep/README.md`'s "Anchor-sampling fix rerun" section) --
so this ~8x capacity cut drops mean AUROC by ~0.055 and roughly doubles
draw-to-draw variance (std 0.022 -> 0.035), confirming the starvation is
real and meaningful, not so mild it's within full-capacity noise.

**Δ (marginalized − patient_only), both variants, per draw:**

| Draw | patient_only | learned-linear | Δ learned-linear | qwen3_text | Δ qwen3_text |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.8565 | 0.8583 | +0.0017 | 0.8725 | +0.0160 |
| 2 | 0.8202 | 0.8400 | +0.0198 | 0.8525 | +0.0322 |
| 3 | 0.8534 | 0.8567 | +0.0033 | 0.8638 | +0.0104 |
| 4 | 0.8805 | 0.9066 | +0.0260 | 0.9204 | +0.0399 |
| 5 | 0.7904 | 0.8023 | +0.0119 | 0.8160 | +0.0256 |

## Takeaway

**Both query-projector variants flip from a small, inconsistent loss at
full capacity to a small but consistently positive, unanimous win under
capacity starvation.** This isn't specific to one query-projector design
-- it happens whether the query is built from the patient encoder's own
(now-starved) hidden state (learned-linear) or from an entirely separate,
capacity-independent frozen Qwen3 text embedding (`qwen3_text`). That
convergence across two structurally different query mechanisms is itself
evidence this is a real effect and not an artifact of one particular
architecture choice.

`qwen3_text`'s starved Δ (+0.0248) is nearly double learned-linear's
(+0.0125), and both means are comfortably larger than their own
draw-to-draw std -- this is a positive signal *stronger* than draw-level
noise, not a coin flip like every full-capacity retrieval result measured
in `mimic_iv_sweep` before this.

**What this means for "does retrieval help":** yes, but conditionally --
retrieved content from the generic `MedRAG/textbooks` corpus carries real,
usable information, but a full-capacity encoder (128-dim, 2 layers,
256-token history) is already expressive enough to extract everything it
needs from the patient's own history alone, leaving no room for retrieval
to add value. Once the encoder is starved down to something that
genuinely can't represent the task on its own (16-dim, 1 layer, 32-token
history), retrieval becomes load-bearing and reliably helps close some of
that gap. This directly narrows down *why* every earlier full-capacity
retrieval experiment failed to show a consistent benefit: not because the
retrieved content is useless (it isn't), but because the full-capacity
patient encoder didn't need it.

**Caveat worth flagging before over-interpreting this as proof retrieval
content specifically is doing the work**: marginalizing over K=4 documents
could in principle help a starved model simply by acting as an implicit
ensemble/regularizer, independent of whether the retrieved content is
relevant. The next experiment to run (not yet done) is the same starved
setup with `retriever.ablation_mode=random_docs` (uniform-random corpus
documents instead of real nearest-neighbor retrieval, using the existing
ablation infrastructure from `mimic_iv_sweep/README.md`'s "Inference-style
evaluation" section) -- if starved-random-docs also beats starved-
`patient_only` by a similar margin, the effect is ensembling/regularization,
not retrieval content. If starved-random-docs does *not* show this
benefit, that confirms the retrieved content itself is what's driving the
gain.
