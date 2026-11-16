# scaling_laws_v2: results

The full log of every round is in [`EXPERIMENTS.md`](EXPERIMENTS.md); the design is in
[`design.md`](design.md). Figures are made by [`figures.py`](figures.py) and
[`curves.py`](curves.py).

## What we are doing

**Question.** Does a patient model predict better if it also retrieves and reads medical textbook
passages for each patient? And how does the answer change with model size and the amount of
training data?

**Data.**
- **Patients:** MEDS records. Each patient is a sequence of up to 256 medical codes.
- **Tasks:** 1,000 binary tasks, each "does code X occur in the next 30 days?", with X drawn at
  random. One model predicts all 1,000 at once.
- **Corpus:** 125,848 medical textbook passages of 256 tokens each.

**The models.**

- **#1 patient only.** A transformer encoder reads the patient's codes. The mean of its output
  states, `h`, goes to a linear head with 1,000 outputs.
- **#2 retrieval.** The same encoder, plus four steps:
  1. **Query.** `h` is projected to a 1,024-dimensional query.
  2. **Search.** An exact inner-product search over the passages' Qwen3-Embedding vectors returns
     the top K passages. An empty "null" passage is always added, so the model can rely on the
     patient alone for a given patient.
  3. **Read.** A *reader* combines the patient's encoder states with each passage's token vectors
     and makes one prediction per passage. Which reader to use is the main architectural question
     (see "The readers").
  4. **Combine.** The K + 1 predictions are averaged, weighted by the softmax of the search scores
     (REALM-style marginalization). The query gets its gradient through these weights, so the
     search learns which passages help.
- **#4 random passages.** #2 trained and tested with K random passages instead of the retrieved
  ones. This is the control: #2 − #4 is the value of what the passages *say*, and #4 − #1 is the
  value of the extra layers alone.
- **#R retrieval only.** The patient reaches the head *only* through the passages it retrieved.
- **#3** (a frozen Qwen query from the last 32 codes) was dropped, as requested.

**Settings varied.**
- `W` is the encoder size, written width × layers × heads: 64x2x4, 128x2x4 or 256x4x8.
- `D` is the number of training patients, from 3K to 153K.
- `K` is the number of passages per patient, from 4 to 512. K is the same in training and at test.

**How results are reported.**
- Each result is a **paired** test-set difference, Δ = L(#1) − L(#2), as a % of #1's test loss.
  **Positive means retrieval is better.**
- A *draw* is one random set of tasks, patients and training seed, shared by every model.
- Each number is the mean over 3 draws (6 where noted), with a 95% t-interval (the error bars in
  every figure). "3/3" means retrieval was better in all three draws.
- *Cold* models train from scratch and are compared with patient only. *Warm* models start their
  encoder from the trained patient-only model, and are compared with patient only continued from
  the same checkpoint for the same extra epochs.

## Bottom line

1. **Retrieval helps with enough data.** At 100K–153K training patients it beats patient only by
   **+0.3% to +0.9%** at 64x2x4 and 256x4x8 (128x2x4: +0.13%, interval includes 0), and the gain
   replicated on fresh draws.
2. **The passages themselves carry part of that gain.** The same model trained on *random* passages
   gains less. Passage knowledge accounts for about +0.2% to +0.5%; the rest is extra capacity.
3. **It hurts with little data:** −0.3% to −3% below about 30K patients.
4. **At equal training compute, patient only wins.** Retrieval needs about 6.5× the compute to
   reach the same loss, and the fitted laws put patient only ahead at every budget.
5. **Two changes made it work, both added during the experiments (neither is from the design
   review):**
   - *Warm start:* the encoder starts from the trained patient-only model. This removed v1's
     4–18% harm.
   - *The `xres` reader* ("The readers", and "Architecture changes in detail").
   - Neither is a gate: every model always reads its passages.
6. **The newest experiments didn't change this.**
   - Drawing K passages at random in training is slightly worse than the top K.
   - All Qwen layers are no better than the last layer, with either reader.
   - Retrieval only with a learned query is much better than the previous retrieval only (by 2–8%)
     but still 3–7% worse than patient only.
7. **The retriever barely weights its passages.** Every reader puts almost equal weight on the 4
   retrieved passages and the null passage ("How the retriever behaves").

![Paired differences between the models against D](figures/delta.png)

*Left: `xres` vs patient only. Middle: the model trained on random passages vs patient only.
Right: retrieved vs random passages, the part of the gain that comes from what the passages say.*

---

## The readers

v1's retrieval was 4–18% worse than patient only, so most of the work went into finding out why
and changing the reader. Three readers were compared, all with the same encoder, search and
marginalization. **Only v1's reader came with the original design. `residual` and `xres` were
designed during these experiments, each to fix a problem found in the one before. Neither was
suggested in the design review.**

| Reader | What it does | Problem it addresses | Origin |
|---|---|---|---|
| **v1 per-passage cross-attention** | Projects the encoder states to width 256. Two cross-attention layers let every patient code attend to the passage's tokens, then a LayerNorm and the head. | – | v1 design |
| **residual** | The head reads `h + c_k`, where `c_k` is one query vector's reading of passage k. With no passage content, this is exactly #1's input. | v1 starts away from #1: its head sees the patient only through new, untrained layers. | added (Round 7) |
| **`xres`** | v1's reader applied in place to the encoder's own states: each patient code cross-attends to the passage and adds the result to its own state, with no new projection or LayerNorm. The head then reads the patient-only input plus the passage updates. | `residual` reads each passage with a single query, which is too weak a reader. | added (Round 8) |

![Readers](figures/readers.png)

| Reader | Encoder start | Passages | 64x2x4, 10K | 64x2x4, 153K | 256x4x8, 10K | 256x4x8, 153K |
|---|---|---|---|---|---|---|
| v1 | cold | word pieces | −4.5% | −13.4% | −5.6% | −18.1% |
| v1 | warm | word pieces | −0.25% | +0.14% | −1.0% | −0.61% |
| v1 | warm | Qwen, last layer | – | +0.10% | – | −0.64% |
| v1 | warm | Qwen, all layers | −0.36% | **+0.36%** | −2.2% | −0.54% |
| residual | warm | Qwen, last layer | −0.22% | **+0.14%** | −0.35% | −0.12% |
| **`xres`** | warm | Qwen, last layer | −0.30% (6 draws) | **+0.27%** (6/6) | −0.48% (6 draws) | **+0.79%** (6/6) |

- **Warm start removes most of the harm.** v1 becomes about even with patient only, but its
  random-passage model (#4) does just as well, so the passages add nothing with v1.
- **`xres` is the only reader that is ahead at 153K for both sizes**, and it beats its own #4
  control, so part of its gain comes from what the passages say.
- **Caveat on choosing it:** `xres` was chosen after seeing draws 1–3. Draws 4–6 were then run
  fresh, with nothing chosen on them, and gave the same result (+0.3% and +0.8% at 153K).

---

## Architecture changes in detail

Every change made to the retrieval model during these experiments: what it does, where it is in MedRAP,
why it was made, and what it did. All are off by default in MedRAP and switched on by `run.sh`. The
MedRAP code is on PRs #123–#125.

### The model they change

Shapes for one batch: `B` = 32 patients, `S` = 256 codes per patient, `W` = encoder width, `K` passages
plus the null passage, 256 tokens per passage, `T` = 1,000 tasks.

```mermaid
flowchart LR
    P[["codes<br/>B × S"]] --> E["patient encoder<br/>(RoPE transformer)"]
    E -->|"states B × S × W"| Q["query: masked mean →<br/>linear W → 1024"]
    Q --> F["exact search over<br/>125,848 passage keys<br/>top K + null"]
    F -->|"B × (K+1) × 256 × 1024"| R["reader"]
    E -->|"states B × S × W"| R
    R -->|"B × (K+1) × W"| H["head: linear W → T"]
    H -->|"B × (K+1) × T"| M["Σ_k w_k σ(z_k)<br/>w = softmax(scores)"]
    F -.->|"scores"| M
```

The reader is the only part that differs between the models below. Patient only skips the search
and the reader: the head reads the masked mean `h` of the encoder states.

### Summary

| # | Change | MedRAP switch | Why | Effect | Kept |
|---|---|---|---|---|---|
| 1 | Masked mean in the reader | `PerDocCrossAttentionFusion` pools valid codes only | v1 averaged over padding: a 50-code patient's vector was 80% padding positions, which after cross-attention hold mostly passage content | small | yes (bug fix) |
| 2 | Same K in training and at test | `run.sh`: `retriever.k=K` for both | v1 trained on 4 + null and tested on 1 + null, doubling the null passage's weight at test | about a third of v1's harm | yes |
| 3 | Reader output starts at zero | `zero_init_attn_out=true` | stop random passages drowning the patient at the start | none | no |
| 4 | Centred search keys | `center_keys.py` | patients all retrieved the same passages | none | no |
| 5 | **Warm start** | `training.module.init_encoder_from` | the encoder learned while being fed passage noise | **removed the 13–18% harm** | yes |
| 6 | **Contextual passages** | `PrecomputedTokenStatesRetrievalEncoder` (last layer), `QwenHiddenStatesRetrievalEncoder` (all layers) | word pieces have no context | **first real gains** | last layer |
| 7 | `residual` reader | `GatedDocFusion(gated=False)` | v1's head sees the patient only through new layers | better than v1, still weak | no |
| 8 | **`xres` reader** | `PerDocCrossAttentionFusion(residual=True)` | combine v1's per-code reading with an untouched patient path | **+0.3% / +0.8% at 153K** | yes |
| 9 | Sampled K | `HFDatasetRetriever(sample_pool=16)` | let the query learn from passages outside its top K | slightly worse | no |
| 10 | Learned query, retrieval only | `GatedDocFusion(mode="learned")` | the patient may affect the prediction only through retrieval | much better than the previous retrieval only | experiment |

### Warm start (5)

- **What:** before training, the patient encoder (code embeddings and transformer layers) is copied
  from the trained patient-only checkpoint of the same `W`, `D` and draw. Only `model.encoder.*` is
  copied. The query projection, the reader and the head start fresh.
- **Why:** in v1 the encoder and the reader trained together from random weights. Early on the
  passages are noise, and the encoder learned to work with that noise instead of learning the
  patient. v1 was worse even on *training* loss, so it was not overfitting.
- **Fair baseline:** a warm run trains for more epochs in total. It is compared with
  `patient_only_warm`, which is patient only continued from the same checkpoint for the same epochs,
  also with a fresh head. That baseline alone is 1.5–2.6% better than the original patient only.
- **Effect:** −13.4% → +0.1% (64x2x4) and −18.1% → −0.6% (256x4x8) at D 153K with v1's reader
  (Rounds 3–4).

### Contextual passages (6)

What the reader sees for each passage, 256 vectors of width 1,024:

| Encoding | How it is built | Runtime cost |
|---|---|---|
| word pieces (v1) | Qwen3's input embedding of each token, a table lookup; no context | none |
| last layer (`DOCS=pre`) | Qwen3-Embedding-0.6B run once over the whole library; the last-layer state of each of the 256 tokens, stored as fp16 (66 GB, `encode_last_layer.py`). At training time the retrieved rows are read and cached on the GPU | a lookup |
| all layers (`DOCS=attend`) | the frozen Qwen3-Embedding runs over each retrieved passage at every step; all 29 hidden states (embeddings + 28 layers) of each token, each LayerNorm-ed, plus a learned embedding per layer, so the reader attends over 29 × 256 vectors | the Qwen pass, 6–9 GPU-h per run |

The search keys stay the same in all three: one Qwen3-Embedding vector per passage.

### The `xres` reader (8)

For each patient and each of its K + 1 passages separately:

```
x = encoder states of the patient                     S × W   (no projection: d_model = W)
p = LN(passage token vectors)                         256 × 1024
repeat 2 times:
    x = x + Attention(query = LN(x), keys = values = p)   each code reads the passage
    x = x + FFN(LN(x))                                     W → 2W → W, GELU
v_k = masked mean of x over the patient's codes      W       (no output LayerNorm)
z_k = head(v_k)                                       T logits
```

- **Heads:** the attention uses the encoder's own head count (4 or 8). The keys and values are
  projected 1,024 → W inside the attention.
- **What v1 does differently:** v1 first projects the states to width 256 (`patient_proj`) and
  applies a LayerNorm at the end (`out_norm`). `residual=True` replaces both with the identity.
- **Why that matters:** with the identity, `v_k` = (the patient-only head input `h`) + (the mean of
  the updates from passage k). The head is fitted on the same kind of vector as in patient only,
  and passages enter only as additions. In v1, the head saw the patient only after a fresh random
  projection, two layers and a LayerNorm, so it started far from the patient-only model and had to
  relearn it.
- **Not a gate:** the attention output is not initialised at zero and nothing can scale the
  passages away. Every passage changes every prediction from step 1.
- **Size:** 379K (64x2x4) and 2.1M (256x4x8) extra trained parameters. Without the code table that
  is N = 515K and 4.5M, against 136K and 2.4M for patient only.
- **Compute:** 16× and 5× patient only's FLOPs per training step. Most of it is projecting the
  5 × 256 passage vectors from 1,024 to W.

### The `residual` reader (7)

```
h   = masked mean of the encoder states               W
q   = linear(h)                                        one query vector
c_k = Attention(q, passage k) then FFN                 W
z_k = head(h + c_k)
```

It starts at the patient-only head input, like `xres`. But the whole patient asks one question of
each passage, which is too little: random-passage models were worse than it, but it gained only
+0.14% / −0.12%.

### Sampled K (9)

- **What:** in training, the search returns the top 16 passages and K = 4 are drawn from them
  without replacement, in proportion to `softmax(score / √1024)` (Gumbel-top-K). At test the top 4
  are used.
- **How:** `HFDatasetRetriever(sample_pool=16)` adds Gumbel noise `−log(−log u)` to the scaled
  scores and keeps the 4 largest. The mixing weights still use the unperturbed scores of the
  passages drawn.
- **Why:** the query only gets a learning signal from the passages it is shown. Always showing it
  its top 4 can lock it into the same few passages.

### Learned query for retrieval only (10)

- **Before** (`GatedDocFusion(mode="query")`): passage k is read by a query vector computed from
  the patient's retrieval query, and the head sees only that reading, `z_k = head(c_k(patient, k))`.
- **Now** (`mode="learned"`): one learned vector, the same for every patient, reads every passage.
  So `c_k` depends on passage k only, and `z_k = head(c_k(k))`.
- **What that means:** every passage carries a fixed prediction profile over the 1,000 tasks, read
  from its text. The patient chooses which 4 profiles to mix, and with what weights, through the
  search alone.

**Warm start** is also our addition. The review asked for the good patient-only *configuration*
to be used as the seed for retrieval. We went further and start the retrieval model's encoder from
the trained patient-only *weights*.

---

## The review's points

### Reduce runs: drop #3, fewer `W`, ~1,000 tasks, 3 draws

- **#3** (frozen Qwen query on the last 32 codes) is dropped.
- **Model sizes:** three, 64x2x4, 128x2x4 and 256x4x8. The single-layer sizes are gone.
- **Tasks:** T = 1,000 everywhere.
- **Draws:** 3 everywhere. The key cells have 6: draws 4–6 were added later as a fresh
  replication, and nothing was chosen on them.

### Minimal patient-only sweep, then seed retrieval from the good configuration

Patient only at D 153K: 3 sizes × 3 learning rates × 2 dropouts, 3 draws each.

![Patient-only sweep](figures/stage0.png)

| `W` | Best lr | Test loss |
|---|---|---|
| 64x2x4 | 3e-3 | 0.02127 |
| 128x2x4 | 1e-3 | 0.02122 |
| 256x4x8 | 3e-4 | 0.02077 |

- Dropout matters little, and the last checkpoint is always the best one.
- Retrieval at each size's tuned lr was *worse* than at 1e-3 (−1.6% and −0.74% with v1), so all
  runs use lr 1e-3, and the baseline gets the same lr.
- Retrieval is seeded from this configuration, and in addition from the trained patient-only
  weights (warm start, above).

### More documents (128, 512)

![More passages](figures/k.png)

| D 153K | K 4 | K 32 | K 128 | K 512 |
|---|---|---|---|---|
| v1, 64x2x4 (random-trained) | +0.10 (−0.05) | +0.32 (+0.33) | +0.45 (+0.43) | |
| v1, 256x4x8 (random-trained) | −0.64 (−0.85) | −0.40 (−0.39) | −0.52 (−0.73) | |
| `xres`, 64x2x4 (random-trained) | +0.27 (−0.02) | **+0.34** (+0.07) | | rerunning |
| `xres`, 256x4x8 (random-trained) | **+0.79** (+0.31) | **+0.86** (+0.55) | | rerunning |

- **Performance rises a little with K, but the random-passage control (#4) rises about as much.**
  More passages add capacity, not knowledge.
- With `xres`, K 32 beats K 4 by only +0.07% / +0.10% (intervals include 0). It mainly makes the
  gain steadier: ±0.04% and ±0.10%, the tightest intervals of any setting.
- **K 512** (`xres` and its #4 control, both sizes, 3 draws): two of the twelve runs finished
  (`xres`, 64x2x4, draws 2–3). The other ten were killed at the same minute on the cluster on
  Nov 15, some after 19 h, before saving a checkpoint. All are running again, about 15 h (64x2x4)
  to 22 h (256x4x8) each. One run had finished training and is only being re-tested.
- **Annealing K** during training is not done. No K beats its random control, so there is no
  number to anneal towards yet.

### Choose random K instead of top K in training

**What it is.** In training, the search returns the top 4K candidates, and K of them are drawn at
random, in proportion to softmax(score / √1024) (Gumbel-top-K). At test, the top K are used as
usual. The query then also gets a learning signal from passages it would not have ranked first.
Uniformly random passages are #4, already in every figure.

**Run:** `xres` with K 4 drawn from the top 16, at D 10K and 153K, both sizes, 3 draws. MedRAP:
`HFDatasetRetriever(sample_pool=16)`.

![Sampled K](figures/sampled_k.png)

| Paired, % of loss | 64x2x4, 10K | 64x2x4, 153K | 256x4x8, 10K | 256x4x8, 153K |
|---|---|---|---|---|
| sampled vs patient only | −0.51 | +0.14 | −0.72 | +0.66 |
| **sampled vs top K** | −0.32 ± 0.95 (0/3) | −0.13 ± 0.22 (0/3) | −0.48 ± 0.33 (0/3) | −0.09 ± 0.20 (0/3) |
| sampled vs trained on random passages | +0.25 ± 0.94 | +0.14 ± 0.36 (3/3) | −0.33 ± 0.85 | +0.36 ± 0.48 (3/3) |

- **Sampling is slightly worse than top K in every draw of every cell** (0.1–0.5%), though only
  256x4x8 at 10K excludes 0.
- **It does spread retrieval out:** more distinct top passages (0.42 / 0.54 per patient, against
  0.32 / 0.32 with top K; see "How the retriever behaves"). But the model leans less on its
  passages: swapping in random passages at test costs it 0.4–0.5%, against 0.6–0.9% for top K.
- **Decision:** not kept. Exploration changes *which* passages are retrieved, not how useful they
  are.

### Retrieval-only models, at the same settings and with richer encodings

![Retrieval only](figures/retrieval_only.png)

| Δ vs patient only | 64x2x4, 10K | 64x2x4, 153K | 256x4x8, 10K | 256x4x8, 153K |
|---|---|---|---|---|
| word pieces, retrieval query (cold) | −8.0% | −16.7% | −9.7% | −17.2% |
| Qwen last layer, retrieval query | −5.5% | −15.7% | −7.0% | −13.9% |
| Qwen all layers, retrieval query | −2.9% | −17.3% | −9.7% | −16.6% |
| Qwen last layer, **learned query** | **−3.3%** | **−6.0%** | **−4.4%** | **−6.7%** |

- **Retrieval only is 5–17% worse than patient only, and the gap grows with D.** The passages work
  only as an *addition* to the patient record; on their own they describe the patient poorly.
- **All layers don't help retrieval only either:** against the last layer, −1.3% to −2.5% at three
  of the four cells (+2.4% at 64x2x4/10K); no interval excludes 0.
- **Learned query** (requested): one learned vector, the same for every patient, reads every
  passage, so the patient affects the prediction *only* through which passages are retrieved and
  how they are weighted (details in "Architecture changes in detail").
  - **It is much better than the patient-dependent query:** +2.1%, +8.4%, +2.5% and +6.3% (paired,
    3/3 draws in every cell; the interval excludes 0 at 64x2x4/153K).
  - **It is still worse than patient only:** 3–7% on test.
  - **What this suggests:** with a fixed reading per passage, the 125,848 passages act as a large
    library of learned prediction profiles, and the search learns to pick the patient's nearest
    ones. It is also the model that concentrates its mixing weights most (see "How the retriever
    behaves"): the weights are its only channel for the patient.
  - **Why it beats the patient-dependent query is not established.** One guess: there the query
    both searches and reads, and the reading gives it an easier route to the loss than learning to
    search. Not tested.

**Does it get there with less training?** Validation loss against patients seen in training.
- **Training length:** a cold run sees D × epochs patients, which is 50K at D 10K and 460K at
  153K.
- **Warm runs:** they start from a patient-only checkpoint, so their curves begin where that run
  ended. Everything is compared at the same total training.
- **The jump at the start of each warm curve:** only the encoder is copied, so the head and reader
  start fresh.

![Validation loss against patients seen](curves.png)

| D 153K (64x2x4 / 256x4x8) | Patients seen at the end | Final validation loss vs #1's final | Patients seen until it first reaches #1's final loss |
|---|---|---|---|
| #1 patient only | 460K | 0 | 460K |
| #R, word pieces (cold) | 460K | +16.3% / +17.3% | never |
| #R, last layer (warm) | 919K | +12.9% / +11.0% | never |
| #R, all layers (warm) | 919K | +14.2% / +13.8% | never |
| #R, last layer, learned query (warm) | 919K | **+3.6% / +3.9%** | never |
| #1 continued (warm) | 919K | −2.2% / −2.4% | 613K / 674K |
| `xres`, last layer | 919K | −2.3% / −3.2% | 644K / 674K |
| `xres`, all layers | 919K | −2.4% / −3.0% | 674K / 674K |

- **Retrieval only never reaches patient only**, even with twice the training. The learned query
  comes closest (3.6–3.9% above it at the end).
- **Inside `xres`, richer encodings don't get there sooner.** `xres` with either encoding reaches
  #1's loss after about the same amount of training as #1 continued.

### Richer encodings: run Qwen over the passages, take all layers, attend through them

**What we ran.**
- **Word pieces (v1):** Qwen's input embeddings of the passage tokens, without context.
- **All layers:** a frozen Qwen3-Embedding-0.6B runs over each retrieved passage, and the reader
  attends over all 256 tokens × 29 layers, with a learned layer embedding. This is computed on the
  fly in every step.
- **Last layer:** the last-layer states of all 256 tokens, computed once for the whole corpus
  (66 GB) and looked up.

![Passage encodings](figures/encodings.png)

- **Context is what matters.** Contextual passages gave the first gains with either reader.
- **All layers add nothing over the last layer** (`xres`, paired: +0.05 ± 0.19% and −0.25 ± 0.38%),
  at about 18× the run time. The precomputed last layer is the encoding we kept.

### Count compute cost and number of parameters accurately

**Method:**
- **FLOPs:** measured, not estimated. One real training batch per configuration goes through
  forward and backward under PyTorch's `FlopCounterMode`.
- **Search:** the FAISS search runs outside torch, so it is added analytically (2.6 × 10⁸ FLOPs
  per patient).
- **Parameters:** `N` counts trained parameters without the code embedding table. Frozen
  parameters are listed separately.
- **Script:** [`count_compute.py`](count_compute.py).

| Per patient per step (fwd + bwd) | 64x2x4 | × #1 | 256x4x8 | × #1 | `N` (64 / 256) | frozen |
|---|---|---|---|---|---|---|
| #1 patient only | 2.2 × 10⁸ | 1 | 4.2 × 10⁹ | 1 | 136K / 2.4M | 0 |
| v1 reader, word pieces or last layer | 1.7 × 10¹⁰ | 78 | 2.1 × 10¹⁰ | 5.1 | 2.3M / 4.5M | 155M / 0 |
| v1 reader, all layers on the fly | 1.5 × 10¹² | 6,900 | 1.5 × 10¹² | 364 | 2.3M / 4.6M | 596M |
| **`xres`, last layer** | 3.5 × 10⁹ | 16 | 2.1 × 10¹⁰ | 5.0 | 515K / 4.5M | 0 |

![Test loss against training compute](figures/compute.png)

- **Each line is one model size,** going from D 3K to 153K. Training compute for a warm run
  includes the patient-only checkpoint it starts from.
- **The faint thick line is the lower envelope:** the best loss reached at that compute.
- **Patient only's envelope is left of `xres`'s everywhere.** Its best model reaches 0.02073 at
  5.9 × 10¹⁴ FLOPs, while `xres` reaches 0.02071 at 3.8 × 10¹⁵ (6.5×).
- **Encoding the corpus once** costs 3.9 × 10¹⁶ FLOPs. It is shared by all runs and not included
  above.

### Scaling laws: other law forms, loss scale only, drop poor fits

**Grid:** 3 sizes × D {3K, 10K, 30K, 100K, 153K} × 3–6 draws, with four models per cell: #1 (cold,
then warm), `xres`, and `xres` trained on random passages (#4).

| Δ vs patient only | 64x2x4 | 128x2x4 | 256x4x8 |
|---|---|---|---|
| D 3K | −1.55 ± 0.40 | −2.85 ± 1.15 | −1.88 ± 1.89 |
| D 10K | −0.30 ± 0.20 | −1.50 ± 1.32 | −0.48 ± 0.44 |
| D 30K | −0.24 ± 0.58 | −0.65 ± 1.09 | −0.05 ± 2.15 |
| D 100K | **+0.21 ± 0.23** (3/3) | −0.04 ± 0.58 | **+0.82 ± 0.37** (3/3) |
| D 153K | **+0.27 ± 0.08** (6/6) | +0.13 ± 0.89 | **+0.79 ± 0.50** (6/6) |

**Law fits** (script: [`grid.py`](grid.py); fits: [`grid_fits.csv`](grid_fits.csv)):
- **Five forms tried,** all fitted on loss: separable power, shifted separable, Kaplan joint,
  Skaling joint, bounded rational.
- **Quality gate:** a fit is kept only if all three hold:
  - R² over the cell means ≥ 0.9;
  - leave-one-cell-out RMSE below the run-to-run SD;
  - no ill-conditioning or non-convergence warning.
- **Result:** only separable power, `L = E + A·N^−α + B·D^−β`, passes. It passes for patient only
  (R² 0.994) and `xres` (0.996), and narrowly fails for #4. The other four forms fail on
  ill-conditioning or optimizer warnings.

![IsoFLOP curves](figures/isoflop.png)

- **How the isoFLOP curves are made:** for each training budget, loss against model size N, where
  each N gets the D that the budget allows. They come from each model's fitted law and its
  measured compute per patient.
- **Compute-optimal frontier (right):** patient only is below `xres` at every budget. The gap
  narrows as the budget grows, but the curves don't cross within the range shown.
- **Caveat:** the laws are fitted on 3 model sizes, so the curves are extrapolation outside the
  shaded range.
![Δ against model size and compute](figures/delta_scale.png)

- **Δ against model size (left):** at each D the gain is largest at 256x4x8 and smallest at
  128x2x4. Model size matters less than data: the lines are ordered by D, not by N.
- **Δ against compute (right):** 64x2x4 crosses 0 near 10¹⁵ FLOPs, 128x2x4 near 3 × 10¹⁵ and
  256x4x8 between 3 and 7 × 10¹⁵, each at 30K–100K patients. Larger models need more compute
  before retrieval pays off.
- **Comparing the models through their laws is inconclusive** (P(`xres` better) 0.5–0.6). Laws
  fitted separately per model can't resolve a 0.3–0.8% difference, and an additive law can't
  represent a gain that grows with D. The paired table above is the evidence.

### How the retriever behaves

Statistics MedRAP logs on every test batch (32 patients) from the retrieved passages, averaged
over batches and draws, K 4, D 153K. No new runs were needed.

![Retrieval behaviour](figures/retrieval.png)

- **Cold v1 collapsed.** In a batch of 32, 71–82% of patients shared the same top passage, and
  only 13% of top passages were distinct. Its retrieval was close to a constant.
- **Warm start broke the collapse.** With the trained encoder, 35–52% share the top passage and
  29–46% of top passages are distinct, for every reader.
- **`xres` retrieves about as diversely as the other warm readers** (42% shared, 32% distinct). Its
  gain comes from reading the passages better, not from finding different ones.
- **The mixing weights are nearly uniform for every reader:** 4.3–4.7 of 5 passages effectively
  weighted. The model reads all 4 retrieved passages and the null passage almost equally, so the
  search scores do little beyond choosing the 4. A sharper retriever is the most obvious unused
  lever.
- **Sampled K retrieves more diversely** (up to 54% distinct), with no gain (above).
- **Learned-query retrieval only concentrates its weights the most** (2.9–3.8 of 5): it is the
  only model whose patient information must flow through the weights.

---

## Analysis: paired differences

- **Every number here is one paired difference per draw** (same split, tasks, patients and seed),
  averaged over draws with a t-interval.
- **Why this matters:** the draw-to-draw spread in loss (SD about 5%) is much larger than the
  effects (under 1%), so unpaired comparisons would show nothing.
- **For the law fits,** each draw's loss is centred on its own mean, for the same reason.

## Requested plots

| Plot | Figure |
|---|---|
| Test loss vs training compute | [`figures/compute.png`](figures/compute.png) |
| IsoFLOP curves | [`figures/isoflop.png`](figures/isoflop.png) |
| Difference between models | [`figures/delta.png`](figures/delta.png), [`figures/delta_scale.png`](figures/delta_scale.png), [`figures/readers.png`](figures/readers.png), [`figures/sampled_k.png`](figures/sampled_k.png) |
| Qwen over the passages, all layers | [`figures/encodings.png`](figures/encodings.png), [`figures/retrieval_only.png`](figures/retrieval_only.png) |
| Random K instead of top K | [`figures/sampled_k.png`](figures/sampled_k.png) |
| Retrieval only with a learned query | [`figures/retrieval_only.png`](figures/retrieval_only.png), [`curves.png`](curves.png) |
| Retrieval behaviour (collapse, weights) | [`figures/retrieval.png`](figures/retrieval.png) |

## Still running

| What | Runs | Expected |
|---|---|---|
| K 512: `xres` and its #4 control, D 153K, both sizes (rerun after the cluster failure) | 10 | ~15 h (64x2x4) to ~22 h (256x4x8) each |

Finished since the last update: sampled K, retrieval only with a learned query, and retrieval only
with all Qwen layers (36 runs; results above).

## Next

- **Positive control.** Give the model a synthetic passage that states some of the patient's
  labels. If it can't exploit even that, the reader is the limit; if it can, the corpus is.
- **Rare tasks.** Break the existing results down by task frequency. Retrieval should help most
  where the patient data is thin.
- **Code-aligned passages.** One definition per diagnosis or drug code, retrieved for the
  patient's own codes, instead of generic textbook chunks.
- **A cheaper `xres`.** Most of its compute projects 256 vectors per passage. Reading 16 chunk
  means per passage (already built) cuts that about 16×, the most direct attack on the 6.5× compute
  gap.
- **A sharper retriever.** The mixing weights are nearly uniform, so the search scores do little.
  A lower softmax temperature, or more weight on the retrieval loss, would test whether better
  weighting adds to the gain.
