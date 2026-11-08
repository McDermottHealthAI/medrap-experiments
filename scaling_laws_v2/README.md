# Scaling laws v2: why retrieval does worse than patient only

The v1 grid ([`results/scaling_laws`](../results/scaling_laws/README.md)) found retrieval (#2) worse
than patient only (#1) almost everywhere. A model that is given extra input should be able to
ignore it, so that is a sign that the architecture is wrong, not that the documents are useless.
This directory finds where it goes wrong and redesigns the experiment around the fix and the
advisor's comments. The design is in [`design.md`](design.md).

| File | What it does |
|---|---|
| [`design.md`](design.md) | the v2 design: fixed architecture, configurations, stages, compute counting, analysis |
| [`gaps.py`](gaps.py) | the paired v1 numbers below, from `results/scaling_laws/runs.csv` |
| [`run.sh`](run.sh) | one step: `tasks DRAW`, or `train CONFIG W D DRAW` (train, then test) |
| [`submit.sh`](submit.sh) | submits `tasks`, `stage0` or `stage1` to SLURM; skips finished and queued runs |

## 1. What the v1 runs show

From [`gaps.py`](gaps.py). Each pair is #1 and #2 with the same `W`, `D`, `T` and draw, so the same tasks,
patients and seed. **Gap** is #2's test loss relative to #1's (positive means retrieval is worse).
**Swap** is how much #2's loss rises when its top document is replaced by a random one at test.

| `D` | Pairs | Gap | #2 worse | Swap |
|---|---|---|---|---|
| 1K | 200 | +0.4% | 61% | +4% |
| 3K | 200 | +2.5% | 78% | +15% |
| 10K | 200 | +5.6% | 95% | +23% |
| 30K | 200 | +10.0% | 100% | +34% |
| 100K | 200 | +14.4% | 100% | +73% |
| 153K | 200 | +17.0% | 100% | +66% |

| `W` | 4 | 8 | 16 | 32 | 64 | 128 | 128x2x4 | 256x4x8 |
|---|---|---|---|---|---|---|---|---|
| Gap | +2.2% | +4.8% | +7.8% | +9.6% | +10.2% | +10.4% | +10.3% | +11.3% |
| Swap | +11% | +9% | +17% | +29% | +34% | +66% | +54% | +68% |

On the 45 control pairs (`W` 4, 16, 64 × `D` 1K, 10K, 100K × `T` 625), random documents (#4) are
3.6% worse than #1, and #2 is no better than #4 (+0.1%).

Four observations:

1. **#2 is worse in 89% of the 1,200 pairs**, and the gap **grows** with data and model size (+0.4%
   at 1K patients, +17% at 153K). Retrieval doesn't fail to help; it gets in the way, and more so
   the better the patient model is.
2. **The documents carry no knowledge.** #2 (top documents) equals #4 (random documents). Whatever
   #2 retrieves is worth no more than a random passage.
3. **The extra architecture costs loss by itself.** #4 is worse than #1, so adding the retrieval
   path with content-free documents makes the model worse.
4. **Yet #2 depends on its document.** Swapping the top document for a random one at test raises
   the loss by up to 73%. #2 has learned to read the document. But the document is chosen by a
   query computed from the patient, so it is a lossy function of the patient: it adds no information
   the encoder doesn't already have, and the model now relies on it.

## 2. Where the architecture goes wrong

Reading `PerDocCrossAttentionFusion` and `RetrievalAugmentedModel` in MedRAP (`c3700ab`, which is
also current `main`):

| # | Problem | Where | Effect |
|---|---|---|---|
| C1 | **Mean over padding.** Fusion averages the fused vectors over all 256 positions, padding included; #1 uses a masked mean. Padding positions have no patient content, so after cross-attention they hold mostly document content. | `fusion.py:582`, `x.mean(dim=1)` | A patient with 50 codes has only 50/256 of its pooled vector from its own codes; the rest is document. That fits the model's dependence on the document (observation 4); it should be worst for short patients (not measured yet). In v1, 33% of positions were padding. |
| C2 | **No way to ignore the document.** Every patient position cross-attends to the document and adds the result; softmax attention always returns a weighted average of document tokens, so content-free documents still add noise. The patient-only solution is not where training starts and is hard to reach. | `_CrossAttentionLayer` | #4 is worse than #1 (observation 3). |
| C3 | **Different documents in training and at test.** Training averages predictions over the top 4 + null document; test over the top 1 + null. The null document's weight roughly doubles at test, shifting every probability. | `run.sh`: `retriever.k=4` train, `k=1` test | Miscalibration, which loss punishes more than AUROC. |
| C4 | **Different head input and regularisation.** #2's head reads a 256-wide, LayerNormed vector through two extra layers with dropout 0.1; #1 reads the encoder's own mean with no dropout. Both use the learning rate tuned for neither. | `cross_attention_perdoc_medium.yaml` | #2 doesn't gain from a bigger encoder (best #2 cells are at `W` 16–64, best #1 cells at 128x2x4). |
| C5 | **Weak retriever signal.** The query learns only from which of 5 documents predicts best. With 4 candidates, mostly poor early on, that is a noisy estimate. The textbook keys are also anisotropic (mean pairwise cosine 0.80), so many patients may retrieve the same few passages. | `retriever.k=4` | #2 ≈ #4 (observation 2). Retrieval diversity was never measured. |
| C6 | **Context-free documents.** A document is 256 frozen Qwen3 word-piece embeddings, with no context between them. | `qwen3_token_embeddings` | Little usable knowledge, even with the right passage. |

C1–C4 explain why retrieval is **worse** than patient only; C5–C6 why it is **no better** than random
documents. C1 and C3 are bugs; C2 and C4 are design choices that other retrieval models avoid.

## 3. What other retrieval models do

- **Start at the model without retrieval.** Flamingo adds each cross-attention block through a
  `tanh` gate initialised at 0, so at the start the model is exactly the frozen language model and
  only uses the new input once it helps ([Alayrac et al. 2022](https://arxiv.org/abs/2204.14198)).
  ReZero does the same for residual branches in general ([Bachlechner et al. 2020](https://arxiv.org/abs/2003.04887)).
  RETRO adds retrieval as a residual cross-attention branch beside an unchanged decoder
  ([Borgeaud et al. 2021](https://arxiv.org/abs/2112.04426)).
- **Make "no document" an option.** REALM adds a null document to every retrieval for examples that need no
  knowledge ([Guu et al. 2020](https://arxiv.org/abs/2002.08909)); we have this already.
  Irrelevant passages measurably hurt retrieval-augmented models unless training teaches them to
  be ignored ([Shi et al. 2023](https://arxiv.org/abs/2302.00093);
  [Yoran et al. 2023](https://arxiv.org/abs/2310.01558)).
- **Many documents for the retriever's signal.** REALM, RAG and Atlas train the retriever through
  the same marginal likelihood we use, over tens to a hundred passages, not four
  ([Lewis et al. 2020](https://arxiv.org/abs/2005.11401);
  [Izacard et al. 2022](https://arxiv.org/abs/2208.03299)). Fusion-in-Decoder's accuracy keeps
  rising up to 100 passages ([Izacard & Grave 2020](https://arxiv.org/abs/2007.01282)).
- **Same documents in training and at test.** RAG and FiD use the same K for both.
- **Contextual document vectors from several layers.** A learned weighted mix of a frozen model's
  layers (ELMo's scalar mix; [Peters et al. 2018](https://arxiv.org/abs/1802.05365)) is a cheap way
  to use all layers without storing a choice of one.
- **Explore beyond the top K.** Sampling K documents without replacement in proportion to their
  scores (Gumbel-top-K; [Kool et al. 2019](https://arxiv.org/abs/1903.06059)) lets the retriever
  see and learn from passages it would not have ranked first.

## 4. What v2 changes

Details in [`design.md`](design.md).

- **Fixed fusion.** The prediction for document k is `head(h + tanh(g) · c_k)`. `h` is exactly #1's
  masked mean, `c_k` is what the patient reads from document k, and the gate `g` starts at 0. At
  the start #2 *is* #1, and documents are used only if they lower the loss (C1, C2, C4).
- **Same K in training and at test** (C3).
- **K up to 512** and a **sampled-K** variant (C5, advisor). The fused patient side is a single
  query token, so the cost per document is small.
- **Retrieval only** (advisor): the same fusion with the learned retrieval query in place of the
  patient, and no `h`. This shows what the documents carry on their own.
- **Contextual documents** (C6, advisor): Qwen3 hidden states in 16 chunks per passage, last layer
  first, then several layers with a learned mix.
- **Fewer runs** (advisor): #3 dropped, `W` 4–16 dropped, one task count (`T` = 1,000), 3 draws, a
  patient-only configuration search first, then a diagnosis ladder, then the K sweep, then the
  scaling grid.
- **Compute and parameters counted**, FAISS search included, which v1 left out (it is ~2.6 × 10⁸
  FLOPs per patient, roughly 60% of a `W` 128x2x4 encoder's training cost by the 6·N·S estimate).
- **Analysis on paired differences**, loss only, several law forms, and no law reported when none fits.

## Status

| Piece | Status |
|---|---|
| Diagnosis of the v1 results (this page) | done |
| Stage 0: patient-only configuration search | ready to run (needs `run.sh tasks` first) |
| Stage 1 rungs on existing MedRAP (#1, v1 #2 with matched-K test, #4) | ready to run |
| MedRAP changes for the fixed fusion, retrieval only, precomputed documents, sampled K | to implement (design.md §8) |
| Stages 2–4 | after the MedRAP changes and Stage 1 |
