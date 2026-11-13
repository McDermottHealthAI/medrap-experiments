# Scaling laws v2: why retrieval does worse than patient only

The v1 grid ([`results/scaling_laws`](../results/scaling_laws/README.md)) found retrieval (#2) worse
than patient only (#1) almost everywhere. A model that is given extra input should be able to
ignore it, so that pointed to the architecture, not the documents. This directory finds where it
goes wrong, fixes it, and redesigns the experiment around the advisor's comments.

The goal and its rules (no gate or switch that turns retrieval off, paired runs, no test-set
tuning) are in [`GOAL.md`](GOAL.md); each experiment round is logged in [`EXPERIMENTS.md`](EXPERIMENTS.md).

## Where we are

1. **The harm is fixed.** Retrieval was 12–18% worse than patient only because of a **cold start**:
   the patient encoder and the document reader trained together from scratch, and the documents
   drowned the patient signal. Starting the encoder from the trained patient-only checkpoint (the
   advisor's "use the good configuration as a seed") removes it: against patient only trained for
   the same extra epochs, retrieval is now even in all 4 cells (G1 met).
2. **The documents don't help yet.** The model reads them (random documents at test make it worse
   in all 4 cells), but a model trained on random documents does just as well (G2 half met, G3 not
   met).
3. **Richer documents give the first positive signs**, as the advisor suggested. A frozen Qwen3 runs
   over each retrieved passage, and the fusion attends over every token of every layer. Retrieved
   passages now beat random ones, significantly at 256x4x8. Retrieval beats patient only at
   64x2x4/153K (+0.36%, interval excludes 0). The effects are small (0.2–0.4%). At D 10K they turn
   negative (−0.4% and −2.2%, both intervals including 0), and 256x4x8 stays slightly worse at 153K
   (−0.5%).
4. **A reader that starts at the patient-only model helps at large `D`** (Round 8, `xres`). Each
   patient code reads the passage and adds an update to its own state, with no new projection, so
   the head reads the patient-only input plus the documents' updates (no gate). At `D` 153K it beats
   patient only for both sizes (+0.27% and +0.75%, intervals excluding 0), and it beats the same
   model trained on random passages in 3 of 4 cells: the first sign the passages carry knowledge.
   The 256x4x8 gain holds on validation loss too; the 64x2x4 one is fragile. More passages (K 32,
   128) and per-size learning rates did not help.
5. **The gain replicates on fresh draws** (Round 9: new tasks, patients and seeds; nothing was chosen
   on them): +0.3% and +0.8% at `D` 153K again. Over 6 draws, though, it is slightly worse than
   patient only at `D` 10K (−0.3%, −0.5%), where the reader overfits. Running: dropout in the
   reader (Round 10), and the patient-only part of the scaling grid.

## The advisor's comments: what we did

| Comment | What we did | Where | Status |
|---|---|---|---|
| **Drop setting #3** (static query from Qwen on the last 32 codes) | Dropped. | [`design.md`](design.md) §3 | done |
| **Fewer `W` settings**, at sensible points | `W` 4–128 single-layer dropped. Stage 0 uses 64x2x4, 128x2x4, 256x4x8; the retrieval rounds use 64x2x4 and 256x4x8. | [`run.sh`](run.sh), [`submit.sh`](submit.sh) | done |
| **Minimal patient-only sweep first**, then seed retrieval from the good configuration | Stage 0: 3 `W` × 3 learning rates × 2 dropouts at `D` 153K, 3 draws (results below). Retrieval now starts its encoder from the patient-only checkpoint, and this turned out to be the fix for the harm. | `submit.sh stage0`, `WARM=true` | done |
| **One task setting** (~1,000 tasks) | `T` = 1,000 everywhere. | `run.sh tasks` | done |
| **3 draws** instead of 5 | 3 draws everywhere. | `submit.sh` | done |
| **More documents** (128, 512), since the loss learns "which retrieved document is best" | K 4, 32, 128 at `D` 153K with precomputed contextual passages (K 128 with half batches and gradient accumulation). The model reads more with more passages, but a model trained on random passages gains as much (Round 6). 512 not run. | `submit.sh kdocs` | done (128); no gain from knowledge |
| **Retrieval-only models** at the same settings, especially for richer document encodings | Run once, with context-free documents: 17% worse than patient only at `D` 153K, so those documents carry little on their own. Not yet run with contextual documents. | `retrieval_only` in `run.sh` | partly done |
| **Count compute and parameters accurately** | Measured, not estimated: one real batch per configuration through forward and backward under `FlopCounterMode` (attention, cross-attention, and the frozen Qwen pass for on-the-fly documents), plus the FAISS search analytically (2.6 × 10⁸ FLOPs per patient). Parameters split into `N` (trained, without the code table), all trained, and frozen. Results below. | [`count_compute.py`](count_compute.py), [`results_compute.csv`](results_compute.csv) | done |
| **Other law forms, loss scale only, drop poor fits** | Planned: the 7 law families of `simple-scaling-laws`, fitted on loss, with a fit-quality cutoff. All results so far are already in loss. No laws fitted yet: the scaling grid waits until retrieval helps. | [`design.md`](design.md) §10 | to do |
| **Paired analyses**: (patient only − retrieval) as one number per shared split and seed | Every comparison is a paired difference per draw (same tasks, patients and seed), reported as a mean with a 95% t-interval. | [`ladder.py`](ladder.py) | done |
| **Richer document value encodings** (all Qwen tokens; run Qwen over the documents, take all layers, attend through them) | New MedRAP encoder: a frozen Qwen3-Embedding-0.6B runs over each retrieved passage; fusion attends over all 256 tokens × 29 layers, with a learned layer embedding (`DOCS=attend`). Also last layer only (`DOCS=last`) and a learned layer mix (`DOCS=mix`). | MedRAP `feat/qwen-hidden-state-docs`; `run.sh` (`DOCS`) | done: precomputed last layer, read by the residual per-code reader (Round 8), helps at D 153K |

## Results so far

All at lr 1e-3, dropout 0, `K` 4, 3 draws unless the row says otherwise. From Round 4 on, the baseline is patient only trained for the same extra epochs. `Δ` = L(patient only) − L(retrieval) as a share of patient
only's test loss, paired per draw; positive means retrieval is better. Full tables in
[`results/scaling_laws_v2`](../results/scaling_laws_v2/README.md) and [`EXPERIMENTS.md`](EXPERIMENTS.md).

**Stage 0, patient only at `D` 153K** (best learning rate per size; dropout changes the loss by < 1%;
the best checkpoint is the last in every run):

| `W` | Best lr | Test loss |
|---|---|---|
| 64x2x4 | 3e-3 | 0.02127 |
| 128x2x4 | 1e-3 | 0.02122 |
| 256x4x8 | 3e-4 | 0.02077 |

**Retrieval, `D` 153K** (`Δ` vs patient only):

| Round | Change | 64x2x4 | 256x4x8 | Reading |
|---|---|---|---|---|
| 1 | v1 fusion, top 1 at test (as in v1) | −18.7% | −22.6% | the v1 result, reproduced |
| 1 | same K in training and test (top 4) | −12.1% | −18.1% | a third of the harm was the test mismatch |
| 1 | masked mean over patient codes (bug fix) | −13.4% | −18.1% | little effect |
| 2 | one document, no averaging | −15% | | averaging is not the cause |
| 3 | cross-attention output starts at zero | −15.8% | −18.3% | no effect |
| 3 | centered search keys | −12.6% | −15.3% | no effect |
| 3 | **encoder warm-started from patient only** | **+2.5%** | **+2.0%** | the harm is gone, but see Round 4 |
| 4 | the same, against patient only trained for the same extra epochs | +0.1% | −0.6% | even (G1 met); the +2% was the extra training |
| 4 | the same model trained on random documents | +0.1% | −0.9% | as good as retrieved documents: they add no knowledge yet |
| 5 | **contextual documents: all tokens of all 29 Qwen layers** (vs the same baseline) | **+0.36%** (3/3 draws, interval excludes 0) | −0.54% | first gain; retrieved beat random passages (significant at 256x4x8) |
| 5 | the same, trained on random documents | +0.16% | −0.72% | retrieved passages now beat this control, by 0.2% |
| 6 | precomputed last layer, K 4 / 32 / 128 | +0.10 / +0.32 / +0.45% | −0.64 / −0.40 / −0.52% | trained on random passages gains as much |
| 6 | each size at its Stage 0 learning rate | −1.6% | −0.74% | worse; dropped |
| 7 | `residual`: patient-only input + one-query reading of each passage | +0.14% (interval excludes 0) | −0.12% | better than v1, weak reader |
| 8 | **`xres`: per-code reader on the encoder's own states** | **+0.27%** (excludes 0) | **+0.75%** (excludes 0) | not worse at large `D` (G1+) and helps in 2 cells (G3) |
| 8 | the same, trained on random passages | +0.00% | +0.30% | retrieved passages beat this control (intervals exclude 0) |

**Retrieval, `D` 10K** (`Δ` vs patient only trained for the same extra epochs):

| Round | Change | 64x2x4 | 256x4x8 | Reading |
|---|---|---|---|---|
| 4 | encoder warm-started, context-free documents | −0.25% | −1.0% | even (intervals include 0) |
| 5 | contextual documents, all layers | −0.36% | −2.2% | worse, mostly one draw at 256x4x8 (overfitting) |
| 5 | the same, trained on random documents | −0.34% | −1.0% | as good as or better than retrieved |
| 8 | `xres` | −0.19% | −0.25% | even (intervals include 0) |
| 8 | the same, trained on random passages | −0.77% | −0.39% | worse than retrieved (excludes 0 at 64x2x4) |

**Compute and parameters** ([`count_compute.py`](count_compute.py); training FLOPs per patient per step, forward
and backward, including the search; ratio to patient only of the same `W`):

| Configuration | 64x2x4 FLOPs | × patient only | 256x4x8 FLOPs | × patient only | `N` (64 / 256) | frozen |
|---|---|---|---|---|---|---|
| patient only | 2.2 × 10⁸ | 1 | 4.2 × 10⁹ | 1 | 136K / 2.4M | 0 |
| v1 reader, word-piece or precomputed passages | 1.7 × 10¹⁰ | 78 | 2.1 × 10¹⁰ | 5.1 | 2.3M / 4.5M | 155M (E0) / 0 |
| v1 reader, all layers on the fly | 1.5 × 10¹² | 6,900 | 1.5 × 10¹² | 364 | 2.3M / 4.6M | 596M |
| `residual` (Round 7) | 1.5 × 10⁹ | 6.8 | 8.5 × 10⁹ | 2.0 | 361K / 3.6M | 0 |
| **`xres`** (Round 8) | 3.5 × 10⁹ | 16 | 2.1 × 10¹⁰ | 5.0 | 515K / 4.5M | 0 |

Training compute of a warm-started run includes its patient-only checkpoint. At 256x4x8/153K, `xres`
costs 1.2 × 10¹⁶ FLOPs in total against 3.8 × 10¹⁵ for `patient_only_warm`, about 3×, for its +0.75%.
Most of a retrieval step is projecting the 5 × 256 document vectors (1024-wide) into the reader, and
the search alone (2.6 × 10⁸) costs as much as a 64x2x4 patient-only step. Whether the gain survives
at equal compute is what the scaling grid answers. Encoding the corpus once (precomputed passages)
costs 3.9 × 10¹⁶ FLOPs, shared by all runs.

Also ruled out: the loss implementation, gradient clipping (never triggers), and overfitting (v1 is
worse on training loss too).

## Appendix A. What the v1 runs show

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

## Appendix B. Where the architecture goes wrong (first hypotheses)

Reading `PerDocCrossAttentionFusion` and `RetrievalAugmentedModel` in MedRAP (`c3700ab`, which is
also current `main`):

| # | Problem | Where | Effect |
|---|---|---|---|
| C1 | **Mean over padding.** Fusion averages the fused vectors over all 256 positions, padding included; #1 uses a masked mean. Padding positions have no patient content, so after cross-attention they hold mostly document content. | `fusion.py:582`, `x.mean(dim=1)` | A patient with 50 codes has only 50/256 of its pooled vector from its own codes; the rest is document. That fits the model's dependence on the document (observation 4); it should be worst for short patients (not measured yet). In v1, 33% of positions were padding. |
| C2 | **No way to ignore the document.** Every patient position cross-attends to the document and adds the result; softmax attention always returns a weighted average of document tokens, so content-free documents still add noise. The patient-only solution is not where training starts and is hard to reach. | `_CrossAttentionLayer` | #4 is worse than #1 (observation 3). |
| C3 | **Different documents in training and at test.** Training averages predictions over the top 4 + null document; test over the top 1 + null. The null document's weight roughly doubles at test, shifting every probability. | `run.sh`: `retriever.k=4` train, `k=1` test | Miscalibration, which loss punishes more than AUROC. |
| C4 | **Different head input and regularisation.** #2's head reads a 256-wide, LayerNormed vector through two extra layers with dropout 0.1; #1 reads the encoder's own mean with no dropout. Both use the learning rate tuned for neither. | `cross_attention_perdoc_medium.yaml` | #2 doesn't gain from a bigger encoder (best #2 cells are at `W` 16–64, best #1 cells at 128x2x4). |
| C5 | **Weak retriever signal.** The query learns only from which of 5 documents predicts best. With 4 candidates, mostly poor early on, that is a noisy estimate. The keys are only mildly anisotropic (mean pairwise cosine 0.22; `center_keys.py`), so the collapse onto a few passages comes mostly from the query. | `retriever.k=4` | #2 ≈ #4 (observation 2). In v2, 45–80% of test patients retrieve the same top passage. |
| C6 | **Context-free documents.** A document is 256 frozen Qwen3 word-piece embeddings, with no context between them. | `qwen3_token_embeddings` | Little usable knowledge, even with the right passage. |

These were the first hypotheses. C1 (fixed, MedRAP#123) and C3 (fixed) were real bugs but explained
little; the main cause turned out to be the cold start (Rounds 3–4). A gate for C2 was tried and
dropped: it lets the model switch retrieval off, which the goal's rules forbid. C6 is what the
contextual-document encoder addresses.

## Appendix C. What other retrieval models do

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


## Files

| File | What it does |
|---|---|
| [`GOAL.md`](GOAL.md) | the goal, success criteria G1–G3, rules, and the exploration plan |
| [`EXPERIMENTS.md`](EXPERIMENTS.md) | each round: hypothesis, change, numbers, decision |
| [`design.md`](design.md) | the full v2 design: configurations, stages, compute counting, analysis |
| [`run.sh`](run.sh) | one step: `tasks DRAW`, or `train CONFIG W D DRAW` (train, then test); switches `LR`, `DROPOUT`, `K`, `NULL_DOC`, `ZERO_INIT`, `WARM`, `CENTER`, `DOCS`, `LAYER_STRIDE` |
| [`submit.sh`](submit.sh) | submits `tasks`, `stage0`, `stage1`, `nomix`, `ablate`, `warm`, `docs`, `encode`, `kdocs`, `tuned`, `residual`, `confirm`, `grid` to SLURM; skips finished and queued runs |
| [`encode_last_layer.py`](encode_last_layer.py) | the precomputed last-layer passage vectors (`DOCS=pre`) |
| [`count_compute.py`](count_compute.py) | measured training FLOPs and parameter counts per configuration → [`results_compute.csv`](results_compute.csv) |
| [`ladder.py`](ladder.py) | collects test results into `results.csv`; paired differences with 95% intervals |
| [`gaps.py`](gaps.py) | the paired v1 numbers in Appendix A |
| [`center_keys.py`](center_keys.py) | the centered-keys retrieval database (Round 3) |
| [`slim_checkpoint.py`](slim_checkpoint.py) | drops frozen weights and optimizer state from finished checkpoints (disk quota) |
| [`pyproject.toml`](pyproject.toml) | environment: MedRAP with the fixes (v1's configurations use `../scaling_laws/.venv`) |
