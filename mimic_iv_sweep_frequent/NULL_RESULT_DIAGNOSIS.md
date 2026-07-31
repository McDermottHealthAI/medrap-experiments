# Why retrieval doesn't beat no-retrieval in `mimic_iv_sweep_frequent`

**Status:** interim report. Written 2026-07-30 from the run artifacts in
`/groups/mm6677_gp/hs3627/medrap-experiments/mimic_iv_sweep_frequent` and the MedRAP source at
`/groups/mm6677_gp/zzw2102/MedRAP/src/medrap`. A 26-agent verification workflow was stopped at 21/26
agents to conserve session budget, so the "further work" items at the end are genuinely open.

**Provenance caveat:** everything here is based on the **local checkout** at
`/groups/mm6677_gp/zzw2102/medrap-experiments` (`main` at `944945b`, "Merge PR #8 feat/duration-variance-n25")
and `hs3627`'s run outputs. **I did not fetch GitHub.** If `main` has moved since `944945b`, re-check
anything that depends on script or README content.

---

## Executive summary

The headline "Average AUROC" table cannot support the conclusion drawn from it, and separately, the
retrieval mechanism was not functioning in any of the runs behind it.

Three things are true simultaneously. **First, the table is a readout artifact.** It reports the *last*
logged `val/auroc/mean`, and the marginalized N=16 run swings across a **0.239 AUROC range within itself**
with no configuration change. Re-reading the *identical runs* at their best validation gives a mean
difference of **+0.0000** and turns N=16's headline −0.1909 into −0.0068. **Second, the estimator is far
too noisy to resolve the effect being sought.** Labels are ~0.05–0.2% positive, so each per-task AUROC is
computed from a median of 2–6 positive examples; the standard error of `val/auroc/mean` is roughly
±0.03–0.10, which brackets every reported delta except N=16's. **Third, the retriever is genuinely dead** —
in all 32 marginalized runs the query vector is essentially identical for every patient, every patient
retrieves the same top document, and the document softmax is fully saturated, so marginalization over K
is a no-op.

The single most important sentence: **for this table to be informative, `val/auroc/mean` would need a
standard error well below the effect size you are hunting (~0.01), and it currently has a standard error
one to ten times *larger* than the effect.** Until that is fixed, no arrangement of architectures can
answer "does retrieval help."

The best evidence already in your repo is not this table. It is the **paired N=25 duration×variance study**
in `../mimic_iv_sweep/README.md` (lines 646–672): mean Δ = **−0.0059 ± 0.0110** across 10 paired runs
(t = −1.68, 6/10 negative). That is the honest current estimate — retrieval contributes nothing and costs
a little — and it is a much better basis for the conversation with your advisor than the 8-point table.

---

## How the numbers in the table were produced

Traced from the sbatch script to the logged metric.

1. **`scripts/sweep_marginalized_binary_n1248.sh`** / **`scripts/sweep_patient_only_n1248.sh`** invoke
   `medrap-train` with Hydra overrides. Both point at the *same* label directory
   `data/tasks/n${N}/tasks` (verified in both `resolved_config.yaml` files).
2. **`cli.py:422`** `lightning.seed_everything(42, workers=True)` → **`cli.py:424`**
   `instantiate_training_module(cfg)`.
3. **`train/factory.py:63–74`** constructs `RetrievalAugmentedModel` with the seven injected stages.
4. **`model/model.py:454–516`** `forward()`:
   - `encoder` → `TimeDeltaRoPEPatientEncoder`, `(B,256) → (B,256,128)`
   - `query_projector` → `SequenceMeanQueryProjector`, `(B,256,128) → (B,1,1024)`
   - `retriever` → `HFDatasetRetriever` FAISS search, `k=4`
   - `retrieval_encoder` → `TokenFeatureRetrievalEncoder`, `(B,1,4,256) → (B,1,4,256,64)`
   - `fusion` → `PerDocCrossAttentionFusion` → `(B,4,256)`
   - marginalized branch (`model.py:472–511`): head applied per document → `per_doc_logits (B,4,N)`;
     `differentiable_retrieval_scores` recomputed in-graph → `doc_scores (B,4)`;
     `_marginal_binary_logits` → `logits (B,N)`.
   The `patient_only` arm takes `PassthroughFusion` and the ordinary tail
   `pooled = self.pooling(fused_state)` at **`model.py:513`**.
5. **`train/lightning_module.py:203–204`** buffers val logits/targets; **`:230–292`** computes
   `val/auroc/mean` via `multitask_auroc_torch` (`train/metrics.py:100–133`), averaging per-task AUROCs
   over tasks with both classes present.
6. The trainer config `conf/training/trainer/lightning_wandb.yaml` sets `val_check_interval: 0.2` and
   **`limit_val_batches: 200`**, so each validation scores a fixed **6,400-row prefix** of the 25,059-row
   tuning split, 15 times over 3 epochs. **The README quotes the 15th (last) value** — verified: my
   extraction of the last `val/auroc/mean` from every `metrics.csv` reproduces all 16 table cells exactly.

**The two arms in equations.** Patient-only:

$$h = \mathrm{Enc}(x) \in \mathbb{R}^{256\times128},\qquad \bar h = \frac{1}{256}\sum_{s=1}^{256} h_s,\qquad \ell = W\bar h + b \in \mathbb{R}^{N}$$

Marginalized-binary, per task $n$:

$$\pi_k = \mathrm{softmax}_k(q^\top k_k),\qquad p_n = \sum_{k=1}^{K}\pi_k\,\sigma(\ell_{k,n}),\qquad \text{logit}_n = \mathrm{logit}\big(\mathrm{clamp}(p_n,\epsilon,1-\epsilon)\big)$$

---

## The five hypotheses, ranked by centrality

Categories: **(M)** measurement/statistics, **(B)** silent implementation bug, **(C)** conceptual.

---

### 1. The reported metric is a last-value readout of a wildly oscillating, few-positive estimator — **(M)**

**What it is.** The table reports the final logged `val/auroc/mean`, not a converged or selected value.
That quantity oscillates violently within a single run because it is built from a handful of positive
examples. The N=16 cell — which alone contributes 73% of the summed gap across the table — is a lottery draw.

**Why I'm confident (high).** All computed by me directly from the `metrics.csv` files:

| N | patient_only within-run range | marginalized within-run range |
|---|---|---|
| 1 | 0.019 | 0.011 |
| 2 | 0.047 | 0.096 |
| 4 | 0.006 | 0.068 |
| 8 | 0.013 | 0.062 |
| 16 | 0.083 | **0.239** |
| 32 | 0.017 | 0.046 |
| 64 | 0.048 | 0.053 |
| 128 | 0.014 | 0.037 |

The marginalized N=16 run alone spans 0.7012–0.9405 with its **maximum at validation 1 of 15**. The
reported "gap" of 0.1909 is *smaller than that one run's own oscillation*.

Re-reading the identical runs under three equally defensible summary rules:

| readout | mean Δ (marg − patient_only) | patient_only wins |
|---|---|---|
| last value (as published) | −0.0327 | 6/8 |
| **best validation** | **+0.0000** | 5/8 |
| mean of last 5 validations | −0.0296 | 6/8 |

Note the honest nuance: the effect does *not* vanish under every rule. Under "best" the arms tie exactly;
under "last" and "mean-of-last-5" the marginalized arm is worse. That pattern says the marginalized arm
*degrades late in training* relative to patient-only — which is a real, small effect (see §2 and §4) —
while the *dramatic per-N structure*, especially N=16, is noise.

**What would change my mind:** if a rerun with 5 seeds showed the N=16 deficit reproducing at similar
magnitude.

**The mechanism in equations.** With ~0.08% positives, each task's AUROC uses $n_1$ positives against
$n_2 \approx 6400$ negatives. The Hanley–McNeil standard error is

$$\mathrm{SE}(\hat A)=\sqrt{\frac{A(1-A)+(n_1-1)(Q_1-A^2)+(n_2-1)(Q_2-A^2)}{n_1 n_2}},\quad Q_1=\frac{A}{2-A},\ Q_2=\frac{2A^2}{1+A}$$

At $A=0.92$ with the real label counts I measured:

| N | positives/task in the 6,400-row val prefix (min/median/max) | per-task SE | SE of `auroc/mean` (indep → correlated) |
|---|---|---|---|
| 4 | 4.1 / 5.7 / 10.5 | 0.078 | 0.039 → 0.077 |
| 16 | 2.0 / 4.0 / 12.5 | 0.095 | 0.025 → 0.097 |
| 64 | 1.5 / 3.8 / 21.7 | 0.096 | 0.012 → 0.093 |
| 128 | 0.8 / 2.8 / 22.2 | 0.112 | 0.010 → 0.114 |

Tasks share patients and the same underlying recency signal, so the truth is near the correlated column.
Because `binary_auroc_torch` (`train/metrics.py:64–98`) is the exact Mann–Whitney statistic, a single
positive changing rank moves a task's AUROC by up to $1/n_1$ — with $n_1=2$, that is **0.5**.

**Location.** `conf/training/trainer/lightning_wandb.yaml` (`limit_val_batches: 200`,
`val_check_interval: 0.2`); `src/medrap/train/lightning_module.py:230–292`; `src/medrap/train/metrics.py:100–133`.

**Fix.** (a) Drop `limit_val_batches` so the full 25,059-row tuning split is scored. (b) Report
mean ± 95% CI over ≥5 seeds, not a single last value. (c) Select and report at the *same* checkpoint —
either monitor `val/auroc/mean` in `ModelCheckpoint` or report the best-val value consistently for both
arms. (d) Add PR-AUC/average-precision, which is far more informative at 0.08% prevalence.

**Discriminating experiment (cheap, ~0 GPU).** Re-score the *existing* checkpoints on the full tuning
split with a bootstrap: for each arm and N, load `checkpoints/last.ckpt`, predict once over all 25,059
rows, then bootstrap-resample rows 1,000× to get a CI on `val/auroc/mean`.
*If H1 is true:* the CIs for the two arms overlap at every N including 16. *If false:* N=16's CIs separate
cleanly. Runtime ≈ 20 min CPU for all 16 checkpoints; no retraining.

---

### 2. The retriever is collapsed in every run: one query for all patients, one document, saturated softmax — **(B + C)**

**What it is.** The query projector converges to a nearly patient-independent function, every patient
retrieves the same document, and the document softmax saturates so hard that marginalizing over K is
arithmetically a no-op. The "retrieval-augmented" arm is therefore patient-only plus a *constant*
side-channel — which explains why it never wins, and why it costs a little.

**Why I'm confident (very high).** I extracted end-of-training diagnostics from **all 32 marginalized
runs** (4 variants × 8 N). Values are the mean of the last 20 logged points:

- `query/train/offdiag_cos_mean`: 0.74–0.999 (median ≈0.93) — queries near-collinear across patients
- `query/train/dim_std_mean`: → 0.0 at several N (N=16 base: exactly 0.000) — query *literally identical*
- `retrieval/train/top1_mode_frac`: **0.92–1.00 in all 32 runs** — nearly every patient gets the same top doc
- `retrieval/train/unique_doc_ratio`: 0.031–0.088 (N=16 base: 0.03125 = exactly 4 unique docs per 128 slots)
- `retrieval/train/differentiable/effective_k_mean`: **1.00–1.15 in all 32 runs**
- `query/train/norm_mean`: grows from ~12 to **56–265**
- `grad_norm/train/query_projector`: decays to 1e-5 … **0.00e+00**

Critically, **k=10 collapses too** (`effective_k` ≈ 1.0 with ten documents available). That single fact
explains two of the README's puzzles: why k=10 didn't systematically help, and why the top-1-doc-only
held-out eval matched the k=4 eval — marginalization was *already* a no-op, so dropping 3 of 4 documents
changes nothing.

**The mechanism in equations.** For the binary marginal $p=\sum_k \pi_k \sigma(\ell_k)$ with
$M=\sum_j \pi_j p_j$, the gradient reaching the retrieval score is

$$\frac{\partial \mathcal{L}}{\partial s_k} \;=\; \pi_k \,\frac{M - p_k}{M}$$

Once the softmax saturates, $\pi$ is one-hot, so $M = p_{\text{top}}$ and **the gradient is exactly zero**.
The collapse is self-reinforcing and self-sealing. It is driven by there being *no temperature parameter*:
`retrieval_scoring.py:80–81` computes a raw dot product, so $\|q\|$ acts as an implicit inverse temperature —

$$\pi_k = \mathrm{softmax}_k(\|q\|\,\hat q^\top k_k)$$

and the loss can be reduced by inflating $\|q\|$ rather than by improving document *ordering*. Observed:
$\|q\|$ 12 → 117 while `top1_top2_margin` → ~18, i.e. $\pi_2/\pi_1 \approx e^{-18}$.

**CORRECTION — the L2-vs-dot "metric mismatch" is NOT a contributor. I initially flagged it; it is
refuted.** The index is indeed `IndexFlatL2` (verified by reading `retrieval.faiss`: `metric_type=1`,
`ntotal=125,847`, `d=1024`), and `marginalized_score_similarity` does default to `dot`. But the Qwen3
keys **are** unit-norm — SentenceTransformer applies a `Normalize` module — measured over 5,000 sampled
docs: mean 1.000026, std 0.001386, range [0.9971, 1.0028]. Since

$$\arg\min_k \|q-k\|^2 = \arg\min_k \big(\|q\|^2 - 2q^\top k + \|k\|^2\big) = \arg\max_k q^\top k \quad\text{when } \|k\| \text{ is constant}$$

selection and weighting rank identically, **independently of $\|q\|$**. Measured over 200 random queries
against 20,000 real keys: **100.0% top-1 agreement, 99.2% top-4 set overlap.** So the collapse is caused
purely by the saturation/zero-gradient mechanism above, not by a metric mismatch. Note the query norm
still matters — not for *ranking*, but as the softmax temperature.

**Location.** `src/medrap/model/model.py:495–511`; `src/medrap/model/retrieval_scoring.py:76–86`;
`src/medrap/model/query_projection.py:162–175`; `conf/_train.yaml` (`marginalized_score_similarity: dot`).

**Fix.** Since selection and weighting already agree, the fix is entirely about the temperature and the
vanishing gradient: (a) set `marginalized_score_similarity=cosine`, which bounds scores to $[-1,1]$ and
removes $\|q\|$ as an implicit inverse temperature — **this is a one-word config change and is the single
cheapest intervention in this document**; (b) add an explicit learned temperature $\tau$ with
$\pi=\mathrm{softmax}(s/\tau)$, or LayerNorm the query projector output; (c) add an entropy regularizer on
$\pi$, or a small auxiliary contrastive retrieval loss, so the retriever keeps receiving gradient at
saturation.

**Discriminating experiment.** Run the `random_docs` ablation — **it exists in the code
(`retrievers.py:86–96`) and has never been run.** Train the marginalized arm with
`retriever.ablation_mode=random_docs` at N=16 and N=64.
*If H2 is true:* random documents perform **indistinguishably** from real retrieval, because real
retrieval was already returning one arbitrary constant document. *If false:* random docs are clearly worse.
Runtime ≈ 1.5h/run on one L40S; 2 runs. **This is the single highest-value experiment in this document** —
it separates "retrieval is broken" from "retrieval is useless here" and costs 3 GPU-hours.

---

### 3. The task has no headroom, because the prediction anchor is sampled over the patient's whole lifetime — **(C)**

**What it is.** `_sample_prediction_anchors` draws the anchor uniformly from
`[first_event + 1 day, last_event − 7 days]`. In MEDS, a subject's first event is `MEDS_BIRTH`, so the
window spans the patient's **entire life**, while their actual clinical events occupy a few days. Most
anchors therefore land in a dead zone decades from any hospital contact, the label is almost always
negative, and the only learnable signal is recency — which retrieval cannot improve.

**Why I'm confident (high on the measurements, medium on the full causal story).** Measured by me:

- Positive rates across all N: **0.048%–0.21%** (train/tuning/held_out). For N=16 tuning: **277 positives
  across 16 tasks over 25,059 patients**.
- `val/accuracy` is **exactly 0.999209** for *both* arms at *every* validation at N=16 — the all-negative
  predictor, unchanged for the whole run.
- The model input contains a mean of **~11 valid events** in a 256-slot window
  (`mask/train/valid_tokens_mean` ≈ 10.8–11.4; `pad_fraction` 0.74–0.81). Patients are being classified
  from ~11 tokens.

That these are the *most frequent codes in MIMIC-IV* (`LAB//50920//UNK`, `ED_REGISTRATION`,
`MEDICATION//START//Acetaminophen`) yet yield a 0.08% positive rate is itself the proof that the anchor,
not the code, is the problem.

**Caveat I must flag:** a subagent reported that "94.6% of scored samples are literally the same two
tokens." I could **not** verify that figure before stopping, and my own measurement (mean ≈11 valid
events) does not by itself establish it — a mean of 11 is compatible with either a modal-2 heavy tail or a
tighter distribution. **Treat the 94.6% number as unverified.** The direction is supported; the magnitude is not.

**The mechanism in equations.** If a subject has lifetime span $T$ (decades) and clinical activity
concentrated in a window of width $W$ (days), a uniform anchor lands within horizon $H$ of an occurrence
with probability of order

$$\Pr[\text{positive}] \;\approx\; \frac{W + H}{T} \;\sim\; \frac{7\ \text{days}}{25{,}000\ \text{days}} \approx 3\times10^{-4}$$

which is the observed order of magnitude. The conditional signal reduces to "is the anchor near live
activity," obtainable from the last `time_delta_days` alone.

**The corpus is also wrong for the task.** `MedRAG/textbooks` is general medical prose. The question is
"will *this patient* have lab 50920 redrawn in 7 days" — a scheduling/timing question about an individual
whose answer is not contained in any textbook. Formally, if $D$ is the retrieved document and $Y$ the label,
retrieval can only help if $I(Y; D \mid X_{\text{patient}}) > 0$; a corpus with no patient-specific content
and (per H2) no patient-dependent selection has $D \perp Y \mid X$, so that mutual information is zero
**by construction**.

**Location.** `src/medrap/preprocess/task_generation.py:120–147` (anchor sampling), `:164` (event scan —
note the `TIMELINE//` exclusion at `:81` applies only to *task-code selection*, never to the anchor bounds);
`data/retrieval_db/resolved_config.yaml` (corpus choice).

**Fix.** (a) Anchor on clinically meaningful times — admission, discharge, or a uniform draw restricted to
*days with at least one event* — rather than uniform over the lifetime. (b) Exclude `MEDS_BIRTH`/static
rows from the `first_event` bound. (c) Choose tasks with 1–20% prevalence (mortality, readmission,
long length-of-stay) instead of "most frequent code in a 7-day window." (d) If the scientific claim is
about retrieval over *patient* records, the corpus must be patient records, not textbooks.

**Discriminating experiment (cheap, CPU-only).** Fit a logistic regression on **one feature** — days from
anchor back to the last preceding event — for N=16's tasks, and compute its `val/auroc/mean` on the same
6,400-row prefix.
*If H3 is true:* that single scalar reaches ≈0.85–0.95, i.e. within noise of both deep models, proving
there is no headroom for retrieval. *If false:* it lands near 0.5–0.7 and the neural models are using real
structure. Runtime ≈ 15 min CPU.

---

### 4. The retrieval arm carries an optimization and regularization handicap the patient-only arm doesn't — **(B)**

**What it is.** The two arms differ in far more than "retrieval on/off." The marginalized arm must train
9.7M randomly-initialized document-token embeddings (50.1% of its 19.4M parameters) of which **<1% ever
receive a gradient** — because only ~4 documents are ever retrieved (H2) — while AdamW's weight decay
shrinks *all* 151,936 rows every step. It also carries dropout 0.1 on its only nonlinear path, where the
patient-only arm has none.

**Why I'm confident (medium).** Verified: `train/loss_epoch` is higher for the marginalized arm at **8 of 8
N** (mean ratio 1.097). `TokenFeatureRetrievalEncoder` is a bare `nn.Embedding(151936, 64)` with no padding
index (`retrieval_encoder.py:26`), and `_grouped_parameters` (`lightning_module.py:126–128`) assigns it to
the **decay** group because `ndim == 2` — so unretrieved rows are decayed without ever being trained.
A subagent measured, by diffing `epoch=0` against `progress=0.90` checkpoints, that every embedding row
shrank by the same factor (median norm ratio 0.9209, p1–p99 spread 3e-4), which is the signature of pure
weight decay with no gradient.

But I explicitly checked and **ruled out** gradient clipping as a confound: total grad norm decays to
~1e-3, far below `gradient_clip_val=1.0`, so clipping never fires. And AdamW is approximately invariant to
constant gradient rescaling, which weakens the "effective LR differs" story.

This hypothesis explains the *small consistent negative* Δ (−0.006 in the paired study), not the large per-N
swings. That is why it ranks 4th: it is real but small.

**Location.** `src/medrap/model/retrieval_encoder.py:14–56`; `src/medrap/train/lightning_module.py:118–136`;
`conf/fusion/cross_attention_perdoc_medium.yaml:8` (`dropout: 0.1`).

**Fix.** Set `dropout: 0.0` in the fusion for a controlled comparison; exclude the retrieval-encoder
embedding from weight decay (add an `ndim==2 and is_embedding` carve-out); or initialize it from a
pretrained embedding table (`PerDocMeanPooledRetrievalEncoder` already supports
`pretrained_model_name_or_path`).

**Discriminating experiment.** Train a "**null-retrieval control**": the *marginalized architecture*, same
parameter count and dropout, but with the document input replaced by a fixed constant vector.
*If H4 is true:* this control performs the same as the real retrieval arm, isolating the handicap as
architectural rather than informational. *If false:* real retrieval beats the constant-document control.
~1.5h/run, 2 N values.

---

### 5. The float32 logit clamp in binary marginalization creates ties that only the retrieval arm suffers — **(B)**

**What it is.** `_marginal_binary_logits` clamps the marginal probability to $[\epsilon, 1-\epsilon]$ with
$\epsilon$ = float32 eps, bounding returned logits to exactly ±15.942385. The patient-only arm takes the
unclamped `else:` branch. Clamped values become ties, and `binary_auroc_torch` resolves ties by average
rank, dragging AUROC toward 0.5 — an arm-asymmetric metric penalty.

**Why I'm confident this is NOT central (low centrality, high confidence in the ruling-out).** The
mechanism is real and arm-asymmetric — verified: `model.py:79–82`, bound confirmed numerically as
±15.942385; `metrics.py:127` applies `sigmoid` before ranking and fp32 sigmoid maps every logit above
~15.5 to the identical value. **But** three independent adversarial reviews scored its centrality at
0.0, 0.0, and 0.5 out of 10, because with a 0.08% positive rate the optimum bias is around
$\log(0.0008/0.9992) \approx -7.1$, nowhere near −15.9, so the clamp should rarely fire in practice.

**RESOLVED — this hypothesis is dead as an explanation.** I ran the check: across all 8 marginalized runs
the maximum `|logit|` ever observed is **13.417** (train) and 13.129 (val), against a clamp bound of
15.942385. Per-N maxima: 9.62, 10.24, 12.27, 11.42, 11.33, 13.42, 12.83, 13.35. **The clamp never fires.**
It is a genuine latent bug worth fixing before it *does* bite (lower prevalence, longer training, or
float16), but it contributes exactly nothing to the observed table.

**Location.** `src/medrap/model/model.py:79–82`.

**Fix.** Clamp in log-space or use a wider epsilon (e.g. 1e-6 in float32 terms is already the eps; better:
return `log_p_pos - log_p_neg` computed via `logsumexp`, exactly as `MultiTaskBCEMarginalizedLoss`
already does at `losses.py:222–228`, instead of round-tripping through a probability).

**Discriminating experiment — already run, result above.** Max `|logit|` = 13.42 < 15.94, so no clamping
occurred. Closed as latent-only.

---

## Consistency check

| Observation | H1 noise/readout | H2 collapse | H3 task/corpus | H4 handicap | H5 clamp |
|---|---|---|---|---|---|
| patient_only wins 6/8 on last-value | **supports** | supports | supports | supports | neutral |
| N=16 gap −0.19, largest anywhere | **supports** (0.239 within-run range) | neutral | neutral | refutes (too large) | neutral |
| longer training fixes N=16 (0.72→0.93) | **supports** (resampled oscillation) | neutral | neutral | supports weakly | neutral |
| k=10 fixes N=16 (0.72→0.88) | **supports** (same) | supports (k=10 also collapses, so not "more docs") | neutral | refutes | neutral |
| top-1 eval ≈ k=4 eval | neutral | **supports strongly** (effective_k already 1.0) | supports | neutral | neutral |
| win rate wanders 0%→60%, no trend | **supports** | neutral | neutral | refutes | neutral |
| held_out all-tasks-valid, tuning not | **supports** (full split vs 6,400-row prefix) | neutral | supports | neutral | neutral |
| paired N=25: Δ = −0.006 ± 0.011 | supports (small true effect) | **supports** | **supports** | **supports** | neutral |
| `val/accuracy` pinned at all-negative | neutral | neutral | **supports strongly** | neutral | neutral |
| collapse in all 32 runs | neutral | **supports strongly** | neutral | supports | neutral |

The ranking is driven by which hypotheses explain the *most* rows. H1 explains the per-N *pattern*;
H2 and H3 explain why the *true* effect is ~zero. They are complementary, not competing: **H1 explains why
the table looks dramatic, H2+H3 explain why the underlying effect is nil.**

---

## What I would do next, in order

1. **Run the `random_docs` ablation.** It's already implemented, never run, and 3 GPU-hours. It is the
   cleanest single discriminator between "retrieval is broken" and "retrieval is useless here."
2. **Fix the measurement before running any more architecture comparisons.** Remove `limit_val_batches`,
   report mean ± CI over ≥5 seeds, monitor and report the same checkpoint for both arms, add average
   precision. Without this, every future sweep will keep producing uninterpretable tables.
3. **Fix the task.** Anchor on clinical events rather than uniformly over the lifetime, and pick outcomes
   with 1–20% prevalence. A task where the base predictor sits at 0.999 accuracy and 0.9+ AUROC has no room
   for retrieval to demonstrate anything.
4. **Fix the retriever geometry.** Normalized keys + inner-product FAISS + `similarity=cosine`, plus an
   explicit temperature. Then re-check `effective_k_mean` and `top1_mode_frac` as *acceptance criteria* —
   if `effective_k` is not meaningfully above 1, the marginalization is not doing anything and the run
   should be treated as invalid.
5. **Match the corpus to the claim.** If the thesis is retrieval-augmented *pretraining over patient
   records*, retrieve from patient records. Textbooks cannot answer "will this patient's lab be redrawn."
6. **Then** re-run the patient_only vs marginalized comparison, with dropout matched and a
   constant-document control arm.

Add these as standing diagnostics/CI gates: valid-label fraction, `effective_k_mean`,
`top1_mode_frac`, `query/offdiag_cos_mean`, and positives-per-task in the evaluation subset.

---

## Appendix: key raw numbers

All computed by me from `/groups/mm6677_gp/hs3627/medrap-experiments/mimic_iv_sweep_frequent`.

**Label positive rates** (`data/tasks/n{N}/tasks/*.parquet`):

| N | train rows | train pos rate | tuning pos rate | tuning total positives |
|---|---|---|---|---|
| 1 | 200,773 | 0.00205 | 0.00172 | 43 |
| 16 | 200,773 | 0.00077 | 0.00069 | 277 |
| 128 | 200,773 | 0.00048 | 0.00053 | 1,701 |

**True positives in the 6,400-row validation prefix actually scored:** N=1 → 10, N=4 → 27, N=8 → 42,
N=16 → 81 (across *all* tasks, i.e. ~5 per task at N=16).

**Both arms read identical labels** — verified: `mt_labels_dir` is byte-identical in both
`resolved_config.yaml` files at N=1, 4, 8. (A subagent claimed three headline cells used *different* label
sets, inferring from a `val/accuracy` gap; I checked and **that claim is false** — the accuracy gap
reflects differing false positives, not differing labels.)

**Sequence occupancy:** `mask/train/valid_tokens_mean` ≈ 10.8–11.4 of 256; `pad_fraction` 0.74–0.81.

**Gradient norms (N=16, late training):** encoder ~1e-5–1e-3, head ~1e-3, **query_projector 5e-12**,
total ~1e-3 — far below `gradient_clip_val=1.0`, so **clipping never fires**.

**Paired N=25 duration×variance study** (from `../mimic_iv_sweep/README.md:646–672`, my recomputation):
7d mean Δ = −0.0055 ± 0.0086 (3/5 negative); 30d mean Δ = −0.0062 ± 0.0141 (3/5 negative);
pooled mean Δ = **−0.0059 ± 0.0110, t = −1.68, 6/10 negative**.

**H5 clamp check (resolved):** max `|logit|` across all 8 marginalized runs = 13.417 train / 13.129 val,
below the 15.942385 bound. The clamp never fired.

**FAISS / key geometry (resolved):** `retrieval.faiss` is `IndexFlatL2` (`metric_type=1`, ntotal=125,847,
d=1024). Keys are unit-norm (mean 1.000026, std 0.001386 over 5,000 sampled docs). L2-argmin vs dot-argmax
agree on **100.0%** of top-1 and **99.2%** of top-4 over 200 random queries. **No metric mismatch.**

**Validation-subset positives in the "better-designed" N=25 duration×variance study** (`../mimic_iv_sweep/
data/tasks_duration_variance/`): median **0–1 positives per task** in the 6,400-row prefix Lightning
actually scores (d30 draws 1–4: median 0, 1, 1, 1; max 8, 9, 6, 3). The README's "25/25 valid" was checked
on the *full* 25,059-row split, not on the scored prefix. That study is therefore **more** noise-limited
per-run than the frequent sweep, not less — its small draw-to-draw spread comes from the paired design
cancelling shared noise, not from a well-resolved metric.

**Losses actually used** (grep over all sweep scripts): `patient_only` → `multitask_binary_bce`;
`retrieval_n1248` → `multitask_binary_bce` with `marginalized_retrieval` left **false** (the
`marginalized_retrieval=true` string in that script appears only in comments — verified); all
`marginalized*` arms → `multitask_binary_bce_marginalized`. No training script overrides `seed` (all use
42) or `limit_val_batches` (all use 200).

**Open / unverified items** (stopped early): the "94.6% two-token" figure in H3; and a full forward-pass
ablation measuring how much the retrieved documents actually change the logits.
