# Goal: retrieval that helps, without cheating

A model that **retrieves textbook passages for a patient, always reads them, and predicts the 1,000
tasks better than the same model without retrieval**. Work proceeds in rounds: one hypothesis per
round, the smallest change that tests it, paired runs, a logged decision. The log is
[`EXPERIMENTS.md`](EXPERIMENTS.md).

## Success criteria

Measured on paired runs (same tasks, patients, seed): `T` = 1,000; `W` 64x2x4 and 256x4x8;
`D` 10K and 153K; 3 draws. `Δ = L(patient only) − L(retrieval)` on the test set; positive means
retrieval is better.

| | Criterion | Test |
|---|---|---|
| G1 | **No harm** | `Δ` ≥ 0, or its 95% interval includes 0, in all 4 cells |
| G1+ | **Not worse at large `D`** | at `D` 153K, the point estimate of `Δ` is ≥ 0 for both `W`, not only its interval; every change is checked against this |
| G2 | **Uses the documents** | the same model is better with its retrieved documents than with random ones (`randomK` − `topK` > 0, interval excludes 0), and better than the same architecture trained on random documents |
| G3 | **Helps** | `Δ` > 0 with the interval excluding 0 in at least 2 of the 4 cells |

G1, G1+ and G2 are required before any scaling-law runs. G3 is the goal.

## Rules (what counts as cheating)

- **Documents are always read.** No gate, switch or path that lets the model turn retrieval off as
  a whole. The null document stays (REALM has one), but it is one document among K.
- **Same retrieval in training and at test** (same K, same search).
- **No test-set tuning.** Settings are chosen on validation loss; the test set is only reported.
- **Equal tuning for the baseline.** Patient only gets the same learning-rate search as retrieval
  (Stage 0: 3e-4 for 256x4x8, 3e-3 for 64x2x4).
- **No information beyond the patient record and the corpus.** The exception is the positive
  control below, which is labelled as such and never reported as a result.

## Exploration plan

Cheapest first. Each step is kept only if it moves G1, G2 or G3. The order changes when a result says so.

| # | Hypothesis | Change | Status |
|---|---|---|---|
| 1 | Documents add noise as large as the patient signal from step 1 | zero-init cross-attention output | no effect |
| 2 | Cold start: the encoder learns while being fed noise | encoder from the patient-only checkpoint | **removes the harm** (G1 met vs the fair baseline); kept; no gain (G3) |
| 3 | The search finds the same passages for everyone | centered, unit-length keys | no effect |
| 4 | **Positive control**: can the reader use documents at all? | a synthetic document per patient that states some of its labels (leaked on purpose); if the model can't exploit it, the reader is broken, whatever the corpus | next |
| 5 | Cold-start retriever (REALM's ICT warm start) | pre-train the query to match the Qwen3 embedding of the patient's recent codes as text, then train end to end | |
| 6 | Too few candidates for the retriever to learn from | K 4, 32, 128 with precomputed last-layer vectors | **running** (Round 6) |
| 7 | **Context-free documents** | richer document vectors, below | **first gain** at 153K/64x2x4 (Round 5); worse at D 10K; kept, as precomputed last layer |
| 8 | The reader is bolted on after the encoder | REALM-style joint reading: document vectors as extra tokens in the patient encoder's sequence | |
| 9 | The textbooks are the wrong knowledge | documents describing the patient's own codes (code definitions), then PubMed | |

## Richer documents

What a passage contributes, from least to most information per passage:

| | Document vectors | Per passage | Size | Status |
|---|---|---|---|---|
| E0 | Qwen3 word-piece embeddings, no context (v1) | 256 × 1024 | lookup | in use |
| E1 | Qwen3-Embedding-0.6B **last-layer hidden states**, mean over 16 chunks of 16 tokens | 16 × 1024 | 4.1 GB | built (`everyquery/data/docs/memory.npy`) |
| E2 | the same for 7 layers (4, 8, …, 28), attended as 7 × 16 tokens with a layer embedding, or a learned layer mix (ELMo) | 112 or 16 × 1024 | 29 GB | `encode_docs.py --layers` |
| E3 | last-layer hidden states for **all 256 tokens** (no chunk mean) | 256 × 1024 | 66 GB on disk; for small K only | |
| E4 | **code-aligned** passages: a passage per code (its definition) plus the textbook passages, all encoded as E1 | 16 × 1024 | ~+2 GB | |

E1 and E2 need a MedRAP retrieval encoder that reads precomputed vectors by retrieved row (design.md M3).

## Each round

1. Check running jobs; for any failure, find the cause before resubmitting.
2. Analyse finished runs with `ladder.py`: G1, G2, G3 for each new configuration.
3. Log the hypothesis, the change, the runs, the numbers and the decision in `EXPERIMENTS.md`.
4. Pick the next step from the plan, implement it (MedRAP changes on a branch, with doctests,
   off by default), smoke-test it at D 3K, then submit.
5. Stop and report when G3 is met, when the plan is exhausted, or before anything costing more than
   ~50 GPU-hours.
