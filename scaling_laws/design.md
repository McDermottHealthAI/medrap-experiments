# Scaling laws: model design

Keep things simple and clean: start small, add detail only when it's needed.

## Architecture

![Model architecture](architecture.png)

Each box shows the shape of its output.

Source: [`architecture.mmd`](architecture.mmd) (Mermaid). After editing it, re-render with
`npx -y @mermaid-js/mermaid-cli -i architecture.mmd -o architecture.png -w 1800 -s 2 -b white`.

The top path is the **patient-only** model. The bottom path adds **retrieval**: the patient
encoder's output becomes a query, the query looks up K documents in an external database,
and the fusion combines the patient with each document before the head predicts.

## Variables

| Symbol | Meaning | Value |
|---|---|---|
| `B` | patients per batch | 32 |
| `S` | codes per patient | 256 |
| `W` | patient-encoder width | 4 to 256 (**varies**) |
| `T` | tasks | 5 to 2,500 (**varies**) |
| `K` | documents retrieved per patient | 4 |
| `S_doc` | tokens per document | 256 |

## Sizes

**Patient encoder** (the part that scales):

| Width `W` | Layers | Heads | Parameters |
|---|---|---|---|
| 4 | 1 | 1 | 262K |
| 8 | 1 | 1 | 525K |
| 16 | 1 | 1 | 1.05M |
| 32 | 1 | 1 | 2.11M |
| 64 | 1 | 1 | 4.23M |
| 128 | 1 | 1 | 8.52M |
| 128 | 2 | 4 | 8.65M |
| 256 | 4 | 8 | 18.9M |

**Retrieval side** (the same size in every run):

| Part | Size | Parameters |
|---|---|---|
| Retrieval encoder | token embedding, 151,936 × 64 | 9.72M |
| Fusion | 256-d, 2 layers, 8 heads | ~0.9M |
| Query | W → 1024 | 5K–263K |

## Configurations

Every configuration uses the architecture above. Each one changes only the **Query** box or
what comes out of the **External Document Database**, so a difference between two of them has
one cause.

| # | Configuration | Query | Documents in training | Documents at test | Question it answers | Status |
|---|---|---|---|---|---|---|
| 1 | Patient only | none (top path) | none | none | Baseline | Run (round 1, v2) |
| 2 | Retrieval, learned query | learned from the patient encoder | top 4 textbook passages | top 4 textbook passages | Does retrieval help, trained end to end? | Run (round 1; v2 pilot only) |
| 3 | Retrieval, Qwen3 query | patient's last 32 codes as text, embedded by a frozen Qwen3 model | top 4 textbook passages | top 4 textbook passages | Same, with a query that can't collapse onto a few documents | Run (round 1 controls); v2 grid running |
| 4 | Random docs | learned | 4 random passages | 4 random passages | Is it the *relevant* text that helps, or any text? | Run (round 1 controls) |
| 5 | Null docs | learned | 1 empty document | 1 empty document | What do the extra layers add with no knowledge at all? | Run (round 1 controls); v2 grid running |
| 6 | Random docs at test only | as #2 or #3 | top 4 textbook passages | 4 random passages | Does a trained model rely on its documents? | Not run (proposed) |
| 7 | Null docs at test only | as #2 or #3 | top 4 textbook passages | 1 empty document | Same, with the documents removed entirely | Not run (proposed) |

**Comparisons we read off these** (positive = the first one is better):

| Comparison | Name | What it measures |
|---|---|---|
| 5 − 1 | Architecture value | the extra fusion layers and head alone |
| 2 or 3 − 5 | Document value | what the documents' content adds |
| 2 or 3 − 1 | Total value | architecture + documents |
| 2 or 3 − 4 | Relevance value | relevant text vs. arbitrary text |
| 2 or 3 − 6 | Reliance | how much a trained model depends on getting the right documents |

**To decide:**

- Which query is the default retrieval arm: learned (#2) or Qwen3 (#3)? The v2 grid uses #3.
- Should the controls (#4, #5) use the learned query, the Qwen3 query, or both? So far
  they use the learned one.
- Add #6 and #7? They need no extra training, only a second test pass before each run's
  checkpoint is deleted.
