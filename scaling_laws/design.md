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

**Database.** The external database always holds the medical-textbook passages **plus a null
(empty) document**, so the model can retrieve "nothing" when no passage helps. There is no
separate null-docs configuration.

**Training.** Each configuration changes only the **Query** box:

| # | Configuration | Query | Status |
|---|---|---|---|
| 1 | Patient only | none (top path) | Run (round 1, v2) |
| 2 | Retrieval, learned query | learned from the patient encoder | Round 1 and v2 pilot, without the null document |
| 3 | Retrieval, Qwen3 query | patient's last 32 codes as text, embedded by a frozen Qwen3 model | v2 grid running, without the null document |

**Testing.** Every retrieval model is tested twice, with no extra training:

| Test | Documents |
|---|---|
| Top documents (always) | the top 4 the query retrieves |
| Random documents | 4 random documents from the database |

**Comparisons** (positive = the first one is better):

| Comparison | Name |
|---|---|
| 2 or 3, top documents − 1 | Total value of retrieval |
| Top documents − random documents, same model | Reliance on the retrieved documents |

**To decide:**

- Which query is the default: learned (#2) or Qwen3 (#3)? The v2 grid uses #3.
- The null document's key. An all-zero key always scores 0, so it is among the top 4 only when
  fewer than 4 passages score above 0. A learned key would let training decide when "nothing" wins.
- Runs so far used a database without the null document, so #2 and #3 need rerunning with it.
