# Scaling laws

The design, configurations and run list are in [`design.md`](design.md).

| File | What it does |
|---|---|
| [`run.sh`](run.sh) | one task-generation job (`tasks T DRAW`) or one training run plus its tests (`train CONFIG W D T DRAW`) |
| [`submit.sh`](submit.sh) | submits `tasks`, the `pilot` (one small run per configuration) or the full `grid` to SLURM; skips work already done |
| [`slices.sh`](slices.sh) | sets up a fresh clone on another SLURM account and submits the `slices` (#3 Qwen3-query slice + #4 control) |
| [`add_null_doc.py`](add_null_doc.py) | copies the textbook retrieval database with the null document appended |
| [`pyproject.toml`](pyproject.toml) | environment: MedRAP pinned to `c3700ab`, `faiss-gpu-cu12` on Linux (L40S only; no H100 kernels) |

## Running on the cluster

```bash
cd /groups/mm6677_gp/hs3627/medrap-experiments/scaling_laws
uv sync                                            # .venv with pinned MedRAP + faiss-gpu

# FAISS sees the GPU (expect 1)
srun --account=mm6677_gp --partition=gpu --gres=gpu:1 --time=00:10:00 \
    .venv/bin/python -c "import faiss; print(faiss.get_num_gpus())"

# Data: reuse the cohort built for mimic_iv_scaling_laws; textbook DB + null document
mkdir -p data
ln -s ../../mimic_iv_scaling_laws/data/cohort data/cohort
.venv/bin/python add_null_doc.py ../mimic_iv_sweep/data/retrieval_db data/retrieval_db

# Cache Qwen3-Embedding-0.6B (frozen token embeddings, Qwen3 query) before jobs need it
.venv/bin/python -c "from transformers import AutoModel, AutoTokenizer as T; \
    m = 'Qwen/Qwen3-Embedding-0.6B'; AutoModel.from_pretrained(m); T.from_pretrained(m)"

./submit.sh tasks    # CPU, ~25 array tasks
./submit.sh pilot    # after tasks: 4 small runs; check outputs/ and their run times
./submit.sh grid     # after the pilot looks right: 537 array jobs, 2,685 runs
                     # (= ./submit.sh main [#1, #2] + ./submit.sh slices [#3, #4])
```

Each run writes `outputs/<CONFIG>_W<W>_D<D>_T<T>/draw<d>` (training logs) and `..._test`
(test metrics), plus `..._test_random_doc` for #2 and #3. Checkpoints stay on the node's local
disk and are deleted when the job ends.
