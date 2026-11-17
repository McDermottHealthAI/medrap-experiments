"""Precompute every passage's Qwen3 last-layer token vectors for DOCS=pre (GOAL.md, richer documents).

    python scaling_laws_v2/encode_last_layer.py RETRIEVAL_DB OUT.npy

Runs MedRAP's QwenHiddenStatesRetrievalEncoder (layers=last, the same code as DOCS=last) over each
row of the retrieval database in row order, null document included, and writes an
(N_rows, 256, 1024) float16 array (~66 GB for 125,848 rows); padding positions are zero.
"""

import sys

import numpy as np
import torch
from datasets import load_from_disk
from medrap.model.retrieval_encoder import QwenHiddenStatesRetrievalEncoder
from medrap.types import RetrieverOutput

BATCH = 64

db, out_path = load_from_disk(sys.argv[1]), sys.argv[2]
db = db.with_format("torch", columns=["doc_tokens", "doc_attention_mask"])
encoder = QwenHiddenStatesRetrievalEncoder("Qwen/Qwen3-Embedding-0.6B", layers="last").cuda()
n, s_doc = len(db), len(db[0]["doc_tokens"])
out = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float16, shape=(n, s_doc, encoder.embedding_dim))
for start in range(0, n, BATCH):
    rows = db[start : start + BATCH]
    retrieval = RetrieverOutput(
        doc_tokens=rows["doc_tokens"][:, None, None].cuda(),
        doc_attention_mask=rows["doc_attention_mask"][:, None, None].bool().cuda(),
    )
    with torch.no_grad():
        memory = encoder(retrieval).retrieval_memory[:, 0, 0]
    out[start : start + len(memory)] = memory.half().cpu().numpy()
    if start // BATCH % 200 == 0:
        print(f"{start + len(memory)}/{n}", flush=True)
out.flush()
print(f"wrote {out_path}: {out.shape}, finite: {bool(np.isfinite(out[-BATCH:]).all())}")
