"""Encode the retrieval database once: a key index for search and contextual memory per document.

    python encode_docs.py RETRIEVAL_DB OUT_DIR [--memory 16] [--batch 64]

RETRIEVAL_DB is the hf_dataset database of scaling_laws/ (textbook passages, null document last):
columns doc_tokens (256 Qwen3 token ids), doc_attention_mask and doc_key_embeddings (1024-d).

Writes to OUT_DIR:
    keys.faiss       inner-product index over every passage except the null document
    memory.npy       float16 (n_docs, M, 1024): Qwen3-Embedding-0.6B last hidden states, mean over
                     M consecutive chunks of 256 / M tokens (padding excluded)
    memory_mask.npy  bool (n_docs, M): chunk has at least one real token
    meta.json        n_docs, null_index, M, model
"""

import argparse
import json
from pathlib import Path

import faiss
import numpy as np
import torch
from datasets import load_from_disk
from transformers import AutoModel

MODEL = "Qwen/Qwen3-Embedding-0.6B"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("db", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--memory", type=int, default=16)
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    db = load_from_disk(str(args.db))
    ds = db.with_format("numpy", columns=["doc_tokens", "doc_attention_mask"])
    keys = db.with_format("numpy", columns=["doc_key_embeddings"])["doc_key_embeddings"].astype(np.float32)
    n_docs, seq_len = len(ds), len(ds[0]["doc_tokens"])
    null_index = n_docs - 1
    if seq_len % args.memory:
        raise ValueError(f"{seq_len} tokens do not split into {args.memory} chunks")
    chunk = seq_len // args.memory

    index = faiss.IndexFlatIP(keys.shape[1])
    index.add(keys[:null_index])
    faiss.write_index(index, str(args.out / "keys.faiss"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(device).eval()
    hidden = model.config.hidden_size
    memory = np.lib.format.open_memmap(args.out / "memory.npy", mode="w+", dtype=np.float16,
                                       shape=(n_docs, args.memory, hidden))
    memory_mask = np.zeros((n_docs, args.memory), dtype=bool)

    with torch.no_grad():
        for start in range(0, n_docs, args.batch):
            rows = ds[start:start + args.batch]
            ids = torch.as_tensor(rows["doc_tokens"], device=device)
            mask = torch.as_tensor(rows["doc_attention_mask"], device=device).bool()
            states = model(input_ids=ids, attention_mask=mask).last_hidden_state.float()
            states = states * mask.unsqueeze(-1)
            states = states.view(len(ids), args.memory, chunk, hidden).sum(2)
            counts = mask.view(len(ids), args.memory, chunk).sum(2)
            pooled = states / counts.clamp_min(1).unsqueeze(-1)
            memory[start:start + len(ids)] = pooled.half().cpu().numpy()
            memory_mask[start:start + len(ids)] = (counts > 0).cpu().numpy()
            if start // args.batch % 200 == 0:
                print(f"{start + len(ids):,} / {n_docs:,}", flush=True)

    memory[null_index] = 0
    memory_mask[null_index] = False
    memory.flush()
    np.save(args.out / "memory_mask.npy", memory_mask)
    (args.out / "meta.json").write_text(json.dumps(
        {"n_docs": n_docs, "null_index": null_index, "memory": args.memory, "model": MODEL}, indent=2))
    print(f"wrote {n_docs:,} documents to {args.out}")


if __name__ == "__main__":
    main()
