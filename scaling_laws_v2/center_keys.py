"""Copy the retrieval database with centered, unit-length document keys.

    python scaling_laws_v2/center_keys.py SOURCE_DB OUTPUT_DB

The Qwen3 keys share a large common direction (mean pairwise cosine 0.80), so most queries rank the
same passages first. Subtracting the mean key and normalizing each key to length 1 removes it. The
null document (last row, all-zero key) stays zero. The FAISS inner-product index is rebuilt over every
row, null included, as add_null_doc.py does.
"""

import sys
from pathlib import Path

import faiss
import numpy as np
from datasets import load_from_disk


def mean_cosine(keys: np.ndarray, n: int = 2000, seed: int = 0) -> float:
    sample = keys[np.random.default_rng(seed).choice(len(keys), n, replace=False)]
    unit = sample / np.linalg.norm(sample, axis=1, keepdims=True)
    cos = unit @ unit.T
    return float((cos.sum() - n) / (n * (n - 1)))


def main(src: Path, dst: Path) -> None:
    ds = load_from_disk(str(src))
    keys = ds.with_format("numpy", columns=["doc_key_embeddings"])[:]["doc_key_embeddings"].astype(np.float32)
    docs = keys[:-1]
    centered = docs - docs.mean(axis=0, keepdims=True)
    centered /= np.linalg.norm(centered, axis=1, keepdims=True)
    new = np.concatenate([centered, np.zeros_like(keys[-1:])])
    print(f"mean pairwise cosine: {mean_cosine(docs):.3f} before, {mean_cosine(centered):.3f} after")

    out = ds.remove_columns("doc_key_embeddings").add_column("doc_key_embeddings", list(new))
    out = out.cast_column("doc_key_embeddings", ds.features["doc_key_embeddings"])
    out.save_to_disk(str(dst))
    index = faiss.IndexFlatIP(new.shape[1])
    index.add(new)
    faiss.write_index(index, str(dst / "retrieval.faiss"))
    print(f"wrote {len(out)} documents to {dst}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
