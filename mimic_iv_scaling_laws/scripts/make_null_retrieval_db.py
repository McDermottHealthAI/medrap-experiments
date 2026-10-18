"""Build the content-free retrieval corpus for the null-docs control arm.

16 identical documents, each a single Qwen pad token (id 151643, attention on that one
position, padded to 256) with an all-zero 1024-d key embedding, saved as a HF dataset plus a
FAISS index in the layout ``retriever=hf_dataset`` loads. Every query retrieves the same
empty document, so the retrieval arm's fusion/head still run but see no knowledge. Same
structure as the null corpus earlier MIMIC ablations used.

Usage:  cd mimic_iv_scaling_laws && .venv/bin/python scripts/make_null_retrieval_db.py [data/retrieval_db_null]
"""

import sys
from pathlib import Path

from datasets import Dataset

N_DOCS, MAX_TOKENS, EMBED_DIM, PAD_ID = 16, 256, 1024, 151643


def main(out_dir: Path) -> None:
    ds = Dataset.from_dict(
        {
            "doc_text": [""] * N_DOCS,
            "doc_tokens": [[PAD_ID] * MAX_TOKENS] * N_DOCS,
            "doc_attention_mask": [[1] + [0] * (MAX_TOKENS - 1)] * N_DOCS,
            "doc_key_embeddings": [[0.0] * EMBED_DIM] * N_DOCS,
        }
    )
    ds.add_faiss_index(column="doc_key_embeddings", index_name="retrieval")
    out_dir.mkdir(parents=True, exist_ok=True)
    ds.save_faiss_index("retrieval", str(out_dir / "retrieval.faiss"))
    ds.drop_index("retrieval")
    ds.save_to_disk(str(out_dir))
    print(f"wrote {N_DOCS} null docs to {out_dir}")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "data/retrieval_db_null"))
