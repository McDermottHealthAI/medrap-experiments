"""Copy a retrieval database with one null (empty) document appended as its last row.

The null document is a single Qwen pad token (id 151643, attention on that position only) with an
all-zero key embedding. ``retriever.append_null_doc=true`` (MedRAP#122) then adds this last row to
every query's retrieved documents (and skips it in search, so it is never returned twice). Its zero
key is added to a copy of the FAISS index so the index and the dataset have the same rows.

Usage:  python scaling_laws/add_null_doc.py SOURCE_DB OUTPUT_DB
"""

import sys
from pathlib import Path

import numpy as np
from datasets import Dataset, concatenate_datasets, load_from_disk

PAD_ID = 151643


def main(src: Path, dst: Path) -> None:
    ds = load_from_disk(str(src))
    first = ds[0]
    null = {name: ("" if isinstance(value, str) else value) for name, value in first.items()}
    null["doc_tokens"] = [PAD_ID] * len(first["doc_tokens"])
    null["doc_attention_mask"] = [1] + [0] * (len(first["doc_attention_mask"]) - 1)
    null["doc_key_embeddings"] = [0.0] * len(first["doc_key_embeddings"])
    if "doc_ids" in null:
        null["doc_ids"] = max(ds["doc_ids"]) + 1
    out = concatenate_datasets([ds, Dataset.from_list([null], features=ds.features)])
    out.save_to_disk(str(dst))

    ds.load_faiss_index("retrieval", str(src / "retrieval.faiss"))
    index = ds.get_index("retrieval").faiss_index
    index.add(np.zeros((1, index.d), dtype=np.float32))
    ds.save_faiss_index("retrieval", str(dst / "retrieval.faiss"))
    print(f"wrote {len(ds)} documents + 1 null document to {dst}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
