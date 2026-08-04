"""Measure FAISS search + payload materialization cost vs k, cached vs uncached."""

import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/groups/mm6677_gp/zzw2102/MedRAP/src")

from medrap.model.retrievers import load_hf_dataset_retriever  # noqa: E402

DB = "/groups/mm6677_gp/zzw2102/medrap-experiments/mimic_iv_ablation_ladder/data/retrieval_db"
B = 32
D = 1024


def timeit(fn, reps=5):
    fn()  # warm
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return min(ts), sum(ts) / len(ts)


def main():
    print("=== uncached retriever (cache_payloads=false, as configured) ===")
    t0 = time.perf_counter()
    r = load_hf_dataset_retriever(
        dataset_path=DB,
        index_name="retrieval",
        doc_tokens_column="doc_tokens",
        doc_attention_mask_column="doc_attention_mask",
        k=4,
        doc_ids_column=None,
        doc_key_embeddings_column="doc_key_embeddings",
        index_path=f"{DB}/retrieval.faiss",
    )
    print(f"load time {time.perf_counter() - t0:.1f}s  N_docs={r._dataset_num_rows}")

    q = torch.randn(B, 1, D)
    for k in (4, 32, 128, 256):
        r.k = k
        qn = q.detach().to(torch.float32).cpu().reshape(B, D).numpy()
        s_min, s_avg = timeit(lambda: r._dataset.search_batch("retrieval", qn, k=r.k), reps=5)
        scores, rows = r._search_index(q)
        m_min, m_avg = timeit(
            lambda: r._materialize_output(
                row_indices=rows, scores=scores, output_device=torch.device("cpu")
            ),
            reps=3,
        )
        print(
            f"k={k:4d}  faiss_search_batch(B=32): min {s_min * 1000:8.1f} ms | "
            f"payload_materialize (uncached HF row fetch): min {m_min * 1000:8.1f} ms"
        )

    # single-thread FAISS to show the brute-force cost structure
    import faiss

    nthreads = faiss.omp_get_max_threads()
    print(f"(faiss omp threads = {nthreads})")

    print("\n=== cached retriever (cache_payloads=true, CPU cache) ===")
    t0 = time.perf_counter()
    rc = load_hf_dataset_retriever(
        dataset_path=DB,
        index_name="retrieval",
        doc_tokens_column="doc_tokens",
        doc_attention_mask_column="doc_attention_mask",
        k=4,
        doc_ids_column=None,
        doc_key_embeddings_column="doc_key_embeddings",
        index_path=f"{DB}/retrieval.faiss",
        cache_payloads=True,
    )
    print(f"load+cache time {time.perf_counter() - t0:.1f}s")
    for name, t in (
        ("doc_tokens", rc._cached_doc_tokens),
        ("doc_attention_mask", rc._cached_doc_attention_mask),
        ("doc_key_embeddings", rc._cached_doc_key_embeddings),
    ):
        if t is not None:
            print(
                f"  cache {name:20s} {tuple(t.shape)} {t.dtype} "
                f"{t.element_size() * t.nelement() / 1024**3:.3f} GB"
            )
    for k in (4, 128, 256):
        rc.k = k
        scores, rows = rc._search_index(q)
        m_min, m_avg = timeit(
            lambda: rc._materialize_output(
                row_indices=rows, scores=scores, output_device=torch.device("cpu")
            ),
            reps=5,
        )
        print(f"k={k:4d}  payload_materialize (cached index_select): min {m_min * 1000:8.1f} ms")


if __name__ == "__main__":
    main()
