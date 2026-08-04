"""Build a content-free ("null") retrieval corpus that trained checkpoints can be evaluated against.

WHAT A NULL CORPUS IS FOR
    Every retrieval-augmented run in this experiment (sweep_marginalized_binary_n1248.sh and
    friends) fuses k=4 MedRAG/textbooks passages into the patient representation. The obvious
    question is how much of the measured AUROC actually comes from those passages. Swapping the
    corpus for one whose documents carry no information -- a single placeholder token, an
    all-zeros key embedding, every row identical -- answers that at inference time, on the
    already-trained weights, without retraining anything: whatever AUROC survives is what the
    patient encoder alone was carrying, and the gap to the real-corpus eval is the retrieval
    contribution.

    This is a strictly stronger ablation than retriever.ablation_mode=random_docs, which still
    hands the model real textbook prose (just the wrong prose, so the fusion stage still sees
    plausible medical language and the score distribution still varies per document). Here the
    retrieved set is constant, contentless, and its differentiable retrieval scores are exactly
    tied, so the marginalization over documents degenerates to a plain average of K identical
    per-document predictions.

WHY THIS CAN BE HAND-BUILT (no Qwen, no GPU)
    The retriever never re-embeds anything at inference. HFDatasetRetriever.retrieve() only
    (a) searches the saved FAISS index with the model's projected query and (b) reads the stored
    doc_tokens / doc_attention_mask / doc_key_embeddings columns of the returned rows. Nothing
    downstream re-runs the Qwen3-Embedding-0.6B tokenizer or encoder that
    medrap-prepare-retrieval-dataset used to write the real corpus. So the corpus is just an
    on-disk HF dataset plus a FAISS index, and this script writes exactly what the final stage of
    medrap.prepare_retrieval.preparation.prepare_retrieval_dataset writes (add_faiss_index ->
    save_faiss_index -> drop_index -> save_to_disk), only with fabricated rows.

HARD CONSTRAINTS (each one is a real failure mode, all asserted below)
    n_rows >= k
        HFDatasetRetriever._validate_dataset raises "k must be between 1 and the number of
        dataset rows" (retrievers.py:501-502). The default of 16 rows clears both the k=4 trained
        sweeps and the k=10 sweep (sweep_marginalized_binary_k10_n1248.sh) with headroom.
    d_ret == 1024
        The checkpoints were trained with query_projector=sequence_mean_1024, so the query side is
        1024-d. differentiable_retrieval_scores (retrieval_scoring.py:71-75) raises unless the
        query and the doc key embeddings align on the last dim, and the FAISS index dimension has
        to match the query it is searched with. 1024 is not negotiable for these checkpoints.
    >= 1 valid token per document
        PerDocCrossAttentionFusion (fusion.py:393) builds its MultiheadAttention key_padding_mask
        as ~flat_mask (fusion.py:539-546); CrossAttentionFusion uses the identical idiom at
        fusion.py:381-388. An all-padding document therefore gives an all-True key_padding_mask,
        i.e. cross-attention over zero valid keys, whose softmax over an entirely -inf row is NaN --
        which then poisons the whole batch through the marginalization. So each row gets one real
        placeholder token followed by padding, never zero valid tokens.
    placeholder_token < 151936
        retrieval_encoder=token_feature is built with vocab_size=151936; its nn.Embedding will
        index-error on anything at or above that. The default placeholder is 151643, Qwen3's
        <|endoftext|>/pad id -- the most content-free token in the vocabulary, and the same id the
        real corpus pads with.
    string_factory stays None
        That makes datasets.add_faiss_index build a plain untrained IndexFlat. Any trained index
        (IVF*, PQ) would run k-means over 16 identical vectors and fail (or silently produce a
        degenerate quantizer).
    column names / index name
        conf/retriever/hf_dataset.yaml reads doc_tokens, doc_attention_mask, doc_key_embeddings,
        index_name=retrieval, index_path=${dataset_path}/retrieval.faiss. doc_ids is deliberately
        omitted -- every sweep here passes retriever.doc_ids_column=null, and the real corpus
        stores string ids that would not survive the retriever's torch.as_tensor(..., long) cast
        anyway.
    s_doc == 256
        Matches the real corpus's max_length=256 (see resolved_config.yaml of
        ../mimic_iv_sweep/data/retrieval_db), so the fused tensor shapes are byte-for-byte what
        the checkpoints saw at training time.

Usage:
    cd mimic_iv_sweep_frequent
    python scripts/build_null_retrieval_db.py --output-dir data/retrieval_db_null
    python scripts/build_null_retrieval_db.py --output-dir data/retrieval_db_null --n-rows 32

    Runs on CPU in seconds; no SLURM job needed.
"""

import argparse
from pathlib import Path

import numpy as np
from datasets import Dataset

# Qwen3-Embedding-0.6B <|endoftext|>, which is also the pad id the real corpus pads with.
DEFAULT_PLACEHOLDER_TOKEN = 151643
# retrieval_encoder.vocab_size in every sweep script here.
RETRIEVAL_ENCODER_VOCAB_SIZE = 151936
# query_projector=sequence_mean_1024 output dim == real corpus embedding dim.
REQUIRED_D_RET = 1024
# Largest retriever.k used by any sweep in this experiment (sweep_marginalized_binary_k10_n1248.sh).
MAX_SWEEP_K = 10
# Floor on corpus size: clears MAX_SWEEP_K with headroom, and 16 rows cost nothing.
MIN_ROWS = 16
# max_length used to build the real corpus.
REAL_CORPUS_S_DOC = 256

DOC_TOKENS_COLUMN = "doc_tokens"
DOC_ATTENTION_MASK_COLUMN = "doc_attention_mask"
DOC_KEY_EMBEDDINGS_COLUMN = "doc_key_embeddings"
DOC_TEXT_COLUMN = "doc_text"
INDEX_NAME = "retrieval"


def build_null_dataset(n_rows: int, s_doc: int, d_ret: int, placeholder_token: int) -> Dataset:
    """Build the in-memory null corpus, validating every hard constraint as it goes."""
    if n_rows < MIN_ROWS:
        raise ValueError(
            f"--n-rows must be >= {MIN_ROWS} so HFDatasetRetriever._validate_dataset accepts every "
            f"retriever.k this experiment uses (largest is k={MAX_SWEEP_K}); got {n_rows}"
        )
    if s_doc < 1:
        raise ValueError(f"--s-doc must be >= 1; got {s_doc}")
    if d_ret != REQUIRED_D_RET:
        raise ValueError(
            f"--d-ret must be {REQUIRED_D_RET} to match query_projector=sequence_mean_1024 and the "
            f"real corpus's FAISS index dimension; got {d_ret}"
        )
    if not 0 <= placeholder_token < RETRIEVAL_ENCODER_VOCAB_SIZE:
        raise ValueError(
            f"--placeholder-token must be in [0, {RETRIEVAL_ENCODER_VOCAB_SIZE}) to be indexable by "
            f"retrieval_encoder=token_feature; got {placeholder_token}"
        )

    # One valid placeholder token, then padding -- never an all-padding document.
    doc_tokens = np.full((n_rows, s_doc), placeholder_token, dtype=np.int64)
    doc_attention_mask = np.zeros((n_rows, s_doc), dtype=np.int64)
    doc_attention_mask[:, 0] = 1

    # All-zeros keys: dot-product scores against any query are identically 0, so the softmax over
    # documents is exactly uniform and the marginalization carries no retrieval signal at all.
    # (Under marginalized_score_similarity=cosine, F.normalize's eps clamp keeps this at 0 too, so
    # this stays NaN-free either way.)
    doc_key_embeddings = np.zeros((n_rows, d_ret), dtype=np.float32)

    dataset = Dataset.from_dict(
        {
            DOC_TEXT_COLUMN: [""] * n_rows,
            DOC_TOKENS_COLUMN: doc_tokens.tolist(),
            DOC_ATTENTION_MASK_COLUMN: doc_attention_mask.tolist(),
            DOC_KEY_EMBEDDINGS_COLUMN: doc_key_embeddings.tolist(),
        }
    )

    assert len(dataset) == n_rows, f"expected {n_rows} rows, built {len(dataset)}"
    assert set(dataset.column_names) >= {
        DOC_TOKENS_COLUMN,
        DOC_ATTENTION_MASK_COLUMN,
        DOC_KEY_EMBEDDINGS_COLUMN,
    }, f"missing required columns, have {dataset.column_names}"
    assert doc_attention_mask.sum(axis=1).min() >= 1, "at least one row is all padding"
    assert doc_tokens.max() < RETRIEVAL_ENCODER_VOCAB_SIZE, "token id outside retrieval encoder vocab"
    assert doc_tokens.shape == (n_rows, s_doc), f"doc_tokens shape {doc_tokens.shape}"
    assert doc_key_embeddings.shape == (n_rows, d_ret), f"doc_key_embeddings shape {doc_key_embeddings.shape}"
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="destination for save_to_disk + retrieval.faiss, e.g. data/retrieval_db_null",
    )
    parser.add_argument(
        "--n-rows",
        type=int,
        default=MIN_ROWS,
        help=f"corpus size; must be >= {MIN_ROWS}, i.e. > every retriever.k used here (default: {MIN_ROWS})",
    )
    parser.add_argument(
        "--s-doc",
        type=int,
        default=REAL_CORPUS_S_DOC,
        help=f"tokens per document; matches the real corpus's max_length (default: {REAL_CORPUS_S_DOC})",
    )
    parser.add_argument(
        "--d-ret",
        type=int,
        default=REQUIRED_D_RET,
        help=f"key embedding dim; must stay {REQUIRED_D_RET} for these checkpoints (default: {REQUIRED_D_RET})",
    )
    parser.add_argument(
        "--placeholder-token",
        type=int,
        default=DEFAULT_PLACEHOLDER_TOKEN,
        help=f"the single valid token id in every document (default: {DEFAULT_PLACEHOLDER_TOKEN})",
    )
    args = parser.parse_args()

    dataset = build_null_dataset(
        n_rows=args.n_rows,
        s_doc=args.s_doc,
        d_ret=args.d_ret,
        placeholder_token=args.placeholder_token,
    )

    # Mirrors the final stage of medrap.prepare_retrieval.preparation.prepare_retrieval_dataset.
    # string_factory is left at its None default on purpose -- see the module docstring.
    dataset.add_faiss_index(column=DOC_KEY_EMBEDDINGS_COLUMN, index_name=INDEX_NAME, string_factory=None)

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    dataset.save_faiss_index(INDEX_NAME, output_path / f"{INDEX_NAME}.faiss")
    dataset.drop_index(INDEX_NAME)
    dataset.save_to_disk(str(output_path))

    index_path = output_path / f"{INDEX_NAME}.faiss"
    assert index_path.is_file(), f"{index_path} was not written"

    print("=== Null retrieval corpus written ===")
    print(f"  Output dir        : {output_path.resolve()}")
    print(f"  FAISS index       : {index_path.name} (index_name={INDEX_NAME!r}, untrained IndexFlat)")
    print(f"  Rows              : {args.n_rows}")
    print(f"  S_doc             : {args.s_doc}")
    print(f"  D_ret             : {args.d_ret}")
    print(f"  Placeholder token : {args.placeholder_token} (1 valid token/doc, rest padding)")
    print(f"  Columns           : {dataset.column_names}")
    print("")
    print("Point a sweep or eval at it with:")
    print(f"  retriever.dataset_path={output_path} retriever.doc_ids_column=null")


if __name__ == "__main__":
    main()
