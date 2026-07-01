import pytest
from src.retrieve import retrieve_dense
from src.rerank import rerank


@pytest.mark.requires_chromadb
def test_reranker_improves_or_reorders_results():
    """Sanity check: reranking a dense top-10 down to top-3 returns
    well-formed chunks that exist in the original dense results."""
    query = "What is the privacy-utility tradeoff when epsilon < 1 in DP-SGD?"

    dense_chunks = retrieve_dense(query, k=10)
    assert len(dense_chunks) >= 1

    reranked_chunks = rerank(query, dense_chunks, top_n=3)
    assert len(reranked_chunks) >= 1
    assert len(reranked_chunks) <= 3

    dense_ids = {(c.arxiv_id, c.chunk_index) for c in dense_chunks}
    for c in reranked_chunks:
        assert (c.arxiv_id, c.chunk_index) in dense_ids


def _print_comparison():
    """Manual inspection tool — not run by pytest.
    Run directly: python tests/test_reranker.py"""
    query = "What is the privacy-utility tradeoff when epsilon < 1 in DP-SGD?"
    print(f"Query: '{query}'\n")

    dense_chunks = retrieve_dense(query, k=10)

    print("─── DENSE TOP 3 (Without Reranking) ───")
    for i, c in enumerate(dense_chunks[:3], 1):
        print(f"[{i}] {c.arxiv_id} | Score: {c.similarity:.3f} | {c.title[:60]}")

    print("\n─── RERANKED TOP 3 (Cross-Encoder) ───")
    reranked_chunks = rerank(query, dense_chunks, top_n=3)

    for i, c in enumerate(reranked_chunks, 1):
        print(f"[{i}] {c.arxiv_id} | Score: {c.similarity:.3f} | {c.title[:60]}")
        orig_rank = next((idx for idx, orig in enumerate(dense_chunks)
                          if orig.arxiv_id == c.arxiv_id and orig.chunk_index == c.chunk_index), -1) + 1
        print(f"    ↳ Originally ranked #{orig_rank} by dense retrieval")


if __name__ == "__main__":
    _print_comparison()