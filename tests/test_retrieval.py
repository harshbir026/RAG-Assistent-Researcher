import pytest
from src.retrieve import retrieve_dense, QueryValidationError
from src.retrieve import retrieve_dense, retrieve_bm25, QueryValidationError


def test_fedavg_dense_retrieval():
    """A well-known FL term should retrieve relevant chunks mentioning it."""
    results = retrieve_dense("What is FedAvg?", k=3)
    assert len(results) >= 1
    combined = " ".join([r.text for r in results]).lower()
    assert any(w in combined for w in ["fedavg", "federated averaging", "mcmahan"])


def test_retrieval_returns_typed_chunks():
    """Results should be RetrievedChunk objects with expected fields, not raw dicts."""
    results = retrieve_dense("differential privacy epsilon", k=3)
    assert len(results) >= 1
    for r in results:
        assert hasattr(r, "text")
        assert hasattr(r, "similarity")
        assert hasattr(r, "arxiv_id")
        assert 0 <= r.similarity <= 1


def test_off_topic_query_returns_few_or_no_results():
    """An unrelated query should be filtered out by the similarity threshold."""
    results = retrieve_dense("best pizza recipes in Italy", k=5)
    assert len(results) <= 1  # threshold should filter most/all of these


def test_empty_query_raises_validation_error():
    """Day 13's validation should reject empty queries before retrieval runs."""
    with pytest.raises(QueryValidationError):
        retrieve_dense("")


def test_too_short_query_raises_validation_error():
    """Queries under 10 characters should be rejected."""
    with pytest.raises(QueryValidationError):
        retrieve_dense("hi")

def test_bm25_fedavg_retrieval():
    """BM25 should retrieve chunks mentioning FedAvg when queried directly."""
    results = retrieve_bm25("FedAvg federated averaging aggregation", k=5)
    assert len(results) >= 1
    combined = " ".join([r.text for r in results]).lower()
    assert any(w in combined for w in ["fedavg", "federated averaging"])


def test_bm25_returns_typed_chunks():
    """BM25 should return RetrievedChunk objects with the same fields as dense."""
    results = retrieve_bm25("differential privacy epsilon noise", k=3)
    assert len(results) >= 1
    for r in results:
        assert hasattr(r, "text")
        assert hasattr(r, "similarity")
        assert hasattr(r, "arxiv_id")
        assert 0 <= r.similarity <= 1