"""Smoke: software_docs collection is isolated and answers domain questions."""

import pytest

from src.corpus_config import get_corpus_config
from src.retrieve import corpus_size, retrieve_hybrid


pytestmark = pytest.mark.requires_chromadb


def test_software_collection_is_populated():
    n = corpus_size("software_docs")
    if n == 0:
        pytest.skip("software_docs Chroma collection not embedded yet")
    assert n > 0
    assert get_corpus_config("software_docs")["collection_name"] == "software_docs"


def test_software_golden_queries_retrieve_right_docs():
    if corpus_size("software_docs") == 0:
        pytest.skip("software_docs Chroma collection not embedded yet")

    cases = [
        ("How do you define a path parameter with type validation in FastAPI?", "fastapi_path_parameters"),
        ("What's the difference between a 401 and 403 response per RFC 7231?", "rfc7231"),
        ("How does FastAPI's dependency injection system work?", "fastapi_dependency"),
    ]
    for query, expected_stem in cases:
        chunks = retrieve_hybrid(query, k=5, corpus="software_docs")
        assert chunks, query
        ids = " ".join(c.arxiv_id for c in chunks)
        assert expected_stem in ids, f"{query} retrieved {ids}"
