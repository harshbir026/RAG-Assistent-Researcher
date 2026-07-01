import pytest
from src.agent import classify_query, decompose_query

def test_simple_query_classified_correctly():
    result = classify_query("What is differential privacy?")
    assert result in ["simple", "complex"]

def test_multihop_query_classified_complex():
    q = "Compare FedAvg and pFedMe on non-IID data and explain the privacy tradeoffs"
    result = classify_query(q)
    assert result == "complex"

def test_decompose_returns_multiple_subquestions():
    q = "Compare FedAvg and pFedMe on non-IID data"
    subqs = decompose_query(q)
    assert isinstance(subqs, list)
    assert len(subqs) >= 2
