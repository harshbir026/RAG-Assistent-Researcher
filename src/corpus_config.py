"""
Corpus-specific settings so the same pipeline (chunker, retriever, agent)
can point at different domains via --corpus. Collections stay separate:
research papers never share an index with software docs.
"""

from __future__ import annotations

from pathlib import Path

CORPUS_CONFIGS = {
    "research": {
        "source_dir": "data/papers/",
        "parsed_dir": "data/parsed/",
        "file_types": [".pdf"],
        "chunk_size": 512,
        "chunk_overlap": 50,
        "splitter": "recursive_text",
        "collection_name": "research_papers",
        "chunks_path": "data/chunks/all_chunks.jsonl",
        "bm25_path": "data/bm25_index.pkl",
    },
    "software_docs": {
        "source_dir": "data/software_docs/",
        "parsed_dir": "data/software_docs/",
        "file_types": [".md", ".pdf", ".html"],
        "chunk_size": 400,
        "chunk_overlap": 40,
        "splitter": "markdown_aware",
        "collection_name": "software_docs",
        "chunks_path": "data/chunks/software_docs.jsonl",
        "bm25_path": "data/bm25_index_software_docs.pkl",
    },
}

DEFAULT_CORPUS = "research"


def get_corpus_config(name: str | None = None) -> dict:
    key = name or DEFAULT_CORPUS
    if key not in CORPUS_CONFIGS:
        valid = ", ".join(CORPUS_CONFIGS)
        raise ValueError(f"Unknown corpus '{key}'. Choose one of: {valid}")
    return CORPUS_CONFIGS[key]


def corpus_path(name: str, field: str) -> Path:
    return Path(get_corpus_config(name)[field])
