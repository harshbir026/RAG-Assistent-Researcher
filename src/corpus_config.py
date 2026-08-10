"""
src/corpus_config.py
Defines corpus-specific settings so the same pipeline (chunker, retriever,
agent) can point at different domains via a config flag, proving the
architecture generalizes rather than being hardcoded to one dataset.
"""

CORPUS_CONFIGS = {
    "research": {
        "source_dir": "data/papers/",
        "file_types": [".pdf"],
        "chunk_size": 512,
        "chunk_overlap": 50,
        "splitter": "recursive_text",
        "collection_name": "research_papers",
    },
    "software_docs": {
        "source_dir": "data/software_docs/",
        "file_types": [".md", ".pdf", ".html"],
        "chunk_size": 400,
        "chunk_overlap": 40,
        "splitter": "markdown_aware",  # respects code blocks — built Day 37
        "collection_name": "software_docs",
    },
}

DEFAULT_CORPUS = "research"
