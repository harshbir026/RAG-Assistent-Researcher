"""Markdown-aware chunker: code fences stay intact; naive splitter does not."""

from langchain_text_splitters import CharacterTextSplitter

from src.chunk import chunk_markdown_aware

SAMPLE_MD = """
## Creating a FastAPI app

```python
from fastapi import FastAPI
app = FastAPI()

@app.get("/")
def read_root():
    return {"hello": "world"}
```

This creates a basic app with one route.
"""


def test_chunker_preserves_code_blocks():
    """The one real engineering change needed for the second domain."""
    chunks = chunk_markdown_aware(SAMPLE_MD, chunk_size=50, chunk_overlap=0)
    fence_chunks = [c for c in chunks if "```python" in c]
    assert fence_chunks, "expected the fenced example to appear in some chunk"
    for chunk in fence_chunks:
        assert chunk.count("```") % 2 == 0, "code block split across chunks"


def test_naive_splitter_breaks_code_block():
    """Document the failure the markdown-aware splitter exists to prevent."""
    naive = CharacterTextSplitter(chunk_size=80, chunk_overlap=0, separator="")
    chunks = naive.split_text(SAMPLE_MD)
    split_fences = [c for c in chunks if "```" in c and c.count("```") % 2 != 0]
    assert split_fences, "naive splitter should cut a fence in half on this sample"
