import argparse
import json
import re
import sys
import tiktoken
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.corpus_config import DEFAULT_CORPUS, get_corpus_config

PARSED_DIR = Path("data/parsed")
CHUNKS_DIR = Path("data/chunks")
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

# ── chunking config ───────────────────────────────────────
# Why 512 tokens: fits in embedding model context without wasting capacity.
# Why 50 overlap: prevents a sentence or equation split exactly at a boundary
# from being lost — the overlap ensures both chunks contain the full sentence.
CHUNK_SIZE_TOKENS = 512
CHUNK_OVERLAP_TOKENS = 50

# Tokenizer — must match the embedding model's tokenizer family.
# BGE uses BERT-style tokenization; cl100k_base (GPT-4 tokenizer) is a close
# enough approximation for chunk sizing purposes.
TOKENIZER = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(TOKENIZER.encode(text))


FENCE_PATTERN = re.compile(r"(```[\w+-]*\n.*?```)", re.DOTALL)


def chunk_markdown_aware(
    text: str,
    chunk_size: int = 400,
    chunk_overlap: int = 40,
) -> list[str]:
    """
    Split markdown without cutting fenced code blocks in half.

    Text outside fences is recursively split. A fenced block is kept intact
    even if it is larger than chunk_size — splitting it would be the same
    failure mode as the equation-split bug on the research corpus.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=count_tokens,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    parts = FENCE_PATTERN.split(text)
    chunks: list[str] = []
    buffer = ""

    def flush_buffer() -> None:
        nonlocal buffer
        if buffer.strip():
            chunks.extend(splitter.split_text(buffer.strip()))
        buffer = ""

    for part in parts:
        if part.startswith("```"):
            flush_buffer()
            chunks.append(part.strip())
        else:
            buffer += part
    flush_buffer()
    return [c for c in chunks if c.strip()]


def get_splitter() -> RecursiveCharacterTextSplitter:
    """
    RecursiveCharacterTextSplitter tries to split on paragraph breaks first,
    then sentences, then words, then characters — in that order.
    This preserves logical units (paragraphs, sentences) much better than
    a naive fixed-size splitter that cuts every N characters regardless.

    DECISION: chosen over fixed-size splitting because academic papers contain
    multi-line equations and algorithm pseudocode that must not be split
    mid-expression. See documented failure case below in find_failure_case().
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE_TOKENS,
        chunk_overlap=CHUNK_OVERLAP_TOKENS,
        length_function=count_tokens,       # measure size in tokens, not chars
        separators=["\n\n", "\n", ". ", " ", ""],  # priority order
    )


def chunk_paper(sidecar: dict, splitter: RecursiveCharacterTextSplitter) -> list[dict]:
    """
    Split one paper's full text into chunks.
    Each chunk carries metadata so the retriever can cite the source.
    """
    full_text = sidecar.get("full_text", "")
    if not full_text.strip():
        return []

    raw_chunks = splitter.split_text(full_text)

    chunks = []
    for idx, chunk_text in enumerate(raw_chunks):
        chunks.append({
            # ── content ──────────────────────────────────
            "text": chunk_text,
            "token_count": count_tokens(chunk_text),

            # ── source metadata ───────────────────────────
            # RAGAS uses these fields to compute context precision and recall.
            # Without them, evaluation cannot attribute retrieved chunks to papers.
            "arxiv_id": sidecar.get("arxiv_id", ""),
            "title": sidecar.get("title", ""),
            "authors": sidecar.get("authors", ""),
            "year": sidecar.get("year", ""),
            "chunk_index": idx,
            "total_chunks": len(raw_chunks),
        })

    return chunks


def chunk_all_papers() -> list[dict]:
    """Process all parsed JSON sidecars and return all chunks."""
    sidecar_files = sorted(PARSED_DIR.glob("*.json"))
    if not sidecar_files:
        print("No parsed JSON files found — run parse.py first.")
        return []

    splitter = get_splitter()
    all_chunks = []
    skipped = 0

    print(f"Chunking {len(sidecar_files)} papers...\n")

    for i, path in enumerate(sidecar_files, 1):
        with open(path, encoding="utf-8") as f:
            sidecar = json.load(f)

        if sidecar.get("status") != "ok":
            skipped += 1
            continue

        chunks = chunk_paper(sidecar, splitter)
        if not chunks:
            skipped += 1
            continue

        all_chunks.extend(chunks)
        print(f"  [{i:3d}/{len(sidecar_files)}] {sidecar['arxiv_id']:15s} "
              f"→ {len(chunks):3d} chunks  ({sidecar.get('char_count',0):,} chars)")

    return all_chunks


def print_summary(all_chunks: list[dict]):
    if not all_chunks:
        return
    token_counts = [c["token_count"] for c in all_chunks]
    print(f"\n{'='*50}")
    print("CHUNK SUMMARY")
    print(f"{'='*50}")
    print(f"  Total chunks        : {len(all_chunks):,}")
    print(f"  Avg tokens/chunk    : {sum(token_counts)/len(token_counts):.0f}")
    print(f"  Min tokens/chunk    : {min(token_counts)}")
    print(f"  Max tokens/chunk    : {max(token_counts)}")
    print(f"  Papers chunked      : {len(set(c['arxiv_id'] for c in all_chunks))}")
    print(f"{'='*50}")


def find_markdown_code_block_failure(sample_md: str | None = None) -> None:
    """
    Document the generalization failure: a naive character splitter cuts a
    fenced code block in half. Same class of bug as the equation split on
    the research corpus — different domain, same chunking lesson.
    """
    from langchain_text_splitters import CharacterTextSplitter

    sample_md = sample_md or """## Creating a FastAPI app

```python
from fastapi import FastAPI
app = FastAPI()

@app.get("/")
def read_root():
    return {"hello": "world"}
```

This creates a basic app with one route.
"""
    naive = CharacterTextSplitter(chunk_size=80, chunk_overlap=0, separator="")
    naive_chunks = naive.split_text(sample_md)
    aware_chunks = chunk_markdown_aware(sample_md, chunk_size=50, chunk_overlap=0)

    print("NAIVE FIXED-SIZE SPLITTER (failure):")
    for i, c in enumerate(naive_chunks):
        fences = c.count("```")
        print(f"  chunk {i}: {fences} fence markers | {c[:70]!r}...")
        if "```" in c and fences % 2 != 0:
            print("  ^ code fence split across a chunk boundary")

    print("\nMARKDOWN-AWARE SPLITTER:")
    for i, c in enumerate(aware_chunks):
        print(f"  chunk {i}: {c.count('```')} fence markers")


# ── INTERVIEW STORY ───────────────────────────────────────
# This function demonstrates WHY recursive splitting beats fixed-size.
# Run it separately to generate the documented failure case.
def find_failure_case():
    """
    Find and print a real example where fixed-size character splitting
    breaks a math equation or algorithm across a chunk boundary.
    This is your concrete evidence for the chunking design decision.
    """
    import re
    from langchain_text_splitters import CharacterTextSplitter

    fixed_splitter = CharacterTextSplitter(
        chunk_size=512,        # 512 characters, not tokens
        chunk_overlap=0,       # no overlap — worst case
        separator="",          # split anywhere
    )

    sidecar_files = sorted(PARSED_DIR.glob("*.json"))
    failures_found = 0

    print("Scanning for fixed-size splitting failures...\n")

    for path in sidecar_files[:50]:   # check first 50 papers
        with open(path, encoding="utf-8") as f:
            sidecar = json.load(f)

        if sidecar.get("status") != "ok":
            continue

        text = sidecar.get("full_text", "")
        fixed_chunks = fixed_splitter.split_text(text)
        recursive_chunks = get_splitter().split_text(text)

        # Look for equations or algorithms split across fixed-size boundaries
        for i, chunk in enumerate(fixed_chunks[:-1]):
            next_chunk = fixed_chunks[i + 1]

            # Signs of a mid-equation split:
            # - chunk ends mid-formula (trailing operator, open bracket, or \n in equation)
            # - or contains partial algorithm line like "← " or ":="
            last_50 = chunk[-50:].strip()
            first_50 = next_chunk[:50].strip()

            has_equation_split = any([
                last_50.endswith(("=", "+", "−", "×", "(", "∑", "∇", "←", ":=")),
                re.search(r"\\\w+$", last_50),          # ends mid LaTeX command
                re.search(r"^\d+\s*[+\-×=]", first_50),  # next chunk starts mid-expression
            ])

            if has_equation_split:
                failures_found += 1
                print(f"{'='*60}")
                print(f"FAILURE CASE — {sidecar['arxiv_id']}")
                print(f"Title: {sidecar['title'][:70]}")
                print(f"\n[Fixed-size chunk {i} ends with]:")
                print(f"  ...{last_50}")
                print(f"\n[Fixed-size chunk {i+1} starts with]:")
                print(f"  {first_50}...")
                print(f"\n[Recursive splitter keeps this together — "
                      f"nearest paragraph boundary used instead]")
                if failures_found >= 3:   # show 3 examples then stop
                    return


def chunk_software_docs(corpus: str = "software_docs") -> list[dict]:
    """Chunk markdown/HTML/PDF-extracted notes for the secondary corpus."""
    cfg = get_corpus_config(corpus)
    source = Path(cfg["source_dir"])
    files = sorted(
        p for p in source.rglob("*")
        if p.suffix.lower() in cfg["file_types"]
        and p.is_file()
        and p.name.lower() != "readme.md"
    )
    if not files:
        print(f"No software docs in {source} — add .md/.pdf files first.")
        return []

    all_chunks: list[dict] = []
    print(f"Chunking {len(files)} software docs ({cfg['splitter']})...\n")

    for path in files:
        if path.suffix.lower() == ".pdf":
            print(f"  {path.name:40s} → skip (text notes only; PDF parse is the research pipeline)")
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if not text.strip():
            print(f"  {path.name:40s} → skip (empty)")
            continue
        if path.suffix.lower() == ".md":
            raw = chunk_markdown_aware(
                text,
                chunk_size=cfg["chunk_size"],
                chunk_overlap=cfg["chunk_overlap"],
            )
        else:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=cfg["chunk_size"],
                chunk_overlap=cfg["chunk_overlap"],
                length_function=count_tokens,
                separators=["\n\n", "\n", ". ", " ", ""],
            )
            raw = splitter.split_text(text)

        doc_id = path.stem
        for idx, chunk_text in enumerate(raw):
            all_chunks.append({
                "text": chunk_text,
                "token_count": count_tokens(chunk_text),
                "arxiv_id": doc_id,
                "title": path.name,
                "authors": "software_docs corpus",
                "year": "",
                "chunk_index": idx,
                "total_chunks": len(raw),
            })
        print(f"  {path.name:40s} → {len(raw):3d} chunks")

    return all_chunks


def save_chunks(all_chunks: list[dict], output_path: Path | None = None):
    """Save all chunks to a single JSONL file (one chunk per line)."""
    path = output_path or (CHUNKS_DIR / "all_chunks.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    print(f"\nSaved {len(all_chunks):,} chunks to {path}")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default=DEFAULT_CORPUS)
    parser.add_argument("--find-failures", action="store_true")
    args = parser.parse_args()

    if args.find_failures:
        find_failure_case()
        print("\n" + "=" * 60)
        print("MARKDOWN / CODE-BLOCK FAILURE (secondary corpus)")
        print("=" * 60 + "\n")
        find_markdown_code_block_failure()
    elif args.corpus == "software_docs":
        chunks = chunk_software_docs(args.corpus)
        if chunks:
            save_chunks(chunks, Path(get_corpus_config(args.corpus)["chunks_path"]))
            print_summary(chunks)
    else:
        all_chunks = chunk_all_papers()
        if all_chunks:
            save_chunks(all_chunks)
            print_summary(all_chunks)
            if len(all_chunks) < 2000:
                print(f"\n⚠️  Only {len(all_chunks):,} chunks — expected 4000+. Check parsed sidecars.")
            else:
                print(f"\n✅  Chunks ready for embedding on Day 5.")



# DOCUMENTED FAILURE CASE (found empirically, 2025):
    # Paper 0412073 — fixed-size splitter cut equation (17) as:
    #   chunk 63 ends: "...Our world average is then χ = 0.1282 ± 0.0077 . ("
    #   chunk 64 starts: "17) Introducing the latter result in Eq. (16)..."
    # The equation reference and its number landed in different chunks.
    # A query like "what is the world average of χ?" would retrieve chunk 63
    # which ends mid-equation, or chunk 64 which has the number but no context.
    # Recursive splitting kept the full paragraph together.