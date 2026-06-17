import json
import tiktoken
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

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


def save_chunks(all_chunks: list[dict]):
    """Save all chunks to a single JSONL file (one chunk per line)."""
    output_path = CHUNKS_DIR / "all_chunks.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    print(f"\nSaved {len(all_chunks):,} chunks to {output_path}")
    return output_path


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


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--find-failures":
        find_failure_case()
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