import fitz  # PyMuPDF
import json
import re
import sys
from pathlib import Path

PAPERS_DIR = Path("data/papers")
PARSED_DIR = Path("data/parsed")
PARSED_DIR.mkdir(parents=True, exist_ok=True)

# Papers shorter than this are likely abstracts-only or broken — skip them
MIN_PAGES = 3
MIN_TEXT_CHARS = 500

# Reference section headings to detect and strip
REFERENCE_HEADINGS = [
    r"^\s*references\s*$",
    r"^\s*bibliography\s*$",
    r"^\s*works cited\s*$",
    r"^\s*\d+\.\s*references\s*$",   # e.g. "7. References"
]


def looks_like_reference_heading(line: str) -> bool:
    line_lower = line.strip().lower()
    for pattern in REFERENCE_HEADINGS:
        if re.match(pattern, line_lower, re.IGNORECASE):
            return True
    return False


def strip_references(text: str) -> tuple[str, bool]:
    """
    Remove everything from the References heading onward.
    Returns (cleaned_text, was_stripped).
    Only strips if the heading appears in the last 40% of the text
    (avoids false-positives like 'References' in the intro).
    """
    lines = text.split("\n")
    cutoff_line = int(len(lines) * 0.60)   # only look in last 40%

    for i in range(cutoff_line, len(lines)):
        if looks_like_reference_heading(lines[i]):
            return "\n".join(lines[:i]).strip(), True

    return text.strip(), False


def extract_metadata(doc: fitz.Document, arxiv_id: str) -> dict:
    """
    Extract title, authors, abstract, and year.
    Strategy: arXiv PDFs follow a consistent layout —
    title is on page 1 in large font, abstract follows.
    We use font-size heuristics for title, then regex for abstract.
    """
    meta = {
        "arxiv_id": arxiv_id,
        "title": "",
        "authors": "",
        "abstract": "",
        "year": "",
        "pages": doc.page_count,
    }

    # ── year from arxiv_id ───────────────────────────────
    # IDs like 2207.02337 → year 2022; old IDs like 0704.3575 → 2007
    id_digits = arxiv_id.replace(".", "")
    if len(id_digits) >= 4:
        yy = id_digits[:2]
        year_int = int(yy)
        meta["year"] = str(2000 + year_int if year_int <= 99 else 1900 + year_int)

    # ── extract page 1 text with font sizes ─────────────
    page0 = doc[0]
    blocks = page0.get_text("dict")["blocks"]

    # Collect all text spans with font sizes
    spans_with_size = []
    for block in blocks:
        if block.get("type") != 0:   # type 0 = text
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span["text"].strip()
                size = span["size"]
                if text:
                    spans_with_size.append((size, text))

    if spans_with_size:
        # Title = the largest-font text on page 1 (concatenate spans of that size)
        max_size = max(s[0] for s in spans_with_size)
        title_threshold = max_size * 0.85   # allow slight size variation
        title_parts = [s[1] for s in spans_with_size if s[0] >= title_threshold]
        meta["title"] = " ".join(title_parts)[:300]   # cap at 300 chars

    # ── extract full text for abstract + body ────────────
    full_text_pages = []
    for page_num in range(min(doc.page_count, 3)):   # first 3 pages for abstract
        full_text_pages.append(doc[page_num].get_text())
    header_text = "\n".join(full_text_pages)

    # Abstract: text between "Abstract" and the next section heading
    abstract_match = re.search(
        r"abstract[:\s]*\n?(.*?)(?:\n\s*\n|\n(?:1\.?\s+introduction|keywords|index terms))",
        header_text,
        re.IGNORECASE | re.DOTALL,
    )
    if abstract_match:
        meta["abstract"] = abstract_match.group(1).strip()[:2000]

    # Authors: usually on page 1, between title and abstract
    # Simple heuristic: second-largest font block on page 1
    if spans_with_size:
        max_size = max(s[0] for s in spans_with_size)
        second_sizes = [s for s in spans_with_size if s[0] < max_size * 0.85]
        if second_sizes:
            second_max = max(s[0] for s in second_sizes)
            author_threshold = second_max * 0.90
            author_parts = [s[1] for s in second_sizes if s[0] >= author_threshold]
            meta["authors"] = " ".join(author_parts)[:500]

    return meta


def parse_pdf(pdf_path: Path) -> dict | None:
    """
    Parse one PDF. Returns a result dict or None if the PDF should be skipped.
    """
    arxiv_id = pdf_path.stem

    # Skip if sidecar already exists (re-run safety)
    sidecar_path = PARSED_DIR / f"{arxiv_id}.json"
    if sidecar_path.exists():
        return {"status": "skipped_exists", "arxiv_id": arxiv_id}

    try:
        doc = fitz.open(str(pdf_path))
    except Exception as e:
        return {"status": "failed_open", "arxiv_id": arxiv_id, "error": str(e)}

    # Skip encrypted PDFs
    if doc.is_encrypted:
        doc.close()
        return {"status": "skipped_encrypted", "arxiv_id": arxiv_id}

    # Skip too-short papers
    if doc.page_count < MIN_PAGES:
        page_count = doc.page_count
        doc.close()
        return {"status": "skipped_too_short", "arxiv_id": arxiv_id, "pages": page_count}

    # Extract full text from all pages
    full_text_parts = []
    for page in doc:
        full_text_parts.append(page.get_text())
    raw_text = "\n".join(full_text_parts)

    # Skip if text is too short (scanned PDF with no OCR layer)
    if len(raw_text.strip()) < MIN_TEXT_CHARS:
        doc.close()
        return {"status": "skipped_scanned", "arxiv_id": arxiv_id, "chars": len(raw_text)}

    # Strip references
    clean_text, refs_stripped = strip_references(raw_text)

    # Extract metadata
    meta = extract_metadata(doc, arxiv_id)
    doc.close()

    # Build sidecar
    sidecar = {
        **meta,
        "full_text": clean_text,
        "references_stripped": refs_stripped,
        "char_count": len(clean_text),
        "status": "ok",
    }

    # Save sidecar
    with open(sidecar_path, "w", encoding="utf-8") as f:
        json.dump(sidecar, f, ensure_ascii=False, indent=2)

    return {"status": "ok", "arxiv_id": arxiv_id, "title": meta["title"][:60]}


def main():
    pdf_files = sorted(PAPERS_DIR.glob("*.pdf"))
    total = len(pdf_files)

    if total == 0:
        print("No PDFs found in data/papers/ — run ingest.py first.")
        sys.exit(1)

    print(f"Parsing {total} PDFs → JSON sidecars in data/parsed/\n")

    counts = {"ok": 0, "skipped_exists": 0, "skipped_encrypted": 0,
              "skipped_too_short": 0, "skipped_scanned": 0, "failed_open": 0}

    for i, pdf_path in enumerate(pdf_files, 1):
        result = parse_pdf(pdf_path)
        if result is None:
            continue

        status = result.get("status", "unknown")
        counts[status] = counts.get(status, 0) + 1

        if status == "ok":
            print(f"  [{i:3d}/{total}] ✓ {result['arxiv_id']} — {result.get('title','')}")
        elif status == "skipped_exists":
            pass   # silent on re-runs
        else:
            print(f"  [{i:3d}/{total}] ⚠ {result['arxiv_id']} — {status}")

    print(f"\n{'='*50}")
    print("PARSE SUMMARY")
    print(f"{'='*50}")
    print(f"  Parsed successfully : {counts['ok']}")
    print(f"  Already existed     : {counts.get('skipped_exists', 0)}")
    print(f"  Skipped (encrypted) : {counts.get('skipped_encrypted', 0)}")
    print(f"  Skipped (too short) : {counts.get('skipped_too_short', 0)}")
    print(f"  Skipped (scanned)   : {counts.get('skipped_scanned', 0)}")
    print(f"  Failed to open      : {counts.get('failed_open', 0)}")
    print(f"  Total JSON sidecars : {len(list(PARSED_DIR.glob('*.json')))}")
    print(f"{'='*50}")

    usable = counts['ok'] + counts.get('skipped_exists', 0)
    if usable < 100:
        print(f"\n⚠️  Only {usable} usable papers — consider re-running ingest.py.")
    else:
        print(f"\n✅  {usable} papers parsed and ready for chunking on Day 4.")


if __name__ == "__main__":
    main()