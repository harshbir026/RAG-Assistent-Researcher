import argparse
import os
import time
import json
import sys
from pathlib import Path
from dotenv import load_dotenv
import ssl

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.corpus_config import DEFAULT_CORPUS, get_corpus_config

ssl._create_default_https_context = ssl.create_default_context

load_dotenv()

# ── config ────────────────────────────────────────────────
OUTPUT_DIR = Path("data/papers")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEARCH_QUERIES = {
    "federated_learning": [
        "federated learning optimization",
        "federated averaging FedAvg",
        "personalized federated learning",
        "federated learning non-IID data",
    ],
    "privacy_ml": [
        "differential privacy machine learning",
        "privacy-preserving deep learning",
        "secure aggregation federated learning",
        "membership inference attacks",
    ],
    "deepfake_detection": [
        "deepfake detection neural network",
        "face forgery detection",
        "GAN-generated image detection",
        "synthetic media detection",
    ],
}

MAX_RESULTS_PER_QUERY = 25
# ──────────────────────────────────────────────────────────


def get_downloaded_ids() -> set:
    return {f.stem for f in OUTPUT_DIR.glob("*.pdf")}


def sanitize_id(arxiv_id: str) -> str:
    return arxiv_id.split("v")[0]


def download_domain(domain_name: str, queries: list) -> dict:
    import arxiv

    print(f"\n{'='*50}")
    print(f"Domain: {domain_name.upper()}")
    print(f"{'='*50}")

    already_downloaded = get_downloaded_ids()
    downloaded = 0
    skipped_duplicate = 0
    failed = 0
    failed_list = []

    client = arxiv.Client(
        page_size=50,
        delay_seconds=3,
        num_retries=3,
    )

    for query in queries:
        print(f"\n  Query: '{query}'")
        search = arxiv.Search(
            query=query,
            max_results=MAX_RESULTS_PER_QUERY,
            sort_by=arxiv.SortCriterion.Relevance,
        )

        for paper in client.results(search):
            arxiv_id = sanitize_id(paper.entry_id.split("/")[-1])

            if arxiv_id in already_downloaded:
                skipped_duplicate += 1
                continue

            pdf_path = OUTPUT_DIR / f"{arxiv_id}.pdf"
            try:
                import urllib.request
                urllib.request.urlretrieve(paper.pdf_url, str(pdf_path))
                already_downloaded.add(arxiv_id)
                downloaded += 1
                print(f"    ✓ {arxiv_id} — {paper.title[:60]}")
                time.sleep(0.5)
            except Exception as e:
                failed += 1
                failed_list.append(arxiv_id)
                print(f"    ✗ {arxiv_id} — FAILED: {e}")

    return {
        "domain": domain_name,
        "downloaded": downloaded,
        "skipped_duplicate": skipped_duplicate,
        "failed": failed,
        "failed_ids": failed_list,
    }


def print_summary(results: list):
    actual_count = len(list(OUTPUT_DIR.glob("*.pdf")))
    total_size_mb = sum(
        f.stat().st_size for f in OUTPUT_DIR.glob("*.pdf")
    ) / (1024 * 1024)
    total_downloaded = sum(r["downloaded"] for r in results)
    total_failed = sum(r["failed"] for r in results)

    print(f"\n{'='*50}")
    print("DOWNLOAD SUMMARY")
    print(f"{'='*50}")
    for r in results:
        print(f"  {r['domain']:25s}  downloaded: {r['downloaded']:3d}  failed: {r['failed']:2d}")
    print(f"{'─'*50}")
    print(f"  {'Total PDFs on disk':25s}  {actual_count}")
    print(f"  {'Total size':25s}  {total_size_mb:.1f} MB")
    print(f"  {'Failed downloads':25s}  {total_failed}")
    if total_failed > 0:
        print(f"  (some failures are normal — arXiv rate-limits rapid requests)")
    print(f"{'='*50}")

    if actual_count < 150:
        print(f"\n⚠️  Only {actual_count} PDFs — target is 150+. Re-run to retry.")
    else:
        print(f"\n✅  {actual_count} PDFs ready for parsing on Day 3.")


def ingest_research() -> None:
    print("Starting arXiv corpus download...")
    print(f"Output directory: {OUTPUT_DIR.resolve()}")

    all_results = []
    for domain_name, queries in SEARCH_QUERIES.items():
        result = download_domain(domain_name, queries)
        all_results.append(result)

    print_summary(all_results)

    manifest = {
        "total_pdfs": len(list(OUTPUT_DIR.glob("*.pdf"))),
        "domains": all_results,
    }
    manifest_path = Path("data/manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved to {manifest_path}")


def ingest_software_docs(corpus: str = "software_docs") -> None:
    """Inventory the secondary corpus. Notes already live in data/software_docs/."""
    cfg = get_corpus_config(corpus)
    source = Path(cfg["source_dir"])
    files = sorted(
        p for p in source.rglob("*")
        if p.suffix.lower() in cfg["file_types"]
        and p.is_file()
        and p.name.lower() != "readme.md"
        and p.stat().st_size > 0
        and p.suffix.lower() != ".pdf"
    )
    print(f"Software docs corpus at {source.resolve()}")
    print(f"  files: {len(files)}")
    for path in files:
        print(f"    {path.name}")
    if len(files) < 25:
        print(f"\n⚠️  Only {len(files)} docs — target is ~30 teaching notes.")
    else:
        print(f"\n✅  {len(files)} docs ready. Next: python src/chunk.py --corpus {corpus}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default=DEFAULT_CORPUS)
    args = parser.parse_args()
    if args.corpus == "software_docs":
        ingest_software_docs(args.corpus)
    elif args.corpus == "research":
        ingest_research()
    else:
        raise SystemExit(f"Unknown corpus '{args.corpus}'")


if __name__ == "__main__":
    main()