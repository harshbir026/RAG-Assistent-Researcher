import arxiv
import os
import time
import json
from pathlib import Path
from dotenv import load_dotenv
import ssl
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


def main():
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


if __name__ == "__main__":
    main()