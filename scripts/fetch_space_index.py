#!/usr/bin/env python3
"""Download the retrieval index into data/ if this checkout has no Chroma DB.

Used on Hugging Face Spaces (the index is gitignored). Set INDEX_DATASET
to your dataset id, default Harshbirr/researchmind-index.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

MARKER = Path("data/chroma_db/chroma.sqlite3")


def main() -> None:
    if MARKER.exists():
        print(f"Index already present at {MARKER} — skip download.")
        return

    repo_id = os.getenv("INDEX_DATASET", "Harshbirr/researchmind-index")
    print(f"Downloading retrieval index from dataset {repo_id} ...")

    from huggingface_hub import snapshot_download

    staging = Path("data/_hf_index")
    if staging.exists():
        shutil.rmtree(staging)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(staging),
    )

    mapping = {
        staging / "chroma_db": Path("data/chroma_db"),
        staging / "chunks": Path("data/chunks"),
        staging / "bm25_index.pkl": Path("data/bm25_index.pkl"),
        staging / "bm25_index_software_docs.pkl": Path("data/bm25_index_software_docs.pkl"),
    }
    for src, dest in mapping.items():
        if not src.exists():
            print(f"  missing in dataset (ok if unused): {src.name}")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            if dest.is_dir():
                shutil.rmtree(dest)
            else:
                dest.unlink()
        if src.is_dir():
            shutil.copytree(src, dest)
        else:
            shutil.copy2(src, dest)
        print(f"  placed {dest}")

    shutil.rmtree(staging, ignore_errors=True)
    if not MARKER.exists():
        raise SystemExit(
            f"Download finished but {MARKER} is missing. "
            "Check that the dataset contains chroma_db/chroma.sqlite3."
        )
    print("Index ready.")


if __name__ == "__main__":
    main()
