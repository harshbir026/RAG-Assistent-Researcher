#!/usr/bin/env python3
"""Upload the local retrieval index to a Hugging Face dataset (free).

  hf auth login
  python scripts/upload_space_index.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo
from huggingface_hub.errors import HfHubHTTPError

load_dotenv()


def main() -> None:
    repo_id = os.getenv("INDEX_DATASET", "Harshbirr/researchmind-index")
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    api = HfApi(token=token)
    try:
        me = api.whoami()
    except HfHubHTTPError:
        print(
            "Not logged in (401).\n"
            "Do not paste the token in chat.\n"
            "Add HF_TOKEN=hf_... to .env (gitignored), then re-run this script.\n"
            "Or in your own terminal: hf auth login",
            file=sys.stderr,
        )
        raise SystemExit(1)

    print(f"Logged in as {me.get('name') or me.get('email')}")
    create_repo(repo_id, repo_type="dataset", exist_ok=True, private=True)
    root = Path("data")
    files = [
        (root / "chroma_db", "chroma_db"),
        (root / "chunks" / "all_chunks.jsonl", "chunks/all_chunks.jsonl"),
        (root / "chunks" / "software_docs.jsonl", "chunks/software_docs.jsonl"),
        (root / "bm25_index.pkl", "bm25_index.pkl"),
        (root / "bm25_index_software_docs.pkl", "bm25_index_software_docs.pkl"),
    ]
    for local, path_in_repo in files:
        if not local.exists():
            print(f"skip missing {local}")
            continue
        print(f"upload {local} -> {repo_id}:{path_in_repo}")
        if local.is_dir():
            api.upload_folder(
                folder_path=str(local),
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type="dataset",
            )
        else:
            api.upload_file(
                path_or_fileobj=str(local),
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type="dataset",
            )
    print(f"Done. Dataset: https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    main()
