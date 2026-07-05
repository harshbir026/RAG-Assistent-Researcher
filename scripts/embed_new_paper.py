"""
scripts/embed_new_paper.py
Embeds and upserts chunks for ONE specific paper into the existing
ChromaDB collection, without re-embedding the other papers.
Bypasses embed.py's "skip if collection already has data" safety check
by calling collection.upsert() directly.
"""

import json
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

CHUNKS_PATH = Path("data/chunks/all_chunks.jsonl")
CHROMA_DIR = Path("data/chroma_db")
COLLECTION_NAME = "research_papers"
BGE_MODEL_NAME = "BAAI/bge-base-en-v1.5"

TARGET_ARXIV_ID = "1602.05629"


def main():
    print(f"Loading chunks for arxiv_id={TARGET_ARXIV_ID}...")
    target_chunks = []
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        for line in f:
            chunk = json.loads(line)
            if chunk.get("arxiv_id") == TARGET_ARXIV_ID:
                target_chunks.append(chunk)

    if not target_chunks:
        print(f"❌ No chunks found for arxiv_id={TARGET_ARXIV_ID}. Check the ID matches your parsed sidecar.")
        return

    print(f"Found {len(target_chunks)} chunks for this paper.")

    print("Loading BGE model...")
    model = SentenceTransformer(BGE_MODEL_NAME, device="cpu")

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(COLLECTION_NAME)

    before_count = collection.count()

    texts = [c["text"] for c in target_chunks]
    texts_with_prefix = [f"Represent this passage for retrieval: {t}" for t in texts]

    embeddings = model.encode(
        texts_with_prefix,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    ids = [f"{c['arxiv_id']}_chunk_{c['chunk_index']}" for c in target_chunks]
    metadatas = [
        {
            "arxiv_id": c.get("arxiv_id", ""),
            "title": c.get("title", "")[:500],
            "authors": c.get("authors", "")[:500],
            "year": c.get("year", ""),
            "chunk_index": c.get("chunk_index", 0),
            "total_chunks": c.get("total_chunks", 0),
            "token_count": c.get("token_count", 0),
        }
        for c in target_chunks
    ]

    collection.upsert(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=texts,
        metadatas=metadatas,
    )

    after_count = collection.count()
    print(f"\n✅ Done. Collection size: {before_count} → {after_count} chunks (+{after_count - before_count})")


if __name__ == "__main__":
    main()
