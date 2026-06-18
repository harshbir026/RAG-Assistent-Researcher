import json
import time
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# ── config ────────────────────────────────────────────────
CHUNKS_PATH = Path("data/chunks/all_chunks.jsonl")
CHROMA_DIR  = Path("data/chroma_db")
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# Collection name — everything lives under this namespace in ChromaDB
COLLECTION_NAME = "research_papers"

# BGE model — downloads ~440MB on first run, cached locally after
# Why bge-base: 768-dim vectors, best free model for technical text
# on MTEB retrieval benchmark. Runs on CPU, no API cost.
BGE_MODEL_NAME = "BAAI/bge-base-en-v1.5"

# Batch size for embedding — how many chunks to embed at once.
# 64 is safe for CPU RAM. Increase to 128 if you have 16GB+ RAM.
BATCH_SIZE = 64
# ──────────────────────────────────────────────────────────


def load_chunks() -> list[dict]:
    """Load all chunks from the JSONL file produced by chunk.py."""
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"No chunks file at {CHUNKS_PATH} — run chunk.py first."
        )
    chunks = []
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    print(f"Loaded {len(chunks):,} chunks from {CHUNKS_PATH}")
    return chunks


def load_model() -> SentenceTransformer:
    """
    Load BGE embedding model.
    First run: downloads ~440MB to ~/.cache/huggingface/
    Subsequent runs: loads from cache instantly (~3 seconds)
    """
    print(f"\nLoading embedding model: {BGE_MODEL_NAME}")
    print("(First run downloads ~440MB — subsequent runs load from cache)\n")
    model = SentenceTransformer(BGE_MODEL_NAME, device="cpu")
    print(f"Model loaded. Embedding dimension: {model.get_sentence_embedding_dimension()}")
    return model


def get_or_create_collection(client: chromadb.PersistentClient):
    """
    Get existing ChromaDB collection or create a new one.
    Using get_or_create means re-runs are safe — won't duplicate data
    if you run embed.py twice.
    """
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine",   # use cosine similarity, not L2 distance
            "description": "RAG research assistant — FL, privacy ML, deepfakes",
        }
    )
    return collection


def embed_and_store(
    chunks: list[dict],
    model: SentenceTransformer,
    collection,
) -> None:
    """
    Embed all chunks in batches and store in ChromaDB.

    ChromaDB requires three parallel lists per upsert call:
    - ids: unique string ID per chunk
    - embeddings: the vector for each chunk
    - documents: the raw text (stored for retrieval)
    - metadatas: dict of metadata fields per chunk
    """
    # Check how many are already embedded (re-run safety)
    existing_count = collection.count()
    if existing_count > 0:
        print(f"\nCollection already has {existing_count:,} chunks.")
        print("To re-embed from scratch, delete data/chroma_db/ and re-run.")
        print("Skipping embedding — existing data preserved.\n")
        return

    print(f"\nEmbedding {len(chunks):,} chunks in batches of {BATCH_SIZE}...")
    print("This takes 10–20 minutes on CPU. Go get a coffee.\n")

    start_time = time.time()

    # Process in batches
    for batch_start in tqdm(
        range(0, len(chunks), BATCH_SIZE),
        desc="Embedding batches",
        unit="batch",
    ):
        batch = chunks[batch_start : batch_start + BATCH_SIZE]

        # ── prepare texts for embedding ───────────────────
        texts = [chunk["text"] for chunk in batch]

        # BGE-specific instruction prefix — improves retrieval quality.
        # BGE was trained with this prefix on retrieval tasks.
        # Add it to texts being stored (not to queries — queries get
        # a different prefix: "Represent this sentence for searching")
        texts_with_prefix = [
            f"Represent this passage for retrieval: {t}" for t in texts
        ]

        # ── embed ─────────────────────────────────────────
        embeddings = model.encode(
            texts_with_prefix,
            normalize_embeddings=True,   # normalise to unit length for cosine similarity
            show_progress_bar=False,
            batch_size=BATCH_SIZE,
                             # tqdm handles outer progress
        )

        # ── build ChromaDB inputs ─────────────────────────
        ids = [
            f"{chunk['arxiv_id']}_chunk_{chunk['chunk_index']}"
            for chunk in batch
        ]

        metadatas = [
            {
                "arxiv_id":    chunk.get("arxiv_id", ""),
                "title":       chunk.get("title", "")[:500],   # ChromaDB has metadata size limits
                "authors":     chunk.get("authors", "")[:500],
                "year":        chunk.get("year", ""),
                "chunk_index": chunk.get("chunk_index", 0),
                "total_chunks":chunk.get("total_chunks", 0),
                "token_count": chunk.get("token_count", 0),
            }
            for chunk in batch
        ]

        # ── upsert into ChromaDB ──────────────────────────
        # upsert = insert if new, update if ID already exists
        collection.upsert(
            ids=ids,
            embeddings=embeddings.tolist(),   # numpy array → plain list
            documents=texts,                  # original text without prefix
            metadatas=metadatas,
        )

    elapsed = time.time() - start_time
    print(f"\nEmbedding complete in {elapsed/60:.1f} minutes")
    print(f"Total chunks stored in ChromaDB: {collection.count():,}")


def test_retrieval(model: SentenceTransformer, collection) -> None:
    """
    Run 3 test queries to verify retrieval is working.
    Prints top-3 chunks per query with similarity scores.
    """
    test_queries = [
        "How does FedAvg handle non-IID data across clients?",
        "What is differential privacy and how is epsilon defined?",
        "How do deepfake detection models identify GAN-generated faces?",
    ]

    print(f"\n{'='*60}")
    print("RETRIEVAL TEST — 3 queries across 3 domains")
    print(f"{'='*60}")

    for query in test_queries:
        print(f"\nQuery: {query}")
        print("─" * 60)

        # BGE query prefix — different from document prefix
        query_with_prefix = f"Represent this sentence for searching: {query}"
        query_embedding = model.encode(
            [query_with_prefix],
            normalize_embeddings=True,
        )

        results = collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=3,
            include=["documents", "metadatas", "distances"],
        )

        for i, (doc, meta, dist) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )):
            # ChromaDB cosine distance: 0 = identical, 2 = opposite
            # Convert to similarity: similarity = 1 - distance
            similarity = 1 - dist
            print(f"\n  Result {i+1} — similarity: {similarity:.3f}")
            print(f"  Paper : {meta.get('arxiv_id')} — {meta.get('title','')[:60]}")
            print(f"  Year  : {meta.get('year')}  |  "
                  f"Chunk : {meta.get('chunk_index')}/{meta.get('total_chunks')}")
            print(f"  Text  : {doc[:200]}...")


def main():
    # 1. Load chunks
    chunks = load_chunks()

    # 2. Load BGE model
    model = load_model()

    # 3. Connect to persistent ChromaDB
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
    )
    collection = get_or_create_collection(client)

    # 4. Embed and store
    embed_and_store(chunks, model, collection)

    # 5. Test retrieval
    test_retrieval(model, collection)

    print(f"\n{'='*60}")
    print("✅  Phase 1 complete — corpus ingested and embedded.")
    print(f"    ChromaDB at: {CHROMA_DIR.resolve()}")
    print(f"    Collection : {COLLECTION_NAME}")
    print(f"    Chunks     : {collection.count():,}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()