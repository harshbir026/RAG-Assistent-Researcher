from pathlib import Path
from dataclasses import dataclass, field
from sentence_transformers import SentenceTransformer
import chromadb
from src.logger import get_logger
logger = get_logger(__name__)

class QueryValidationError(ValueError):
    """Raised when a query fails validation before retrieval is attempted."""
    pass


MIN_QUERY_LENGTH = 10
MAX_QUERY_LENGTH = 500


def validate_query(query: str) -> str:
    """
    Validate and normalize a user query before retrieval.
    Raises QueryValidationError with a clear message if invalid.
    """
    query = query.strip()

    if not query:
        raise QueryValidationError("Query cannot be empty.")

    if len(query) < MIN_QUERY_LENGTH:
        raise QueryValidationError(
            f"Query too short ({len(query)} chars). "
            f"Minimum {MIN_QUERY_LENGTH} characters required."
        )

    if len(query) > MAX_QUERY_LENGTH:
        raise QueryValidationError(
            f"Query too long ({len(query)} chars). "
            f"Maximum {MAX_QUERY_LENGTH} characters allowed."
        )

    return query

# ── config ────────────────────────────────────────────────
CHROMA_DIR = Path("data/chroma_db")
COLLECTION_NAME = "research_papers"
BGE_MODEL_NAME = "BAAI/bge-base-en-v1.5"

# Chunks below this similarity are considered noise — filtered out.
# 0.55 is permissive (catches loosely related chunks); raise to 0.6+
# for stricter precision once you do the ablation study on Day 13.
SIMILARITY_THRESHOLD = 0.55
# ──────────────────────────────────────────────────────────


@dataclass
class RetrievedChunk:
    """
    One retrieved chunk with its similarity score and source metadata.
    Using a dataclass instead of a raw dict gives you autocomplete
    and type safety everywhere downstream (generate.py, agent.py).
    """
    text: str
    similarity: float
    arxiv_id: str
    title: str
    authors: str
    year: str
    chunk_index: int
    total_chunks: int


# ── module-level singletons ─────────────────────────────────
# Loading the BGE model takes ~3 seconds. We load it ONCE when this
# module is imported, not on every call to retrieve_dense(). Every
# other file (generate.py, agent.py, evaluate.py) imports this module
# and reuses the same loaded model — avoids reloading 3 seconds per call.
_model = None
_collection = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print("Loading BGE model (one-time)...")
        _model = SentenceTransformer(BGE_MODEL_NAME, device="cpu")
    return _model


def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = client.get_collection(COLLECTION_NAME)
    return _collection


def retrieve_dense(
    query: str,
    k: int = 5,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
) -> list[RetrievedChunk]:
    query = validate_query(query)
    """
    Embed a query and retrieve the top-k most similar chunks from ChromaDB.

    Pipeline:
    1. Add BGE's query instruction prefix
    2. Embed the query with the same BGE model used to embed the corpus
       (CRITICAL: query and corpus must use the same model, or vectors
       live in different spaces and similarity is meaningless)
    3. Cosine similarity search in ChromaDB
    4. Filter out chunks below similarity_threshold
    5. Return as typed RetrievedChunk objects

    Args:
        query: natural language question
        k: how many chunks to retrieve (before threshold filtering)
        similarity_threshold: minimum similarity to keep a result

    Returns:
        List of RetrievedChunk, sorted by similarity descending.
        May return fewer than k if some results fall below threshold.
    """
    model = _get_model()
    collection = _get_collection()

    # BGE query prefix — MUST match what was used at embedding time.
    # Documents were embedded with "Represent this passage for retrieval: "
    # Queries use a different prefix: "Represent this sentence for searching: "
    query_with_prefix = f"Represent this sentence for searching: {query}"

    query_embedding = model.encode(
        [query_with_prefix],
        normalize_embeddings=True,
    )

    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    # ChromaDB returns nested lists (one per query) — we only sent 1 query
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    retrieved = []
    for doc, meta, dist in zip(docs, metas, dists):
        # ChromaDB cosine distance: 0 = identical, 2 = completely opposite
        similarity = 1 - dist

        if similarity < similarity_threshold:
            continue   # filter out low-relevance noise

        retrieved.append(RetrievedChunk(
            text=doc,
            similarity=round(similarity, 4),
            arxiv_id=meta.get("arxiv_id", ""),
            title=meta.get("title", ""),
            authors=meta.get("authors", ""),
            year=meta.get("year", ""),
            chunk_index=meta.get("chunk_index", 0),
            total_chunks=meta.get("total_chunks", 0),
        ))

    if not retrieved:
        logger.warning(f"No chunks passed threshold for query: '{query[:60]}...'")

    return retrieved


def print_results(query: str, results: list[RetrievedChunk]) -> None:
    """Pretty-print retrieval results for manual inspection."""
    print(f"\nQuery: {query}")
    print(f"Retrieved: {len(results)} chunks (threshold: {SIMILARITY_THRESHOLD})")
    print("─" * 60)

    if not results:
        print("  ⚠️  No chunks passed the similarity threshold.")
        return

    for i, r in enumerate(results, 1):
        print(f"\n  [{i}] similarity: {r.similarity:.3f}")
        print(f"      Paper : {r.arxiv_id} — {r.title[:60]}")
        print(f"      Year  : {r.year}  |  Chunk: {r.chunk_index}/{r.total_chunks}")
        print(f"      Text  : {r.text[:200]}...")


# ── manual test set — 5 queries across all 3 domains ────────
TEST_QUERIES = [
    # Federated learning
    "How does FedAvg aggregate client model updates?",
    "What are the main challenges of federated learning with non-IID data?",
    # Privacy ML
    "How does differential privacy provide formal privacy guarantees?",
    "What is a membership inference attack and how does it work?",
    # Deepfake detection
    "What visual artifacts help detect GAN-generated faces?",
]


def run_test_suite():
    """Run all 5 test queries and print results for manual spot-check."""
    print("=" * 60)
    print("RETRIEVAL TEST SUITE — 5 queries across 3 domains")
    print("=" * 60)

    for query in TEST_QUERIES:
        results = retrieve_dense(query, k=5)
        print_results(query, results)
        print()

    print("=" * 60)
    print("✅  Manually verify each result above is topically relevant.")
    print("=" * 60)


if __name__ == "__main__":
    run_test_suite()