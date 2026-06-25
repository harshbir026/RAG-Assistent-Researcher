from pathlib import Path
from dataclasses import dataclass, field
from sentence_transformers import SentenceTransformer
import pickle
import math
from rank_bm25 import BM25Okapi
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
# ── BM25 config ──────────────────────────────────────────
BM25_INDEX_PATH = Path("data/bm25_index.pkl")


def _tokenize(text: str) -> list[str]:
    """
    Simple whitespace + lowercase tokenizer.
    Must be identical for both index building and query tokenization —
    any mismatch silently degrades retrieval quality because the BM25
    vocabulary won't match the query terms.
    """
    return text.lower().split()


def _build_bm25_index() -> tuple[BM25Okapi, list[dict]]:
    """
    Build BM25 index over all chunks from the JSONL file.
    Takes ~2 minutes first time. Saves to disk so subsequent calls
    load instantly (~1 second).
    """
    from pathlib import Path
    import json

    chunks_path = Path("data/chunks/all_chunks.jsonl")
    if not chunks_path.exists():
        raise FileNotFoundError("No chunks file — run chunk.py first.")

    print("Building BM25 index from all chunks (one-time, ~2 min)...")
    corpus = []
    chunk_store = []

    with open(chunks_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunk = json.loads(line)
            corpus.append(_tokenize(chunk["text"]))
            chunk_store.append(chunk)

    bm25 = BM25Okapi(corpus)
    print(f"BM25 index built over {len(corpus):,} chunks.")

    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": chunk_store}, f)
    print(f"Index saved to {BM25_INDEX_PATH}")

    return bm25, chunk_store


def _load_bm25_index() -> tuple[BM25Okapi, list[dict]]:
    """Load from disk if it exists, otherwise build it."""
    if BM25_INDEX_PATH.exists():
        with open(BM25_INDEX_PATH, "rb") as f:
            data = pickle.load(f)
        return data["bm25"], data["chunks"]
    return _build_bm25_index()


# module-level singleton — same lazy-load pattern as dense retrieval
_bm25 = None
_bm25_chunks = None


def _get_bm25():
    global _bm25, _bm25_chunks
    if _bm25 is None:
        _bm25, _bm25_chunks = _load_bm25_index()
    return _bm25, _bm25_chunks


def retrieve_bm25(
    query: str,
    k: int = 5,
) -> list[RetrievedChunk]:
    """
    BM25 sparse retrieval — keyword-based scoring.

    Returns the same List[RetrievedChunk] format as retrieve_dense()
    so downstream code (generate.py, evaluate.py) doesn't need to
    know which retriever produced the result.

    BM25 wins on: exact term matches, acronyms (FedAvg, GAN, BERT),
    technical identifiers, queries that need specific keywords present.
    Dense wins on: semantic/conceptual queries, paraphrase matching,
    queries where the exact term doesn't appear in relevant chunks.
    """
    query = validate_query(query)

    bm25, chunk_store = _get_bm25()
    tokenized_query = _tokenize(query)

    scores = bm25.get_scores(tokenized_query)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

    retrieved = []
    for idx in top_indices:
        score = scores[idx]
        if score <= 0:
            continue   # BM25 score of 0 means no term overlap at all — skip

        chunk = chunk_store[idx]
        # Normalise BM25 score to a 0-1 range for consistent display.
        # BM25 scores have no fixed upper bound, so we use a soft cap:
        # score / (score + 10) maps any positive score to (0, 1).
        normalised_score = round(score / (score + 10), 4)

        retrieved.append(RetrievedChunk(
            text=chunk.get("text", ""),
            similarity=normalised_score,
            arxiv_id=chunk.get("arxiv_id", ""),
            title=chunk.get("title", ""),
            authors=chunk.get("authors", ""),
            year=chunk.get("year", ""),
            chunk_index=chunk.get("chunk_index", 0),
            total_chunks=chunk.get("total_chunks", 0),
        ))

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

def retrieve_hybrid(query: str, k: int = 5) -> list[RetrievedChunk]:
    """
    Hybrid Retrieval Pipeline (Two-Stage):
    Stage 1: Retrieve candidate lists using both Dense (BGE) and Sparse (BM25).
    Stage 2: Fuse the raw ranks using Reciprocal Rank Fusion (RRF).
    Stage 3: Extract top candidates and pass to Cross-Encoder Reranker for final top-k selection.
    
    Formula: RRF_Score(d) = \sum_{m \in M} \frac{1}{60 + r_m(d)}
    """
    # Defensive query validation
    query = validate_query(query)
    
    # 1. Fetch deep candidate pools from both retrieval strategies (fetch k=10 or 15)
    pool_depth = max(k * 2, 10)
    dense_candidates = retrieve_dense(query, k=pool_depth)
    bm25_candidates = retrieve_bm25(query, k=pool_depth)
    
    # 2. Compute Reciprocal Rank Fusion scores
    # We use a standard RRF constant of 60 to prevent early ranks from completely dominating
    rrf_constant = 60
    rrf_scores = {}  # maps (arxiv_id, chunk_index) -> float rrf score
    chunk_lookup = {} # maps (arxiv_id, chunk_index) -> RetrievedChunk object
    
    # Process Dense Ranks
    for rank, chunk in enumerate(dense_candidates, start=1):
        key = (chunk.arxiv_id, chunk.chunk_index)
        chunk_lookup[key] = chunk
        rrf_scores[key] = rrf_scores.get(key, 0.0) + (1.0 / (rrf_constant + rank))
        
    # Process BM25 Ranks
    for rank, chunk in enumerate(bm25_candidates, start=1):
        key = (chunk.arxiv_id, chunk.chunk_index)
        chunk_lookup[key] = chunk
        rrf_scores[key] = rrf_scores.get(key, 0.0) + (1.0 / (rrf_constant + rank))
        
    # 3. Sort unique candidates by their cumulative RRF scores descending
    sorted_keys = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    
    # Build the intermediate candidate list preserving RRF priority order
    rrf_candidates = []
    for key in sorted_keys:
        base_chunk = chunk_lookup[key]
        # Temporarily store normalized RRF score as similarity metric
        base_chunk.similarity = round(rrf_scores[key], 5)
        rrf_candidates.append(base_chunk)
        
    # 4. Local execution of Stage 3: Cross-Encoder Reranking
    # Imported locally to prevent a circular dependency crash with src/rerank.py
    from src.rerank import rerank
    
    # Feed the top RRF candidates directly into the local cross-encoder
    final_candidates = rerank(query, rrf_candidates, top_n=k)
    
    return final_candidates

if __name__ == "__main__":
    run_test_suite()