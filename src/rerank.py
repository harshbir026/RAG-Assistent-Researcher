from flashrank import Ranker, RerankRequest
from src.retrieve import RetrievedChunk
from src.logger import get_logger

logger = get_logger(__name__)

# ── module-level singleton ─────────────────────────────────
# Like BGE, loading the cross-encoder takes a second or two.
# We load it once and reuse it across all rerank calls.
_ranker = None

def _get_ranker() -> Ranker:
    global _ranker
    if _ranker is None:
        print("Loading FlashRank cross-encoder model (one-time)...")
        # FlashRank defaults to 'ms-marco-MiniLM-L-12-v2' 
        # A tiny (~120MB), fast, and highly accurate cross-encoder
        _ranker = Ranker()
    return _ranker

def rerank(query: str, chunks: list[RetrievedChunk], top_n: int = 5) -> list[RetrievedChunk]:
    """
    Takes a list of retrieved chunks, re-scores them using a cross-encoder,
    and returns the top_n most relevant chunks.
    """
    if not chunks:
        return []

    ranker = _get_ranker()

    # 1. Format chunks into the schema FlashRank expects
    passages = []
    for i, c in enumerate(chunks):
        passages.append({
            "id": str(i), 
            "text": c.text,
            "meta": {
                "arxiv_id": c.arxiv_id,
                "title": c.title,
                "authors": c.authors,
                "year": c.year,
                "chunk_index": c.chunk_index,
                "total_chunks": c.total_chunks,
                "original_similarity": c.similarity
            }
        })

    # 2. Run the cross-encoder
    request = RerankRequest(query=query, passages=passages)
    results = ranker.rerank(request)

    # 3. Map back to RetrievedChunk objects, injecting the new cross-encoder score
    reranked_chunks = []
    for r in results[:top_n]:
        meta = r["meta"]
        reranked_chunks.append(RetrievedChunk(
            text=r["text"],
            similarity=round(r["score"], 4),  # Replaced with FlashRank score
            arxiv_id=meta["arxiv_id"],
            title=meta["title"],
            authors=meta["authors"],
            year=meta["year"],
            chunk_index=meta["chunk_index"],
            total_chunks=meta["total_chunks"],
        ))

    return reranked_chunks