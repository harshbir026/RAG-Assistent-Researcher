from src.retrieve import retrieve_dense
from src.rerank import rerank

# A query where dense retrieval sometimes surfaces tangential chunks
query = "What is the privacy-utility tradeoff when epsilon < 1 in DP-SGD?"

print(f"Query: '{query}'\n")

# 1. Retrieve top 10 using dense embeddings
dense_chunks = retrieve_dense(query, k=10)

print("─── DENSE TOP 3 (Without Reranking) ───")
for i, c in enumerate(dense_chunks[:3], 1):
    print(f"[{i}] {c.arxiv_id} | Score: {c.similarity:.3f} | {c.title[:60]}")

print("\n─── RERANKED TOP 3 (Cross-Encoder) ───")
# 2. Rerank the top 10 down to top 3
reranked_chunks = rerank(query, dense_chunks, top_n=3)

for i, c in enumerate(reranked_chunks, 1):
    print(f"[{i}] {c.arxiv_id} | Score: {c.similarity:.3f} | {c.title[:60]}")
    
    # Find where this chunk originally ranked in the dense top 10
    orig_rank = next((idx for idx, orig in enumerate(dense_chunks) 
                      if orig.arxiv_id == c.arxiv_id and orig.chunk_index == c.chunk_index), -1) + 1
    print(f"    ↳ Originally ranked #{orig_rank} by dense retrieval")