import time
from src.retrieve import retrieve_dense, retrieve_bm25, retrieve_hybrid

# A complex question that tests structural understanding and keyword matches simultaneously
query = "Compare FedAvg and FedProx convergence under severe non-IID conditions"

print(f"Testing Query: '{query}'\n")

# Timed execution for benchmark comparisons
t0 = time.time()
dense = retrieve_dense(query, k=3)
t1 = time.time()
sparse = retrieve_bm25(query, k=3)
t2 = time.time()
hybrid = retrieve_hybrid(query, k=3)
t3 = time.time()

print(f"⏱️  Dense Retrieval  : {t1 - t0:.3f}s")
print(f"⏱️  Sparse Retrieval : {t2 - t1:.3f}s")
print(f"⏱️  Hybrid Pipeline  : {t3 - t2:.3f}s (Includes RRF + Reranker)\n")

print("─── HYBRID PIPELINE TOP 3 CHUNKS ───")
for i, c in enumerate(hybrid, 1):
    print(f"[{i}] ArXiv ID: {c.arxiv_id} | Reranker Score: {c.similarity:.4f}")
    print(f"    Title: {c.title[:75]}")
    print(f"    Snippet: {c.text[:140].strip()}...\n")