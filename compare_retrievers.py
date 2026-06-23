from src.retrieve import retrieve_dense, retrieve_bm25

TEST_QUERIES = [
    "How does FedAvg aggregate client model updates?",
    "What is differential privacy and how is epsilon defined?",
    "What visual artifacts help detect GAN-generated faces?",
    "How does client drift occur in federated learning?",
    "What is a membership inference attack?",
]

for query in TEST_QUERIES:
    dense_results = retrieve_dense(query, k=5)
    bm25_results = retrieve_bm25(query, k=5)

    dense_ids = {r.arxiv_id for r in dense_results}
    bm25_ids = {r.arxiv_id for r in bm25_results}
    overlap = dense_ids & bm25_ids
    unique_to_dense = dense_ids - bm25_ids
    unique_to_bm25 = bm25_ids - dense_ids

    print(f"\nQuery: {query[:70]}")
    print(f"  Dense top papers : {sorted(dense_ids)}")
    print(f"  BM25  top papers : {sorted(bm25_ids)}")
    print(f"  Overlap          : {len(overlap)}/5")
    print(f"  Only dense       : {unique_to_dense}")
    print(f"  Only BM25        : {unique_to_bm25}")