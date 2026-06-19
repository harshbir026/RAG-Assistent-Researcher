import sys
from src.retrieve import retrieve_dense, print_results

if len(sys.argv) < 2:
    print("Usage: python check_retrieval.py 'your question here'")
    sys.exit(1)

query = sys.argv[1]
results = retrieve_dense(query, k=5)
print_results(query, results)
