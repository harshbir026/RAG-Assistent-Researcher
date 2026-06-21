import time
import json
from pathlib import Path
from src.retrieve import retrieve_dense
from src.generate import generate_answer

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TEST_QUERIES = [
    "How does FedAvg aggregate client model updates?",
    "What are the main challenges of federated learning with non-IID data?",
    "How does FedProx differ from FedAvg in handling system heterogeneity?",
    "What is the role of secure aggregation in federated learning privacy?",
    "What is differential privacy and how is epsilon defined?",
    "What is a membership inference attack and how does it work?",
    "What facial or frequency-domain artifacts help detect deepfakes?",
    "How do CNN-based deepfake detectors generalize to unseen GAN architectures?",
    "How does client drift occur in federated learning?",
    "What defenses exist against membership inference attacks?",
]


def measure_query(query: str) -> dict:
    t0 = time.time()
    chunks = retrieve_dense(query, k=5)
    t1 = time.time()
    retrieval_latency = t1 - t0

    result = generate_answer(query, chunks)
    t2 = time.time()
    generation_latency = t2 - t1
    total_latency = t2 - t0

    return {
        "query": query,
        "retrieval_latency_s": round(retrieval_latency, 3),
        "generation_latency_s": round(generation_latency, 3),
        "total_latency_s": round(total_latency, 3),
        "num_chunks": len(chunks),
        "total_tokens": result["usage"]["total_tokens"],
    }


def main():
    print(f"Measuring latency across {len(TEST_QUERIES)} queries...\n")

    results = []
    for i, query in enumerate(TEST_QUERIES, 1):
        print(f"[{i:2d}/{len(TEST_QUERIES)}] {query[:60]}...")
        r = measure_query(query)
        results.append(r)
        print(f"         retrieval: {r['retrieval_latency_s']}s | "
              f"generation: {r['generation_latency_s']}s | "
              f"total: {r['total_latency_s']}s")

    # ── summary stats ────────────────────────────────────
    cold_start_retrieval = results[0]["retrieval_latency_s"]
    retrieval_times = [r["retrieval_latency_s"] for r in results[1:]]
    generation_times = [r["generation_latency_s"] for r in results]
    total_times = [r["total_latency_s"] for r in results]

    summary = {
        "n_queries": len(results),
        "retrieval": {
            "avg": round(sum(retrieval_times) / len(retrieval_times), 3),
            "min": min(retrieval_times),
            "max": max(retrieval_times),
        },
        "generation": {
            "avg": round(sum(generation_times) / len(generation_times), 3),
            "min": min(generation_times),
            "max": max(generation_times),
        },
        "total": {
            "avg": round(sum(total_times) / len(total_times), 3),
            "min": min(total_times),
            "max": max(total_times),
        },
    }

    print(f"\n{'='*60}")
    print("LATENCY SUMMARY")
    print(f"{'='*60}")
    print(f"  Cold-start retrieval (incl. BGE load): {cold_start_retrieval}s  (one-time only)")
    print(f"  Retrieval (steady-state, n=9) — avg: {summary['retrieval']['avg']}s  "
          f"min: {summary['retrieval']['min']}s  max: {summary['retrieval']['max']}s")
    print(f"  Generation — avg: {summary['generation']['avg']}s  "
          f"min: {summary['generation']['min']}s  max: {summary['generation']['max']}s")
    print(f"  Total      — avg: {summary['total']['avg']}s  "
          f"min: {summary['total']['min']}s  max: {summary['total']['max']}s")

    pct_retrieval = round(summary['retrieval']['avg'] / summary['total']['avg'] * 100, 1)
    pct_generation = round(summary['generation']['avg'] / summary['total']['avg'] * 100, 1)
    print(f"\n  Bottleneck: generation is {pct_generation}% of total time, "
          f"retrieval is {pct_retrieval}%")
    print(f"{'='*60}")

    output = {"summary": summary, "per_query": results}
    output_path = RESULTS_DIR / "latency_benchmark.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()