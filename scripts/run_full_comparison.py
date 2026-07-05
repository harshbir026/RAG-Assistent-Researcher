"""
scripts/run_full_comparison.py
Runs all 25 golden questions through BM25, Dense, Hybrid, and Agent modes.
Saves results + timing to results/full_comparison.json
"""

import json
import time
from pathlib import Path

from src.retrieve import retrieve_dense, retrieve_bm25, retrieve_hybrid
from src.generate import generate_answer
from src.agent import research_agent, system_prompt, classify_query
from langchain_core.messages import HumanMessage, SystemMessage

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_standard_mode(mode_name, retriever_fn, questions):
    results = []
    for q in questions:
        query = q["question"]
        t0 = time.time()
        chunks = retriever_fn(query, k=5)
        gen = generate_answer(query, chunks)
        elapsed = time.time() - t0
        results.append({
            "id": q["id"],
            "tier": q.get("tier"),
            "tier_name": q.get("tier_name"),
            "question": query,
            "ground_truth": q.get("ground_truth"),
            "generated_answer": gen["answer"],
            "num_chunks_used": gen["num_chunks_used"],
            "sources": [list(s) for s in gen["sources"]],
            "total_tokens": gen["usage"]["total_tokens"],
            "elapsed_s": round(elapsed, 2),
        })
        print(f"  [{mode_name}] {elapsed:.1f}s — {query[:60]}")
    return results


def run_agent_mode(questions):
    results = []
    for q in questions:
        query = q["question"]
        t0 = time.time()
        classification = classify_query(query)
        response = research_agent.invoke({
            "messages": [
                SystemMessage(content=system_prompt),
                HumanMessage(content=query),
            ]
        })
        elapsed = time.time() - t0
        tool_calls = sum(1 for m in response["messages"] if getattr(m, "name", "") == "search_research_papers")
        results.append({
            "id": q["id"],
            "tier": q.get("tier"),
            "tier_name": q.get("tier_name"),
            "question": query,
            "ground_truth": q.get("ground_truth"),
            "generated_answer": response["messages"][-1].content,
            "classification": classification,
            "tool_calls": tool_calls,
            "elapsed_s": round(elapsed, 2),
        })
        print(f"  [agent] {elapsed:.1f}s ({tool_calls} searches) — {query[:60]}")
    return results


def main():
    with open("golden_qa.json") as f:
        data = json.load(f)
    questions = data["questions"]

    print(f"Running {len(questions)} golden questions across 4 modes...\n")

    all_results = {}

    print("── BM25 ──")
    all_results["bm25"] = run_standard_mode("bm25", retrieve_bm25, questions)

    print("\n── Dense ──")
    all_results["dense"] = run_standard_mode("dense", retrieve_dense, questions)

    print("\n── Hybrid ──")
    all_results["hybrid"] = run_standard_mode("hybrid", retrieve_hybrid, questions)

    print("\n── Agent ──")
    all_results["agent"] = run_agent_mode(questions)

    with open(RESULTS_DIR / "full_comparison.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 50)
    print("SUMMARY — Average latency per mode")
    print("=" * 50)
    for mode, results in all_results.items():
        avg = sum(r["elapsed_s"] for r in results) / len(results)
        print(f"{mode:10s}: {avg:.2f}s avg over {len(results)} questions")

    print(f"\nSaved full results to {RESULTS_DIR / 'full_comparison.json'}")


if __name__ == "__main__":
    main()
