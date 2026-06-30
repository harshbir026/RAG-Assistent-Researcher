import json
from pathlib import Path
from src.retrieve import retrieve_dense
from src.generate import generate_answer

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    with open("golden_qa.json") as f:
        data = json.load(f)
    questions = data["questions"]

    print(f"Running full pipeline on {len(questions)} golden questions...\n")

    answers = []
    for i, q in enumerate(questions, 1):
        query = q["question"]
        chunks = retrieve_dense(query, k=5)
        result = generate_answer(query, chunks)

        answers.append({
            "id": q["id"],
            "tier": q["tier"],
            "tier_name": q["tier_name"],
            "question": query,
            "generated_answer": result["answer"],
            "ground_truth": q["ground_truth"],
            "num_chunks_used": result["num_chunks_used"],
            "sources": [list(s) for s in result["sources"]],
            "total_tokens": result["usage"]["total_tokens"],
        })

        print(f"  [{i:2d}/{len(questions)}] {q['id']} ({q['tier_name']}) — done")

    output_path = RESULTS_DIR / "dense_answers.json"
    with open(output_path, "w") as f:
        json.dump({"answers": answers}, f, indent=2)

    print(f"\nSaved {len(answers)} answers to {output_path}")


if __name__ == "__main__":
    main()