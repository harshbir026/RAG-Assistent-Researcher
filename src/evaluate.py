import os
import json
import time
from pathlib import Path
from dotenv import load_dotenv

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    AnswerRelevancy,
    context_precision,
    context_recall,
)
from ragas.llms import llm_factory
from ragas.embeddings import embedding_factory
from openai import OpenAI

from src.retrieve import retrieve_dense
from src.generate import generate_answer

load_dotenv()

GOLDEN_QA_PATH = Path("golden_qa.json")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── RAGAS client/llm/embeddings setup ──────────────────────
_openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
llm = llm_factory("gpt-4o-mini", client=_openai_client, max_tokens=2000)
embeddings = embedding_factory(client=_openai_client)


class EmbedQueryAdapter:
    """
    Bridges the legacy AnswerRelevancy metric (which calls .embed_query()
    and .embed_documents()) to the modern embedding_factory() object
    (which only exposes .embed_text()). A real incompatibility in ragas
    0.4.3 between its legacy and modern embedding APIs — confirmed via
    direct traceback inspection across multiple missing methods.
    """
    def __init__(self, modern_embeddings):
        self._inner = modern_embeddings

    def embed_query(self, text):
        return self._inner.embed_text(text)

    def embed_documents(self, texts):
        return [self._inner.embed_text(t) for t in texts]

    def __getattr__(self, name):
        return getattr(self._inner, name)


# answer_relevancy needs the adapter; other metrics use embeddings directly
answer_relevancy = AnswerRelevancy(
    embeddings=EmbedQueryAdapter(embeddings),
    llm=llm,
)
# ──────────────────────────────────────────────────────────


def load_golden_qa(tier: int | None = None) -> list[dict]:
    """
    Load golden Q&A set, optionally filtered to a single tier.
    tier=None loads all 25. tier=1/2/3 loads just that tier.
    """
    with open(GOLDEN_QA_PATH) as f:
        data = json.load(f)
    questions = data["questions"]
    if tier is not None:
        questions = [q for q in questions if q["tier"] == tier]
    return questions


def build_ragas_dataset(
    questions: list[dict],
    retriever_fn=retrieve_dense,
    k: int = 5,
) -> Dataset:
    """
    Build the dataset RAGAS needs: question, answer, contexts, ground_truth.
    contexts MUST be List[str] — RAGAS will silently fail or score 0
    on every metric if you pass anything else (e.g. RetrievedChunk objects).
    """
    rows = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }

    print(f"Building RAGAS dataset for {len(questions)} questions...\n")

    for i, q in enumerate(questions, 1):
        query = q["question"]

        chunks = retriever_fn(query, k=k)
        result = generate_answer(query, chunks)

        rows["question"].append(query)
        rows["answer"].append(result["answer"])
        # CRITICAL: contexts must be List[str], not List[RetrievedChunk]
        rows["contexts"].append([c.text for c in chunks])
        rows["ground_truth"].append(q["ground_truth"])

        print(f"  [{i:2d}/{len(questions)}] {q['id']} ({q['tier_name']}) — "
              f"{len(chunks)} chunks retrieved")

    return Dataset.from_dict(rows)


def run_ragas_eval(
    retriever_fn=retrieve_dense,
    n_questions: int | None = None,
    tier: int | None = None,
    k: int = 5,
    output_name: str = "dense_eval",
) -> dict:
    """
    Run RAGAS evaluation on the golden set (or a tier subset).

    Args:
        retriever_fn: which retrieval function to use (retrieve_dense for now;
                      retrieve_hybrid comes later)
        n_questions: limit to first N questions (None = all in scope)
        tier: filter to a single tier (1, 2, or 3), None = all tiers
        k: chunks to retrieve per question
        output_name: filename (without .json) to save results under

    Returns:
        dict of metric_name -> score
    """
    questions = load_golden_qa(tier=tier)
    if n_questions:
        questions = questions[:n_questions]

    print(f"{'='*60}")
    print(f"RAGAS EVALUATION — {len(questions)} questions"
          f"{f' (tier {tier})' if tier else ' (all tiers)'}")
    print(f"{'='*60}\n")

    start_time = time.time()

    dataset = build_ragas_dataset(questions, retriever_fn=retriever_fn, k=k)

    print(f"\nRunning RAGAS metrics (this takes several minutes)...\n")

    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        llm=llm,
        embeddings=embeddings,
    )

    elapsed = time.time() - start_time

    scores = result.to_pandas()[
        ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    ].mean().to_dict()

    print(f"\n{'='*60}")
    print("RAGAS RESULTS")
    print(f"{'='*60}")
    for metric, score in scores.items():
        print(f"  {metric:20s}: {score:.4f}")
    print(f"\nTime elapsed: {elapsed/60:.1f} minutes")
    print(f"{'='*60}")

    # Save results
    output_path = RESULTS_DIR / f"{output_name}.json"
    with open(output_path, "w") as f:
        json.dump({
            "scores": scores,
            "n_questions": len(questions),
            "tier": tier,
            "retriever": retriever_fn.__name__,
            "k": k,
            "elapsed_minutes": round(elapsed / 60, 2),
        }, f, indent=2)

    print(f"\nSaved to {output_path}")

    return scores


if __name__ == "__main__":
    # First full run — all 25 questions, dense retrieval only
    run_ragas_eval(
        retriever_fn=retrieve_dense,
        tier=None,
        k=5,
        output_name="dense_eval",
    )