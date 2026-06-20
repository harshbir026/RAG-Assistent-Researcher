import os
from dotenv import load_dotenv
from openai import OpenAI

from src.retrieve import retrieve_dense, RetrievedChunk

load_dotenv()

# ── config ────────────────────────────────────────────────
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODEL_NAME = "gpt-4o-mini"

# Why gpt-4o-mini: cheap enough to run hundreds of evaluation queries
# without worrying about cost (~$0.15/1M input tokens), while still
# strong enough to follow citation instructions reliably. GPT-4o
# would cost ~20x more for marginal quality gain on this task.
TEMPERATURE = 0.1   # low temperature — factual Q&A, not creative writing

SYSTEM_PROMPT = """You are a research assistant specializing in machine learning papers.

Answer the user's question using ONLY the provided context below. Do not use any
outside knowledge, even if you know the answer from training.

Rules:
1. Cite the paper title (and year if available) for every claim you make, using
   the format: (Paper Title, Year)
2. If the context does not contain enough information to answer the question,
   say so explicitly — do not guess or fill gaps with outside knowledge.
3. If different papers disagree or present different approaches, mention this.
4. Keep your answer focused and well-organized. Use bullet points if comparing
   multiple papers or methods.
5. Do not fabricate paper titles, authors, or findings not present in the context.
"""
# ──────────────────────────────────────────────────────────


def build_context_block(chunks: list[RetrievedChunk]) -> str:
    """
    Format retrieved chunks into a numbered context block for the prompt.

    Format:
    [1] From: {paper_title} ({year})
    {chunk_text}

    [2] From: {paper_title} ({year})
    {chunk_text}
    ...

    Numbering lets the model and the user cross-reference which
    chunk supported which claim, and makes citation instructions
    concrete rather than abstract.
    """
    if not chunks:
        return "No relevant context was found in the corpus."

    blocks = []
    for i, chunk in enumerate(chunks, 1):
        year_str = f" ({chunk.year})" if chunk.year else ""
        title = chunk.title.strip() or "Untitled paper"
        blocks.append(
            f"[{i}] From: {title}{year_str}\n{chunk.text.strip()}"
        )

    return "\n\n".join(blocks)


def generate_answer(
    query: str,
    chunks: list[RetrievedChunk],
    model: str = MODEL_NAME,
    temperature: float = TEMPERATURE,
) -> dict:
    """
    Generate a cited answer from retrieved chunks using an LLM.

    Args:
        query: the user's question
        chunks: list of RetrievedChunk from retrieve_dense()
        model: OpenAI model name
        temperature: sampling temperature (low = more deterministic)

    Returns:
        dict with keys: "answer", "context_used", "model", "sources"
    """
    context_block = build_context_block(chunks)

    user_message = f"""Context:
{context_block}

Question: {query}

Answer the question using only the context above, with citations."""

    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )

    answer_text = response.choices[0].message.content

    # Collect unique source papers cited in context (not necessarily
    # all were actually cited by the model — this is what WAS available)
    sources = list({
        (c.arxiv_id, c.title, c.year) for c in chunks
    })

    return {
        "answer": answer_text,
        "context_used": context_block,
        "model": model,
        "num_chunks_used": len(chunks),
        "sources": sources,
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        },
    }


def ask(query: str, k: int = 5) -> dict:
    """
    Full end-to-end RAG pipeline: retrieve + generate in one call.
    This is the function you'll call from app.py, agent.py, and evaluate.py.
    """
    chunks = retrieve_dense(query, k=k)
    result = generate_answer(query, chunks)
    result["query"] = query
    return result


def print_answer(result: dict) -> None:
    """Pretty-print a generate_answer() / ask() result."""
    print("=" * 60)
    print(f"QUERY: {result.get('query', '(not set)')}")
    print("=" * 60)
    print(f"\n{result['answer']}\n")
    print("─" * 60)
    print(f"Chunks used    : {result['num_chunks_used']}")
    print(f"Tokens used    : {result['usage']['total_tokens']} "
          f"(prompt: {result['usage']['prompt_tokens']}, "
          f"completion: {result['usage']['completion_tokens']})")
    print(f"Unique sources : {len(result['sources'])}")
    for arxiv_id, title, year in result["sources"]:
        print(f"  - {arxiv_id} — {title[:60]} ({year})")
    print("=" * 60)


# ── test suite ───────────────────────────────────────────
TEST_QUERIES = [
    "How does FedAvg aggregate client model updates?",
    "What is differential privacy and how is epsilon defined?",
    "What visual artifacts help detect GAN-generated faces?",
]


def run_test_suite():
    print("Running end-to-end RAG test suite...\n")
    total_cost_estimate = 0.0

    for query in TEST_QUERIES:
        result = ask(query, k=5)
        print_answer(result)
        print()

        # GPT-4o-mini pricing: ~$0.15/1M input, ~$0.60/1M output tokens
        cost = (
            result["usage"]["prompt_tokens"] * 0.15 / 1_000_000
            + result["usage"]["completion_tokens"] * 0.60 / 1_000_000
        )
        total_cost_estimate += cost

    print(f"\nEstimated total cost for this test run: ${total_cost_estimate:.4f}")

def generate_answer_stream(query: str, chunks: list[RetrievedChunk]):
    """
    Same as generate_answer(), but yields tokens as they arrive from
    the OpenAI API instead of waiting for the full response.

    This is a generator function (uses yield) — callers iterate over it
    to receive tokens one at a time as they're produced.
    """
    context_block = build_context_block(chunks)

    user_message = f"""Context:
{context_block}

Question: {query}

Answer the question using only the context above, with citations."""

    stream = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        stream=True,  # this is the only difference from generate_answer()
    )

    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


if __name__ == "__main__":
    run_test_suite()