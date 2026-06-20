import os
import time
from dotenv import load_dotenv
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import RateLimitError, APIError
from src.logger import log_query, get_logger

logger = get_logger(__name__)

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
3. If no context was retrieved at all (the context says "No relevant context
   was found"), respond with exactly: "I don't have enough information in my
   knowledge base to answer this question." Do not attempt to answer from
   general knowledge.
4. If different papers disagree or present different approaches, mention this.
5. Keep your answer focused and well-organized. Use bullet points if comparing
   multiple papers or methods.
6. Do not fabricate paper titles, authors, or findings not present in the context.
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

@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=2, max=20),
    retry=retry_if_exception_type((RateLimitError, APIError)),
    reraise=True,
)
def _call_openai_with_retry(model: str, temperature: float, messages: list):
    """
    Calls the OpenAI chat completion API with exponential backoff retry.
    Retries up to 4 times on RateLimitError or APIError, waiting
    2s, 4s, 8s, 16s (capped at 20s) between attempts.
    """
    return client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=messages,
    )
def generate_answer(
    query: str,
    chunks: list[RetrievedChunk],
    model: str = MODEL_NAME,
    temperature: float = TEMPERATURE,
) -> dict:
    start_time = time.time()
    chunk_ids = [f"{c.arxiv_id}_chunk_{c.chunk_index}" for c in chunks]

    try:
        context_block = build_context_block(chunks)

        user_message = f"""Context:
{context_block}

Question: {query}

Answer the question using only the context above, with citations."""

        response = _call_openai_with_retry(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )

        answer_text = response.choices[0].message.content
        elapsed = time.time() - start_time

        log_query(
            query=query,
            chunk_ids=chunk_ids,
            answer_length=len(answer_text),
            latency_seconds=elapsed,
            status="ok",
        )

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

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"generate_answer failed for query '{query[:50]}...': {e}")
        log_query(
            query=query,
            chunk_ids=chunk_ids,
            answer_length=0,
            latency_seconds=elapsed,
            status="error",
            error=str(e),
        )
        raise


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