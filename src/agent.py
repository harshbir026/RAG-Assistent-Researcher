import os
from contextvars import ContextVar
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage

from src.corpus_config import DEFAULT_CORPUS
from src.retrieve import retrieve_hybrid

load_dotenv()

active_corpus: ContextVar[str] = ContextVar("active_corpus", default=DEFAULT_CORPUS)


@tool
def search_research_papers(query: str) -> str:
    """
    Searches the active document corpus for the given query.
    Returns the top retrieved text chunks with their source ids.
    Always use this tool when asked about technical concepts in the corpus.
    """
    corpus = active_corpus.get()
    print(f"\n[Tool Execution] 🔍 Searching {corpus} for: '{query}'")

    chunks = retrieve_hybrid(query, k=5, corpus=corpus)

    if not chunks:
        return "No relevant documents found for this query."

    formatted_context = ""
    for i, chunk in enumerate(chunks, 1):
        formatted_context += (
            f"[Source {i} | id: {chunk.arxiv_id} | Title: {chunk.title[:50]}...]\n"
            f"{chunk.text}\n\n"
        )

    return formatted_context

# ── 2. Initialize LLM & Bind Tools ──────────────────────────────────
_openai_key = os.getenv("OPENAI_API_KEY")
if not _openai_key:
    raise RuntimeError(
        "OPENAI_API_KEY is not set. On Streamlit Cloud add it under "
        "Manage app → Settings → Secrets, then reboot."
    )
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=_openai_key)
tools = [search_research_papers]

# ── 3. Define the System Prompt ─────────────────────────────────────

system_prompt = """You are a specialized AI assistant. Your ONLY purpose is to answer questions based on the provided document corpus.
You have access to a tool called 'search_research_papers'.

STRICT GUARDRAILS:
1. OUT-OF-DOMAIN: If the user asks a question that is not related to the corpus (e.g. personal advice, creative writing), you MUST politely refuse to answer. Say: "I am a specialized assistant and can only answer questions related to the documents in my database."
2. HALLUCINATION PREVENTION: If you search the corpus and cannot find the answer, explicitly state that the information is not present. Do NOT use your internal training data to answer.
3. CITATIONS: When you provide an answer, you MUST cite your sources using the ids provided by the tool.

INSTRUCTIONS:
- Always use the search tool to gather context before answering factual or technical questions.
- If the user's question is complex, use the tool multiple times to gather different pieces of information before synthesizing your final answer."""
# ── 3b. Classification & Decomposition (for UI display only) ───────
import json

def classify_query(question: str) -> str:
    """Classify a question as 'simple' (single-source) or 'complex' (multi-hop)."""
    classifier_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = f"""Is this question simple (answerable from a single source) or complex (requires synthesising multiple sources)?
Answer with exactly one word: simple or complex.

Question: {question}
Answer:"""
    result = classifier_llm.invoke(prompt)
    answer = result.content.strip().lower()
    return "complex" if "complex" in answer else "simple"


def decompose_query(question: str) -> list[str]:
    """Break a complex question into 2-4 answerable sub-questions."""
    decomposer_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = f"""Break this complex research question into 2-4 simpler sub-questions that can each be answered from a single paper. Return ONLY a JSON array of strings, nothing else — no markdown, no explanation.

Question: {question}"""
    result = decomposer_llm.invoke(prompt)
    raw = result.content.strip().strip("```json").strip("```").strip()
    try:
        sub_qs = json.loads(raw)
        if isinstance(sub_qs, list) and all(isinstance(q, str) for q in sub_qs):
            return sub_qs
    except Exception:
        pass
    # fallback if the model didn't return clean JSON
    return [line.strip("-• ") for line in raw.split("\n") if line.strip()][:4]
# ── 4. Create the LangGraph Agent ───────────────────────────────────
# Removed the strict kwargs to avoid version conflicts!
research_agent = create_react_agent(llm, tools)
research_agent = research_agent.with_config({"recursion_limit": 8})

if __name__ == "__main__":
    print("🤖 Initializing ReAct Agent...")
    
    # A complex question that requires the agent to read and synthesize
    test_query = "What are the main challenges of federated learning with non-IID data? Cite the papers."
    print(f"\nUser: {test_query}\n")
    
    # Inject the system prompt dynamically as the first message!
    response = research_agent.invoke({
        "messages": [
            SystemMessage(content=system_prompt),
            HumanMessage(content=test_query)
        ]
    })
    
    print("\n" + "="*50)
    print("FINAL AGENT RESPONSE:")
    print("="*50)
    print(response["messages"][-1].content)
    print("="*50)