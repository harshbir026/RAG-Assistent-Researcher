import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage

# Import your optimized Phase 3 hybrid retriever
from src.retrieve import retrieve_hybrid

load_dotenv()

# ── 1. Define the Tool ──────────────────────────────────────────────
@tool
def search_research_papers(query: str) -> str:
    """
    Searches a database of academic research papers for the given query.
    Returns the top retrieved text chunks with their arXiv IDs.
    Always use this tool when asked about technical concepts, federated learning, or deepfakes.
    """
    print(f"\n[Tool Execution] 🔍 Searching corpus for: '{query}'")
    
    # Use the Hybrid Retriever (Dense + BM25 + Cross-Encoder)
    chunks = retrieve_hybrid(query, k=5)
    
    if not chunks:
        return "No relevant papers found for this query."
    
    # Format the retrieved chunks so the LLM can read them
    formatted_context = ""
    for i, chunk in enumerate(chunks, 1):
        formatted_context += f"[Paper {i} | arXiv: {chunk.arxiv_id} | Title: {chunk.title[:50]}...]\n{chunk.text}\n\n"
    
    return formatted_context

# ── 2. Initialize LLM & Bind Tools ──────────────────────────────────
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
tools = [search_research_papers]

# ── 3. Define the System Prompt ─────────────────────────────────────
system_prompt = """You are an expert AI research assistant. Your task is to answer user questions based on academic literature.
You have access to a tool called 'search_research_papers'. 

INSTRUCTIONS:
1. Always use the search tool to gather context before answering factual or technical questions.
2. If the user's question is complex, you may use the tool multiple times to gather different pieces of information.
3. When you provide an answer, you MUST cite your sources using the arXiv IDs provided by the tool (e.g., [arXiv: 1234.5678]).
4. If you cannot find the answer using the tool, explicitly state that the information is not present in your research corpus. Do not hallucinate."""

# ── 4. Create the LangGraph Agent ───────────────────────────────────
# Removed the strict kwargs to avoid version conflicts!
research_agent = create_react_agent(llm, tools)

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