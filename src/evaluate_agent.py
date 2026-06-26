import sys
from unittest.mock import MagicMock

# ── MONKEY PATCH FOR RAGAS BUG ──────────────────────────────────────────
# Ragas looks for a VertexAI file that LangChain deleted. We create a fake 
# module in system memory so Ragas can import successfully without crashing.
sys.modules['langchain_community.chat_models.vertexai'] = MagicMock()
# ────────────────────────────────────────────────────────────────────────

import json
import os
from pathlib import Path
from dotenv import load_dotenv
from datasets import Dataset

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

# Import the agent and prompt
from src.agent import research_agent, system_prompt

load_dotenv()

print("🤖 Initializing Agent Evaluation...")

# ── RAGAS Modern Setup ───────────────────────────────────────────────
eval_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
eval_embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

# ── Load Tier 3 Questions ─────────────────────────────────────
with open("golden_qa.json") as f:
    all_qs = json.load(f)["questions"]

# Filter ONLY for Tier 3 (Multi-hop / Complex reasoning)
tier3_qs = [q for q in all_qs if q["tier"] == 3]
print(f"Loaded {len(tier3_qs)} Tier 3 (Multi-hop) questions.")

# ── Run Agent & Collect Traces ────────────────────────────────
rows = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

for i, q in enumerate(tier3_qs, 1):
    print(f"\n[{i}/{len(tier3_qs)}] Evaluating: {q['question']}")
    
    # Invoke Agent
    response = research_agent.invoke({
        "messages": [
            SystemMessage(content=system_prompt),
            HumanMessage(content=q["question"])
        ]
    })
    
    final_answer = response["messages"][-1].content
    
    # Extract ALL context gathered by the agent across multiple tool calls
    gathered_context = []
    for msg in response["messages"]:
        if isinstance(msg, ToolMessage):
            gathered_context.append(msg.content)
            
    # Fallback if agent answered without tools
    if not gathered_context:
        gathered_context = ["No tools used."]
        
    rows["question"].append(q["question"])
    rows["answer"].append(final_answer)
    rows["contexts"].append(gathered_context)
    rows["ground_truth"].append(q["ground_truth"])

# ── Evaluate with RAGAS ───────────────────────────────────────
print("\n📊 Running RAGAS Evaluation on Agent traces...")
ds = Dataset.from_dict(rows)

# Evaluate using the modern API
eval_res = evaluate(ds, metrics=metrics, llm=eval_llm, embeddings=eval_embeddings)

df = eval_res.to_pandas()
scores = df[["faithfulness", "answer_relevancy", "context_precision", "context_recall"]].mean().to_dict()

print("\n" + "="*50)
print("AGENT EVALUATION SCORES (TIER 3 ONLY)")
print("="*50)
for k, v in scores.items(): 
    print(f"{k:20s}: {v:.4f}")

# ── Save Results ──────────────────────────────────────────────
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
with open(RESULTS_DIR / "agent_eval.json", "w") as f:
    json.dump({"scores": scores}, f, indent=2)

print(f"\n✅ Agent evaluation complete! Saved to {RESULTS_DIR / 'agent_eval.json'}")