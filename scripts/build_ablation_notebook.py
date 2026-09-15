"""Rebuild notebooks/ablation.ipynb from committed results/*.json files.

Run from repo root:
    python scripts/build_ablation_notebook.py
Optional execute (needs pandas/matplotlib/seaborn):
    jupyter nbconvert --to notebook --execute notebooks/ablation.ipynb --inplace
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "ablation.ipynb"


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [source]}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [source],
    }


cells = [
    md(
        """# RAG Ablation Study

## Introduction

This notebook is the recorded ablation for ResearchMind. Every chart loads a
JSON file already in `results/` — re-running cells does **not** call OpenAI
or rebuild Chroma.

Goal: isolate the contribution of each retrieval choice on a fixed golden set,
then show why hybrid is the production default even though dense wins recall.

## Methodology

- **Golden set**: 25 hand-verified Q/A pairs in `golden_qa.json`.
  Tiers: 11 factual (T1), 11 synthesis (T2), 3 multi-hop (T3).
- **Main RAGAS table** (`dense_eval.json` / `bm25_eval.json` / `hybrid_eval.json`):
  all **25** questions, `gpt-4o-mini` generator + RAGAS judge, k=5.
- **Agent RAGAS** (`agent_eval.json`): **Tier-3 only (3 questions)**.
  Do not stack it against the 25-Q hybrid row.
- **Chunking RAGAS** (`chunk_eval.json`): 10 Tier-1/2 questions, ephemeral
  indexes, three splitters (`src/chunk_ablation.py`).
- **Four-mode run** (`full_comparison.json`): same 25 questions through
  BM25 / dense / hybrid / agent for latency and tool-call traces (not RAGAS).
- **Latency microbenchmark** (`latency_benchmark.json`): 10 dense queries,
  retrieval vs generation split (first query includes BGE/Chroma cold start).
"""
    ),
    code(
        """import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

RESULTS_DIR = Path("results") if Path("results").exists() else Path("..") / "results"
assert RESULTS_DIR.exists(), f"Missing results/ relative to {Path.cwd()}"

METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
sns.set_theme(style="whitegrid")
"""
    ),
    md("## Section 1 — Dense retrieval baseline (25 questions)"),
    code(
        """with open(RESULTS_DIR / "dense_eval.json") as f:
    dense_baseline = json.load(f)

print(f"n_questions={dense_baseline.get('n_questions')}  retriever={dense_baseline.get('retriever')}  k={dense_baseline.get('k')}")
pd.DataFrame([dense_baseline["scores"]])
"""
    ),
    md("## Section 2 — Sparse vs dense vs hybrid (same 25 questions)"),
    code(
        """with open(RESULTS_DIR / "bm25_eval.json") as f:
    bm25_eval = json.load(f)
with open(RESULTS_DIR / "hybrid_eval.json") as f:
    hybrid_eval = json.load(f)

rows = []
for label, blob in [
    ("Dense", dense_baseline),
    ("BM25", bm25_eval),
    ("Hybrid (RRF + FlashRank)", hybrid_eval),
]:
    for metric in METRICS:
        rows.append({"Metric": metric, "Retriever": label, "Score": blob["scores"][metric]})

df = pd.DataFrame(rows)
display = df.pivot(index="Retriever", columns="Metric", values="Score").reindex(
    ["BM25", "Dense", "Hybrid (RRF + FlashRank)"]
)
display.round(3)
"""
    ),
    code(
        """plt.figure(figsize=(12, 6))
ax = sns.barplot(data=df, x="Metric", y="Score", hue="Retriever", palette="viridis")
plt.title("RAGAS on 25 golden questions", fontsize=16, pad=15)
plt.ylim(0.5, 1.05)
plt.ylabel("RAGAS score")
plt.xlabel("")
plt.legend(title="Retriever", bbox_to_anchor=(1.02, 1), loc="upper left")
for p in ax.patches:
    if p.get_height() > 0:
        ax.annotate(
            f"{p.get_height():.3f}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="baseline",
            fontsize=9,
            xytext=(0, 4),
            textcoords="offset points",
        )
plt.tight_layout()
plt.savefig(RESULTS_DIR / "ablation_plot.png", dpi=200)
plt.show()
"""
    ),
    md(
        """## Retrieval tradeoff

1. **BM25 is the weakest standalone retriever.** Context recall 0.764 vs dense 0.964
   (~20 point gap). It still wins some acronym-heavy hits (FedAvg) that embeddings miss.
2. **Dense is the best single-mode recall.** BGE-base cosine search is the right
   default if you only pick one retriever.
3. **Hybrid maximises faithfulness and precision** (0.954 / 0.934) by fusing the
   two pools with RRF k=60 then FlashRank. Context recall drops ~4 points vs dense
   because the cross-encoder is an aggressive filter.

**Production default: hybrid.** Cleaner context → fewer hallucinations. The extra
~100–150 ms of rerank is cheap next to generation (~5.4 s).

## Known failure modes (still true)

- **Survey dominance:** a broad question can return five chunks from one survey.
- **Year / metadata:** “When was FedAvg introduced?” fails when the year lives in
  sidecar metadata, not body text (`KNOWN_ISSUES.md`).
- **Fragmented metrics papers:** experimental tables split across chunks hurt
  context recall even when the generator infers a usable answer.
"""
    ),
    md(
        """## Section 3 — ReAct agent (Tier-3 only)

`src/evaluate_agent.py` scores the **3 multi-hop questions**, not the full 25.
Higher faithfulness here (0.979) is **not** proof the agent beats hybrid on the
whole golden set.

The LangGraph ReAct loop calls the hybrid search tool up to `recursion_limit=8`.
`classify_query` / `decompose_query` are a **UI transparency layer** — they do
not gate the agent.
"""
    ),
    code(
        """with open(RESULTS_DIR / "agent_eval.json") as f:
    agent_eval = json.load(f)

print("WARNING: agent RAGAS is 3 Tier-3 questions, not 25.")
pd.DataFrame(
    [
        {"setup": "Hybrid (25 Q)", **hybrid_eval["scores"]},
        {"setup": "Agent (3 Tier-3 Q)", **agent_eval["scores"]},
    ]
).round(3)
"""
    ),
    md(
        """## Section 4 — Four-mode run on all 25 questions

`full_comparison.json` ran BM25, dense, hybrid, and the agent on the **same 25
questions**. Rows have latency and (for the agent) tool-call counts. They do
**not** include per-question RAGAS — use the eval JSON files above for quality.
"""
    ),
    code(
        """with open(RESULTS_DIR / "full_comparison.json") as f:
    comparison = json.load(f)

latency_rows = []
for mode, items in comparison.items():
    for row in items:
        latency_rows.append(
            {
                "mode": mode,
                "tier": row.get("tier"),
                "tier_name": row.get("tier_name"),
                "elapsed_s": row.get("elapsed_s"),
                "tool_calls": row.get("tool_calls"),
                "n_sources": len(row.get("sources") or []),
            }
        )
lat = pd.DataFrame(latency_rows)

summary = (
    lat.groupby("mode")["elapsed_s"]
    .agg(n="count", mean_s="mean", p50_s="median", max_s="max")
    .reindex(["bm25", "dense", "hybrid", "agent"])
)
print("Latency on 25 golden questions")
summary.round(2)
"""
    ),
    code(
        """tier_lat = (
    lat.groupby(["mode", "tier_name"])["elapsed_s"]
    .mean()
    .unstack("tier_name")
    .reindex(["bm25", "dense", "hybrid", "agent"])
)
print("Mean latency (s) by difficulty tier")
tier_lat.round(2)
"""
    ),
    code(
        """agent_rows = lat[lat["mode"] == "agent"]
print("Agent tool calls on the 25-Q comparison run")
print(agent_rows["tool_calls"].describe().round(2))
print()
print("By tier:")
agent_rows.groupby("tier_name")["tool_calls"].agg(["mean", "max", "count"])
"""
    ),
    md(
        """## Section 5 — Chunking ablation

Naive fixed-256 splitting destroyed recall because equations and sentences were
cut mid-expression. Recursive 512 / 50 overlap is the production splitter.
This eval used 10 Tier-1/2 questions and an ephemeral Chroma per strategy
(`src/chunk_ablation.py`), not the full 25-Q index.
"""
    ),
    code(
        """with open(RESULTS_DIR / "chunk_eval.json") as f:
    chunk_eval = json.load(f)

chunk_df = pd.DataFrame(chunk_eval).T[METRICS]
print("Chunking RAGAS (10 questions)")
chunk_df.round(3)
"""
    ),
    code(
        """chunk_long = (
    chunk_df.reset_index()
    .rename(columns={"index": "strategy"})
    .melt(id_vars="strategy", var_name="Metric", value_name="Score")
)
plt.figure(figsize=(11, 5))
ax = sns.barplot(data=chunk_long, x="Metric", y="Score", hue="strategy", palette="mako")
plt.title("Chunking ablation (10 questions)", fontsize=15, pad=12)
plt.ylim(0, 1.05)
plt.legend(title="Splitter", bbox_to_anchor=(1.02, 1), loc="upper left")
for p in ax.patches:
    if p.get_height() > 0:
        ax.annotate(
            f"{p.get_height():.2f}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="baseline",
            fontsize=8,
            xytext=(0, 3),
            textcoords="offset points",
        )
plt.tight_layout()
plt.savefig(RESULTS_DIR / "chunk_ablation_plot.png", dpi=200)
plt.show()
"""
    ),
    md(
        """Documented split in paper `0412073` (fixed-size):

```
[chunk 63 ends]  ...Our world average is then χ = 0.1282 ± 0.0077 . (
[chunk 64 starts] 17) Introducing the latter result in Eq. (16)...
```

Equation 17 is split from its number. Recursive splitting kept the paragraph intact.
"""
    ),
    md("## Section 6 — Latency split (dense path, 10 queries)"),
    code(
        """with open(RESULTS_DIR / "latency_benchmark.json") as f:
    latency = json.load(f)

s = latency["summary"]
print(
    f"{s['n_queries']} queries | retrieval avg {s['retrieval']['avg']}s | "
    f"generation avg {s['generation']['avg']}s | total avg {s['total']['avg']}s"
)
print(
    "Generation is ~87% of steady-state latency. "
    "The first row in per_query includes model/index cold start."
)
pd.DataFrame(latency["per_query"])[
    ["query", "retrieval_latency_s", "generation_latency_s", "total_latency_s", "num_chunks"]
]
"""
    ),
    md(
        """## Takeaways

| Decision | Evidence | What we ship |
|---|---|---|
| Recursive 512/50 chunks | Context recall 0.94 vs 0.27 fixed-256 | `src/chunk.py` |
| Hybrid default | Faithfulness 0.954 vs dense 0.933 on 25 Qs | Streamlit + FastAPI `/query` |
| Agent for multi-hop only | Extra LLM loops; RAGAS on 3 hard Qs | Streamlit “Agentic” mode |
| Local BGE, not embedding API | No per-token embed cost; 440MB cold start | `sentence-transformers` |

Software-docs is a **separate** Chroma collection (`--corpus software_docs`).
It is not mixed into these 25-Q research numbers. Software-docs RAGAS was not
run for this notebook.
"""
    ),
]

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    },
    "cells": cells,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(notebook, indent=2) + "\n")
print(f"Wrote {OUT} ({len(cells)} cells)")
