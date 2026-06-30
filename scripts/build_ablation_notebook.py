import json

cells = [
    {
        "cell_type": "markdown", 
        "metadata": {}, 
        "source": ["# RAG Ablation Study\n\n## Introduction\nThis notebook documents a systematic ablation study comparing retrieval and generation strategies for the RAG Research Assistant project. The goal is to isolate the contribution of each architectural decision — dense vs sparse vs hybrid retrieval — against a fixed 25-question golden evaluation set spanning 3 difficulty tiers.\n\n## Methodology\n- **Golden set**: 25 hand-verified question/answer pairs (`golden_qa.json`), split into Tier 1 (factual), Tier 2 (synthesis), and Tier 3 (multi-hop).\n- **Evaluation framework**: RAGAS — faithfulness, answer_relevancy, context_precision, context_recall — computed identically across every configuration using `src/evaluate.py`.\n- **Baselines**:\n  - **Dense**: BGE embeddings + ChromaDB cosine similarity.\n  - **Sparse (BM25)**: Lexical keyword matching via rank-bm25.\n  - **Hybrid**: Dense + BM25 combined via Reciprocal Rank Fusion (RRF), rescored with a FlashRank cross-encoder.\n"]
    },
    {
        "cell_type": "code", 
        "metadata": {}, 
        "outputs": [], 
        "source": ["import json\nimport pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nfrom pathlib import Path\n\nRESULTS_DIR = Path(\"../results\")"]
    },
    {
        "cell_type": "markdown", 
        "metadata": {}, 
        "source": ["## Section 1 — Dense Retrieval Baseline (Day 10)"]
    },
    {
        "cell_type": "code", 
        "metadata": {}, 
        "outputs": [], 
        "source": ["with open(RESULTS_DIR / \"dense_eval.json\") as f:\n    dense_baseline = json.load(f)\n\npd.DataFrame([dense_baseline[\"scores\"]])"]
    },
    {
        "cell_type": "markdown", 
        "metadata": {}, 
        "source": ["## Section 2 & 3 — Sparse, Hybrid, and Cross-Encoder Reranking"]
    },
    {
        "cell_type": "code", 
        "metadata": {}, 
        "outputs": [], 
        "source": [
            "with open(RESULTS_DIR / \"bm25_eval.json\") as f:\n",
            "    bm25_eval = json.load(f)\n",
            "with open(RESULTS_DIR / \"hybrid_eval.json\") as f:\n",
            "    hybrid_eval = json.load(f)\n\n",
            "data = []\n",
            "metrics = [\"faithfulness\", \"answer_relevancy\", \"context_precision\", \"context_recall\"]\n",
            "for m in metrics:\n",
            "    data.append({\"Metric\": m, \"Retriever\": \"Dense\", \"Score\": dense_baseline[\"scores\"][m]})\n",
            "    data.append({\"Metric\": m, \"Retriever\": \"BM25\", \"Score\": bm25_eval[\"scores\"][m]})\n",
            "    data.append({\"Metric\": m, \"Retriever\": \"Hybrid (Reranked)\", \"Score\": hybrid_eval[\"scores\"][m]})\n\n",
            "df = pd.DataFrame(data)\n",
            "sns.set_theme(style=\"whitegrid\")\n",
            "plt.figure(figsize=(12, 6))\n",
            "ax = sns.barplot(x=\"Metric\", y=\"Score\", hue=\"Retriever\", data=df, palette=\"viridis\")\n",
            "plt.title(\"RAGAS Evaluation: Retrieval Ablation Study\", fontsize=16, pad=15)\n",
            "plt.ylim(0.5, 1.0)\n",
            "plt.ylabel(\"RAGAS Score\")\n",
            "plt.xlabel(\"Evaluation Metric\")\n",
            "plt.legend(title=\"Retrieval Strategy\", bbox_to_anchor=(1.05, 1), loc='upper left')\n",
            "for p in ax.patches:\n",
            "    if p.get_height() > 0:\n",
            "        ax.annotate(f\"{p.get_height():.3f}\", (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='baseline', fontsize=10, color='black', xytext=(0, 5), textcoords='offset points')\n",
            "plt.tight_layout()\n",
            "plt.savefig(RESULTS_DIR / \"ablation_plot.png\", dpi=300)\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown", 
        "metadata": {}, 
        "source": ["## Ablation Analysis & Tradeoffs\n\n1. **BM25 (Sparse) is the weakest standalone retriever:** It scored the lowest across all metrics, suffering a massive 20% drop in Context Recall compared to Dense retrieval.\n2. **Dense Retrieval is the best single-mode baseline:** It achieved exceptional Context Recall (0.964), proving BGE embeddings map semantic intent well.\n3. **Hybrid + Cross-Encoder maximizes Precision and Faithfulness:** Fusing dense and sparse pools via RRF and reranking with a Cross-Encoder pushed Context Precision to 0.933 and Faithfulness to 0.954.\n\n**Production Decision:** The Hybrid pipeline sacrifices a marginal amount of recall (~4%) compared to raw Dense retrieval, but the resulting context block is much cleaner, causing the LLM to hallucinate less. We adopt Hybrid Retrieval as the default strategy moving forward.\n\n## Failure Analysis\n- **Over-concentration on Surveys:** Broad questions resulted in the top 5 chunks all being pulled from a *single* comprehensive survey paper. While factually correct, this lacks source diversity.\n- **Evaluation Methodology Blindspots:** The corpus lacks dedicated papers strictly on FL evaluation metrics. The LLM successfully inferred answers from experimental result tables, but Context Recall scores dipped slightly on these queries because the information was highly fragmented."]
    },
    {
        "cell_type": "markdown", 
        "metadata": {}, 
        "source": ["## Section 4 — ReAct Agent (Multi-Hop Reasoning)\n\n*To be completed in Phase 4.*"]
    },
    {
        "cell_type": "code", 
        "metadata": {}, 
        "outputs": [], 
        "source": ["# results/agent_eval.json — to be generated"]
    },
    {
        "cell_type": "markdown", 
        "metadata": {}, 
        "source": ["## Section 5 — Tier-by-Tier Comparison\n\n*To be completed once all configurations are evaluated.*"]
    }
]

notebook = {
    "cells": cells,
    "metadata": {},
    "nbformat": 4,
    "nbformat_minor": 5
}

with open("notebooks/ablation.ipynb", "w") as f:
    json.dump(notebook, f, indent=2)

print("✅ Perfect notebook restored and generated successfully!")