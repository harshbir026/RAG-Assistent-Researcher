## Known Limitations

- **Title/abstract extraction accuracy (~80%)**: Some PDFs have inconsistent
  font layouts that cause the title-extraction heuristic to capture the
  arXiv header stamp instead of the actual paper title. Full text and
  retrieval are unaffected.
- **Reference stripping (~95% reliable)**: A small number of papers with
  non-standard section ordering retain partial reference-list fragments
  as retrievable chunks.
- **Single-paper retrieval dominance**: Broad survey-style questions can
  return all top-k results from one dominant paper. This is correct
  per-query behavior but means diversity-aware reranking (planned) would
  improve breadth on synthesis-tier questions.
- **Weak retrieval on FL evaluation-methodology questions**: Corpus has no
  dedicated paper on evaluation metrics for FL convergence; this knowledge
  is scattered implicitly across results tables in other papers.
- **Query length bounds**: Queries must be 10-500 characters. Shorter or
  longer queries are rejected with a clear validation error rather than
  silently truncated or padded.
- **No hallucination defense beyond prompting**: Faithfulness is enforced
  via system prompt instructions and measured via RAGAS (0.93 faithfulness
  on the golden set), but there is no independent verification layer
  checking generated claims against retrieved text.

  ## 📊 Retrieval Ablation Study
To determine the optimal retrieval architecture, I evaluated three strategies against a 25-question golden dataset using the **RAGAS** framework. 

* **Dense Retrieval (BGE-768):** Exceptional recall (0.96), but occasionally surfaced semantically similar noise.
* **Sparse Retrieval (BM25):** High precision on exact terminology, but suffered a 20% drop in recall on semantic queries.
* **Hybrid + Cross-Encoder Reranking (FlashRank):** Fused using Reciprocal Rank Fusion (RRF). Maximized Context Precision (0.93) and Faithfulness (0.95) by aggressively filtering noise.

The Hybrid pipeline was selected for production, accepting a ~150ms latency overhead in exchange for near-perfect answer grounding.

### Agent Observability & Failure Analysis

Utilizing LangSmith production traces, an audit of 10 distinct agent execution graphs was conducted to evaluate the LangGraph ReAct control loop. Traces validated that for complex multi-hop queries requiring cross-paper synthesis, the orchestrator successfully orchestrated sequence tool invocations—autonomously generating secondary, refined search queries after assessing initial context retrieval.

However, trace analysis exposed two core systemic vulnerabilities:

1. **Semantic Query Drift:** This occurred on highly nuanced topics; when initial retrieval results contained peripheral jargon, the agent's secondary tool inputs occasionally over-indexed on noise, causing recursive loop delays. 
2. **Legacy Chunk Boundary Splits:** Phase 1 chunking issues were laid bare in the trace history. Fixed-size chunk partitions frequently truncated algorithmic updates and mathematical proofs across boundary lines. In the traces, this was visible as the agent receiving partial math expressions and attempting to reconstruct the underlying logic, increasing latency and slightly lowering faithfulness. 

This trace evidence directly informed our transition to a recursive, syntax-aware splitting mechanism to preserve technical continuity.