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