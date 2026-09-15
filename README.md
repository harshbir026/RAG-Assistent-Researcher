---
title: ResearchMind
emoji: ⚡
colorFrom: indigo
colorTo: cyan
sdk: docker
app_port: 8501
pinned: false
license: mit
short_description: Agentic RAG over federated learning papers and software docs
---

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![LangGraph](https://img.shields.io/badge/Agent-LangGraph-1C3C3C?style=flat-square)
![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-FF6F00?style=flat-square)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-412991?style=flat-square&logo=openai)
![RAGAS](https://img.shields.io/badge/Eval-RAGAS-8A2BE2?style=flat-square)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat-square&logo=docker)
![CI](https://github.com/harshbir026/RAG-Assistent-Researcher/actions/workflows/ci.yml/badge.svg)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

# ResearchMind — Agentic RAG Research Assistant

🔗 **Live Demo:** [https://rag-assistent-researcher.streamlit.app](https://rag-assistent-researcher.streamlit.app) 

A retrieval-augmented research assistant over **251 arXiv papers (10,715 chunks)**
spanning federated learning, privacy-preserving ML, and deepfake detection. The
system combines hybrid retrieval (BM25 + BGE dense embeddings + FlashRank
cross-encoder reranking, fused with Reciprocal Rank Fusion) with a **self-directed
LangGraph ReAct agent** for multi-hop research questions — evaluated on a
hand-verified, 25-question golden set using RAGAS, and fully traced with LangSmith.


---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                          ResearchMind Pipeline                       │
└──────────────────────────────────────────────────────────────────────┘
│
▼
┌──────────────────┐     arXiv API — federated learning, privacy ML,
│  Corpus Ingestion │ ──► deepfake detection (251 PDFs)
└─────────┬─────────┘     urllib.request.urlretrieve (arxiv lib's
▼                         download_pdf() was removed upstream)
┌──────────────────┐     PyMuPDF — title (largest font, ~80% accurate),
│  PDF Parsing      │ ──► abstract (regex), references stripped from
└─────────┬─────────┘     last 40% of doc only
▼
┌──────────────────┐     Recursive splitter, tiktoken-measured,
│  Chunking         │ ──► 512 tokens / 50 overlap. Recursive-512 hit
└─────────┬─────────┘     94.0% context recall vs 28.5% for fixed-256
▼
┌──────────────────┐     BAAI/bge-base-en-v1.5, instruction-prefixed
│  Embedding Store  │ ──► (doc vs query prefixes), 10,715 chunks
└─────────┬─────────┘     ChromaDB, hnsw:space=cosine
▼
┌─────────────────────────────────────────────────────────┐
│         Self-Directed LangGraph ReAct Agent              │
│         (decides for itself how many times to search)    │
│                                                           │
│   ┌─────────────┐                    ┌────────────────┐ │
│   │  Direct RAG  │                    │  Multi-round    │ │
│   │  (1 search)  │                    │  search + tool  │ │
│   │              │                    │  loop (≤8 turns)│ │
│   └──────┬──────┘                    └────────┬───────┘ │
└──────────┼────────────────────────────────────┼─────────┘
▼                                     ▼
┌──────────────────┐              ┌──────────────────┐
│ BM25 + Dense →    │              │ Same hybrid       │
│ RRF (k=60) →      │              │ retriever, called  │
│ FlashRank rerank  │              │ as a LangChain tool│
└─────────┬─────────┘              └────────┬─────────┘
└──────────────┬───────────────┘
▼
┌──────────────────┐     GPT-4o-mini, cited context, SSE streaming
│  Answer Generation│ ──► LangSmith traces every step (found the
└─────────┬─────────┘     chunk-boundary bug slowing the agent down)
▼
┌───────────────────────────────────────┐
│         FastAPI REST + SSE            │
└──────────────┬────────────┬───────────┘
▼            ▼
┌───────────────┐  ┌────────────────────┐
│  Streamlit UI  │  │  RAGAS Eval Suite  │
│  (routing +    │  │  (25-Q golden set, │
│   trace links) │  │   4-mode ablation) │
└───────────────┘  └────────────────────┘
```

---

## Key Features

- **Hybrid retrieval** — BM25 (`rank-bm25`) and BGE dense embeddings merged with Reciprocal Rank Fusion (k=60), then re-scored by a local FlashRank cross-encoder — two-stage bi-encoder → cross-encoder pipeline, no reranking API cost
- **Self-directed agent, not a hardcoded router** — the LangGraph ReAct agent decides for itself, turn by turn, whether it needs another search — this is an intentional architecture decision that diverged from an earlier "classifier routes to agent" design once the ReAct loop proved capable of deciding on its own
- **Empirically-tuned, not guessed** — the retrieval similarity threshold was raised from 0.3 to 0.55 only after an adversarial off-topic query ("best pizza recipes in Italy") exposed that 0.3 let lexically-overlapping noise through
- **Chunking backed by a real ablation, not intuition** — recursive splitting at 512 tokens/50-token overlap measured at 94.0% context recall vs 28.5% for naive fixed-256 chunking, with a documented real failure case (an equation split mid-expression across two chunks)
- **Full observability** — every retrieval, tool call, and generation step traced in LangSmith; traces were used to catch a chunk-boundary bug that was forcing the agent to spend extra reasoning steps reassembling truncated equations
- **Streaming end-to-end** — GPT-4o-mini tokens stream through a FastAPI SSE endpoint and directly into the Streamlit UI via `st.empty()`, with sources rendered before generation even starts (retrieval is ~1% of total latency; no reason to make the user wait for it)
- **Production hygiene, tested against real failure** — layered failure handling per LLM call, a pytest suite with an environment-aware `requires_chromadb` marker, Docker Compose (build context cut from 4.33GB to ~7MB via `.dockerignore`), and GitHub Actions CI that took 4 real iterations to go green
- **Honest, documented limitations** — an unresolved citation-consistency bug (arXiv publication year is metadata, not text content, so "what year" questions don't reliably retrieve the source paper) is logged in `KNOWN_ISSUES.md` rather than hidden
- **Corpus-agnostic pipeline** — `--corpus software_docs` points the same chunker, hybrid retriever, and ReAct agent at 30 software/API notes in a **separate** Chroma collection. The only new parameter was markdown-aware splitting so fenced code blocks are never cut in half (the same failure class as equation-splitting on papers)

---

## Setup

### Requirements
- Python 3.11+
- OpenAI API key
- (Optional) LangSmith API key for tracing
- (Optional) Docker
- A **second, isolated virtual environment** for evaluation only (see note below)

### Local

```bash
git clone https://github.com/harshbir026/RAG-Assistent-Researcher
cd RAG-Assistent-Researcher
python3.11 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # add your API keys

# build the corpus (one-time, ~20-30 min total)
python src/ingest.py          # downloads 251 papers from arXiv
python src/parse.py           # PyMuPDF extraction, JSON sidecars
python src/chunk.py           # recursive chunking, 512 tokens / 50 overlap
python src/chunk.py --find-failures   # prints real equation-split + code-fence evidence
python src/embed.py           # embeds 10,715 chunks into ChromaDB (~15 min CPU)

# secondary corpus (optional — proves the pipeline is corpus-agnostic)
python src/ingest.py --corpus software_docs
python src/chunk.py --corpus software_docs
python src/embed.py --corpus software_docs
# python src/evaluate.py --corpus software_docs   # RAGAS; costs API, 5 golden Qs

# run it
streamlit run app.py          # chat UI + evaluation dashboard
# — or —
uvicorn api:app --reload      # FastAPI backend (SSE streaming)
```

> **Evaluation environment note:** `evaluate.py` runs in a *separate* venv
> (`venv_eval`) pinned to `langchain-core==0.3.86`, `langchain-community==0.3.27`,
> `langchain-openai==0.3.27`. RAGAS's legacy embeddings interface conflicts with
> the modern LangChain/LangGraph stack the rest of the project uses — isolating
> the two avoids a dependency deadlock rather than downgrading the whole project.

### Docker

```bash
docker compose up --build
# FastAPI backend → http://localhost:8000
# Streamlit UI    → http://localhost:8501
```

Both services share a volume-mounted `./data` directory. `.dockerignore` excludes
`venv/` and `data/`, keeping the build context around 7MB instead of 4.33GB.

### Environment Variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | Yes | Embeddings-free generation, classification, and agent reasoning |
| `LANGCHAIN_API_KEY` | No | Enables LangSmith tracing |
| `LANGCHAIN_TRACING_V2` | No | Set `true` to turn tracing on |
| `LANGCHAIN_PROJECT` | No | LangSmith project name (`rag-research-assistant`) |

---

## Tech Stack

| Layer | Technology | Reason |
|---|---|---|
| Corpus ingestion | arXiv API (`arxiv`) | Direct access to 3 target domains; `download_pdf()` was removed in newer package versions, worked around with `urllib.request.urlretrieve(paper.pdf_url, ...)` — more version-resilient anyway |
| PDF parsing | PyMuPDF (`fitz`) | Title via largest font on page 1 (~80% accuracy), abstract via regex, references stripped only from the last 40% of the document to avoid false positives from in-text mentions |
| Chunking | LangChain `RecursiveCharacterTextSplitter` + `tiktoken` (`cl100k_base`) | 512-token / 50-token-overlap chunks, size measured in tokens not characters (dense math notation packs far more tokens per character than plain English) |
| Dense embeddings | `BAAI/bge-base-en-v1.5` (768-dim, ~440MB) | Free, local, outperforms OpenAI Ada on MTEB retrieval benchmarks for technical text; uses BGE's instruction prefixes (`"Represent this passage for retrieval: "` for documents, `"Represent this sentence for searching: "` for queries) |
| Vector store | ChromaDB (persistent) | Local, `hnsw:space=cosine`, `upsert` for safe re-runs, 10,715 chunks |
| Sparse retrieval | `rank-bm25` (`BM25Okapi`) | Wins on acronym-heavy exact matches; found a real case where it *fails* — see [Ablation](#ablation-results-real-measured-numbers) |
| Reranking | FlashRank (local cross-encoder, `ms-marco-MiniLM` family) | Two-stage bi-encoder→cross-encoder pipeline; local/CPU/free instead of a SaaS reranker (Cohere) that adds latency, cost, and a privacy surface |
| Fusion | Reciprocal Rank Fusion (k=60) | Scale-invariant way to merge cosine similarity and unbounded BM25 scores without normalizing them onto a shared scale |
| Generation | OpenAI GPT-4o-mini | ~20x cheaper than GPT-4o for grounded Q&A where the quality gap is small; `temperature=0.1` for reproducible evaluation |
| Agent | LangGraph ReAct agent | Self-directed control loop (not a hardcoded classifier→decomposer pipeline) that decides autonomously how many search rounds a question needs, capped at `recursion_limit=8` |
| Tracing | LangSmith | Full pipeline + agent tool-call visibility; used to find a real chunk-boundary bug slowing agent reasoning |
| Evaluation | RAGAS | Faithfulness, answer relevancy, context precision, context recall — required a monkey-patched import and a dedicated venv to run at all (see [Real Engineering Challenges](#real-engineering-challenges)) |
| Backend | FastAPI + SSE | Typed, async, token-streaming REST endpoint; Pydantic field constraints reject malformed requests before any retrieval/generation cost is incurred |
| UI | Streamlit | Custom dark theme, live token streaming via `st.empty()`, agent routing/sub-question transparency, one-click LangSmith trace links |
| Testing | pytest + pytest-mock | 12 tests total; 5 marked `requires_chromadb` (skipped in CI, pass locally), 7 run in CI |
| CI | GitHub Actions | ruff lint + pytest on every push — took 4 real iterations to get green |
| Deployment | Streamlit Community Cloud (live), Docker Compose locally | Index is gitignored; Cloud downloads `Harshbirr/researchmind-index` on first boot. Hugging Face Docker Spaces need a paid plan — do not use them for this demo |

---

## Project Structure

```
rag-research-assistant/
├── data/
│   ├── papers/                 # 251 downloaded PDFs (primary corpus)
│   ├── parsed/                 # PyMuPDF JSON sidecars
│   ├── chunks/                 # all_chunks.jsonl — 10,715 chunks; software_docs.jsonl
│   ├── software_docs/          # 30 original teaching notes (secondary corpus)
│   └── chroma_db/               # persistent vector store (separate collections)
├── src/
│   ├── corpus_config.py         # --corpus flag: research vs software_docs
│   ├── ingest.py                # arXiv download (urllib fallback)
│   ├── parse.py                 # PyMuPDF extraction + metadata
│   ├── chunk.py                 # recursive + markdown-aware + --find-failures
│   ├── embed.py                 # BGE embedding + ChromaDB store
│   ├── retrieve.py              # dense / BM25 / hybrid (RRF), corpus-aware
│   ├── rerank.py                # FlashRank cross-encoder
│   ├── generate.py              # LLM call, streaming
│   ├── agent.py                 # LangGraph ReAct agent + display-layer helpers
│   └── evaluate.py              # RAGAS + golden set runner (venv_eval only)
├── notebooks/
│   └── ablation.ipynb           # BM25 vs dense vs hybrid vs agent vs chunking
├── tests/
│   ├── test_retrieval.py        # includes requires_chromadb-marked tests
│   ├── test_hybrid.py
│   ├── test_reranker.py
│   ├── test_agent.py
│   ├── test_markdown_chunker.py # code fences never split
│   └── test_corpus_config.py
├── logs/
│   ├── queries.jsonl             # structured, per-query
│   └── app.log                   # human-readable
├── .github/workflows/ci.yml     # lint + test, requires_chromadb-aware
├── .dockerignore                 # keeps build context ~7MB
├── app.py                        # Streamlit UI
├── api.py                        # FastAPI endpoint (health check, SSE, /query/agent)
├── golden_qa.json                 # 25 hand-verified Q&A pairs (research)
├── golden_qa_software.json        # 5 Q&A pairs (software docs generalization)
├── KNOWN_ISSUES.md                # citation-consistency bug, documented not hidden
├── LICENSE
├── Dockerfile
├── docker-compose.yml
├── requirements.txt               # main pipeline
├── requirements_eval.txt          # venv_eval, RAGAS-compatible pins
├── .env.example
└── README.md
```

---

## Evaluation

### Golden set design

25 questions, hand-written and individually **verified against real retrieval
output before any ground-truth answer was written** — never the reverse. Split two
ways:

| Dimension | Breakdown |
|---|---|
| Domain | 10 federated learning · 8 privacy ML · 7 deepfake detection |
| Question type | Factual · Comparison · Summarisation · Multi-hop (5 questions — hardest category, each requires synthesising 2+ papers) |

Building the set surfaced three real limitations, documented rather than
smoothed over: weak retrieval on "evaluation metrics" questions (similarity
0.72–0.74 vs a typical 0.78+, tied to what the corpus actually discusses
explicitly vs implicitly); reference-list fragments occasionally surviving the
40%-cutoff heuristic; and a single dominant paper crowding out top-5 diversity
on broad survey-style questions.

### Ablation results — real measured numbers

RAGAS scores, all 25 golden questions, `gpt-4o-mini` generation + judge:

| Retrieval mode | Faithfulness | Answer Relevancy | Context Precision | Context Recall |
|---|---|---|---|---|
| BM25 only | 0.889 | 0.808 | 0.794 | 0.764 |
| **Dense (BGE)** | 0.933 | 0.876 | 0.901 | **0.964** |
| **Hybrid (BM25 + Dense + FlashRank rerank)** | **0.954** | — | **0.933** | 0.920 |

Hybrid retrieval is not a strict win on every axis — it's a genuine
**precision/recall tradeoff**: faithfulness and precision both improve, but
recall drops ~4 points relative to dense-only, because the cross-encoder is an
aggressive filter that can push marginally relevant chunks just outside the
top-k cutoff. It was still chosen as the production default: ~100–150ms of
added latency for a meaningfully cleaner, more faithful context block.

### Chunking ablation — the real numbers behind the strategy

| Chunking strategy | Context Recall | Faithfulness |
|---|---|---|
| Fixed-size, 256 tokens | 28.5% | — |
| Fixed-size, 512 tokens | 66.6% | — |
| **Recursive, 512 tokens / 50 overlap** | **94.0%** | **96%** |

Real, documented failure case behind the decision — paper `0412073`:

```
[Fixed-size chunk 63 ends with]: ...Our world average is then χ = 0.1282 ± 0.0077 . (
[Fixed-size chunk 64 starts with]: 17) Introducing the latter result in Eq. (16)...
```

Equation 17 is split across the two chunks it's referenced in; a query asking
for that world average would retrieve one chunk with the number and no context,
or the other with context and no number. Recursive splitting kept the
surrounding paragraph — and the equation — intact.

### Agent architecture — a real design correction

The system originally planned an external classifier → decomposer → router
pipeline in front of the agent. Building the actual LangGraph ReAct loop
revealed it didn't need one: it's a **self-directed control loop** that decides
on its own, turn by turn, whether another search is needed. `classify_query()`
and `decompose_query()` still exist, but only as a **UI transparency layer** —
they run in parallel with the agent to show the user what it's likely to do,
not to gate what it actually does.

| Test case | Sub-questions shown | Searches executed |
|---|---|---|
| "What year was FedAvg introduced?" | 0 | 1 |
| "Compare FedAvg and pFedMe on non-IID data" | 4 | 2 |

Every agentic query links directly to its LangSmith trace via
`collect_runs()` + `share_run()` — the same mechanism used to manually find and
fix a chunk-boundary bug that was making the agent spend extra reasoning steps
reassembling truncated equations mid-retrieval.

Full methodology, plots, 4-mode latency, agent (Tier-3 only), and chunking ablation: [`notebooks/ablation.ipynb`](notebooks/ablation.ipynb). Rebuild with `python scripts/build_ablation_notebook.py`.

---

## Observability

Every query — direct RAG and agentic — is traced end-to-end in LangSmith.
Tracing wasn't optional once the agent introduced a non-deterministic control
loop (it might search once or four times); print-statement debugging doesn't
scale for that. Traces were reviewed manually and were the actual source of
the chunk-boundary fix above, not just a demo feature bolted on afterward.

---

## Testing & CI

```bash
pip install pytest pytest-mock
pytest tests/ -v -m "not requires_chromadb"   # CI-safe subset
pytest tests/ -v                              # full suite, local only
```

12 tests total; 5 are marked `requires_chromadb` and skipped in CI (no vector
store exists in a fresh checkout — `data/` is gitignored), 7 run in CI. Getting
CI green took 4 real iterations: whole-file `--ignore` missed individually
marked tests inside a shared file; two files (`test_hybrid.py`,
`test_reranker.py`) turned out to be debug scripts with retrieval calls at
module level, not real pytest tests — they crashed collection before any
marker could apply, and were rewritten as real assertions while keeping a
standalone debug-print mode; and `pytest.ini` itself was never committed,
silently unrecognizing the custom marker even after the code was fixed.

GitHub Actions runs `ruff check .` and the CI-safe subset on every push and PR
to `main`.

---

## Performance & Latency

**Direct RAG, warm (Day-14 microbenchmark):**

| Stage | Latency |
|---|---|
| Retrieval (steady-state, model warm) | ~83ms avg (64–111ms range) |
| Generation (GPT-4o-mini) | ~5.4s avg (2.4–8.4s range) — **~87% of total time** |
| Cold start (one-time BGE model load) | ~7.4s |

An early measurement bug is worth mentioning: the first pass blended the
one-time BGE cold-start cost into a "retrieval latency" average, making
retrieval look 10x slower than it is. Caught by noticing query #1 was a 100x
outlier vs every other query, and fixed by separating cold-start from
steady-state before reporting the number.

**Full 25-question, 4-mode comparison, same corpus state, same run (Day-35):**

| Mode | Avg latency |
|---|---|
| Dense | 7.32s |
| Hybrid | 7.55s |
| BM25 | 7.81s |
| Agent | 12.13s (~60% overhead vs standard retrieval modes) |

**Agent latency by complexity (via `POST /query/agent`):**

| Query type | Avg latency |
|---|---|
| Simple (1 search) | ~4.6s (3.3–5.4s) |
| Complex (2 searches, decomposed) | ~13.3s (12.1–15.4s) |

The complex-query cost isn't arbitrary — it's 4 sequential LLM calls
(classification, decomposition, the agent's own ReAct reasoning across 2
searches, then synthesis). The `/health` endpoint reports live corpus size and
model: `GET /health → {"status": "ok", "corpus_size": N, "model": "BGE-base"}`.

**Cost (RAGAS evaluation, gpt-4o-mini judge):** a full 25-question RAGAS pass
costs ~$1.50–2 and takes ~25 minutes; a single-question debug run costs ~$0.05
and was used repeatedly during the dependency debugging below. A single
generation call (5 retrieved chunks) costs ~$0.0003–0.0006.

---

## Real Engineering Challenges

The kind of debugging that only shows up in a project run for 35+ days, not a
weekend demo:

| Issue | Root cause | Fix |
|---|---|---|
| `'Result' object has no attribute 'download_pdf'` | `arxiv` package removed the method in a newer version | Switched to `urllib.request.urlretrieve(paper.pdf_url, ...)` |
| `ModuleNotFoundError: langchain.text_splitter` | LangChain moved splitters to a separate package | `pip install langchain-text-splitters`, updated imports |
| Similarity threshold (0.3) let noise through | Adversarial test ("pizza recipes") scored 0.43–0.46 due to lexical overlap on words like "best" — real matches scored 0.72+ | Raised threshold to 0.55 based on the empirical gap, not intuition |
| RAGAS crashed on import | Hardcoded import of a deprecated `langchain_community.chat_models.vertexai` module | 7-root-cause investigation; fixed with a dedicated `venv_eval`, pinned versions, an adapter class for the legacy/modern embeddings API mismatch, and a `sys.modules` monkey-patch to bypass the dead import without downgrading the main project |
| `context_precision`/`context_recall` silently scored 0.0 | RAGAS expects `contexts` as `List[str]`; passed `RetrievedChunk` objects instead | Extract `.text` before passing to RAGAS |
| Circular import between `retrieve.py` and `rerank.py` | Module-level import of a dataclass created a dependency cycle | Scoped, runtime-local import inside `retrieve_hybrid()` |
| Unescaped HTML injection risk in Streamlit | Academic text and LLM answers interpolated into custom HTML for styling — papers are full of `<`, `>`, `&` in equations and pseudocode | `html.escape()` on all dynamic content before interpolation |
| Jupyter notebook JSON corruption | Manual terminal text injection broke the `.ipynb`'s underlying JSON on unicode characters | Wrote a script to construct the notebook programmatically from Python dicts, dumped via `json.dump` |
| Docker build context was 4.33GB | No `.dockerignore` — `venv/` and `data/` were being copied into every build | Added `.dockerignore`, cut context to ~7MB |
| Same factual question, 3 different citations across runs | Investigated in two layers: (1) the actual FedAvg paper was missing from the corpus — confirmed via `grep` and top-10 similarity inspection, fixed by ingesting it directly; (2) even after the fix, its publication year is metadata, not chunk text, so "what year" questions still can't reliably retrieve it on content match alone | Layer 1 fixed and committed; layer 2 documented as an open issue in `KNOWN_ISSUES.md` rather than papered over |
| CI stayed red for 4 iterations | ChromaDB-dependent tests crashed a fresh checkout; whole-file `--ignore` missed individually marked tests; 2 "test" files were actually debug scripts crashing pytest's collection phase; `pytest.ini` was never committed, so a custom marker was silently unrecognized | Rewrote debug scripts as real assertions, added a `requires_chromadb` marker, committed `pytest.ini` |

---

## Known Limitations

- **Citation inconsistency on metadata-shaped questions** ("what year was X
  introduced?") — publication year lives in a paper's metadata, not its body
  text, so dense/hybrid retrieval can't reliably surface it on content
  similarity alone. Logged in `KNOWN_ISSUES.md`; the real fix is either
  injecting metadata into embedded text or adding a separate metadata-lookup
  path for date-shaped questions — not yet implemented
- **Single-paper dominance** — pure similarity-based top-k retrieval can return
  all 5 results from one dominant survey paper on broad questions; correct
  behavior for dense retrieval, but a real diversity limitation (addressed
  partially by the cross-encoder, not eliminated)
- **Streaming responses lack live token-usage stats** — OpenAI's streaming API
  doesn't return prompt/completion token counts inline the way the
  non-streaming call does; fixable via `stream_options={"include_usage": True}`, not yet wired up
- Reference-section stripping (last-40%-of-document heuristic) occasionally
  misses fragments in papers with unusual section ordering
- The agent's `classify_query()`/`decompose_query()` in the UI are a
  transparency layer, not the actual control mechanism — the LangGraph agent's
  internal reasoning drives real behavior, so the two can occasionally disagree
  on how a query "should" be handled vs how it actually gets handled
- Cold start on Hugging Face Spaces' free tier adds a few seconds of latency on
  the first query after idle

---

## How It Compares

| | BM25 only | Dense only | Hybrid | Agent (LangGraph) |
|---|---|---|---|---|
| Faithfulness | 0.889 | 0.933 | **0.954** | inherits hybrid retrieval |
| Context recall | 0.764 | **0.964** | 0.920 | inherits hybrid retrieval |
| Context precision | 0.794 | 0.901 | **0.933** | inherits hybrid retrieval |
| Handles acronym-heavy queries | ✅ | partially | ✅ | ✅ |
| Distinguishes close domains (e.g. "privacy" vs differential privacy) | ❌ (confused DP with membership inference) | ✅ | ✅ | ✅ |
| Decomposes multi-paper comparisons | ❌ | ❌ | ❌ | ✅ (self-directed, up to 8 tool-call turns) |
| Avg latency (25-Q comparison) | 7.81s | 7.32s | 7.55s | 12.13s |
| Full observability (tracing) | ❌ | ❌ | ❌ | ✅ (LangSmith) |

---

## Deploy

### Live demo (free) — Streamlit Community Cloud

**Live:** [https://rag-assistent-researcher.streamlit.app](https://rag-assistent-researcher.streamlit.app)


### Resume bullet (real RAGAS numbers)

Built a domain-agnostic RAG research assistant over 251 arXiv papers (federated learning, privacy ML, deepfakes) plus a software-docs generalization corpus; hybrid retrieval (BM25 + BGE + FlashRank, RRF k=60) reached 0.954 faithfulness on a 25-question golden eval (vs 0.933 dense-only); LangGraph ReAct agent handles multi-hop questions; FastAPI SSE + Streamlit.

---

## Version History

- `v0.2-dense-rag-baseline` — Phase 2 complete: dense RAG, RAGAS baseline (0.93 faithfulness), Streamlit + FastAPI SSE
- `v0.3-ablation-complete` — Phase 3 complete: BM25, hybrid retrieval, FlashRank reranking, chunking ablation
- `v0.4-react-agent` — Phase 4 complete: LangGraph ReAct agent, Docker, CI, full 4-mode comparison

---

## What I'd Build Next

- **Fix the citation-consistency bug properly** — inject paper metadata (year, arXiv ID) into embedded text, or add a dedicated metadata-lookup tool the agent can call for date-shaped questions, rather than relying on content similarity alone
- **MMR or diversity-aware reranking** to address single-paper dominance on broad survey-style queries, instead of relying on the cross-encoder alone
- **Scale testing** to 1,000+ papers with a hosted ChromaDB deployment
- **Reduce agent latency** — the ~60% overhead vs standard retrieval is partly the UI transparency layer's duplicate classification call; the agent's own internal tool-call count could drive the UI instead, cutting a full LLM round-trip
- **`stream_options={"include_usage": True}`** to restore live token-usage stats during streaming
- **A third corpus** to further stress-test the domain-agnosticism claim beyond software documentation

---

## Data Sources

- arXiv API — federated learning, privacy-preserving ML, and deepfake detection papers (251 PDFs, 10,715 chunks)
- Original software/API teaching notes (FastAPI, Flask, HTTP/OAuth RFC summaries) — secondary generalization corpus, separate Chroma collection, 5 golden questions. Not a dump of vendor documentation.
