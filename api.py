import json
import time
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, Field
from src.retrieve import retrieve_dense
from src.generate import generate_answer_stream, build_context_block


from src.agent import research_agent, system_prompt, classify_query, decompose_query
from langchain_core.messages import HumanMessage, SystemMessage


# ── OpenAPI metadata ──────────────────────────────────────────────────────────
app = FastAPI(
    title="RAG Research Assistant",
    description="""
## Streaming RAG over 251 arXiv papers

A **Retrieval-Augmented Generation** pipeline combining dense semantic search
with streaming LLM generation — built for real-time research Q&A over academic literature.

### Pipeline

| Stage | Component | Detail |
|---|---|---|
| Embedding | BGE (BAAI/bge-base-en-v1.5) | 768-dim dense vectors |
| Vector Store | ChromaDB | Cosine similarity search |
| Generator | GPT-4o-mini | Streaming via OpenAI SSE |

### Domains covered

- **Federated Learning** — FedAvg, FedProx, non-IID convergence, communication efficiency
- **Privacy-Preserving ML** — Differential privacy, secure aggregation, homomorphic encryption
- **Deepfake Detection** — GAN-based synthesis, face-swap forensics, multimodal detection

### Streaming protocol

Answers stream as **Server-Sent Events (SSE)**. Three event types are emitted in order:

```
data: {"type": "retrieval", "sources": [...], "num_chunks": 5}

data: {"type": "token", "content": "Federated"}
data: {"type": "token", "content": " learning"}
...

data: {"type": "done", "full_answer": "...", "elapsed_seconds": 1.83}
```

Consume with `EventSource` in the browser or `httpx`/`aiohttp` in Python.
""",
    version="1.0.0",
    contact={
        "name": "RAG Research Assistant",
        "url": "https://github.com",
    },
    license_info={
        "name": "MIT",
    },
    openapi_tags=[
        {
            "name": "core",
            "description": "Primary RAG query endpoint with streaming response.",
        },
        {
            "name": "ops",
            "description": "Health and status endpoints for monitoring and deployment checks.",
        },
    ],
    docs_url=None,   # we serve our own themed /docs
    redoc_url=None,
)


# ── Request / Response schemas ────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        description="Natural-language question about federated learning, privacy ML, or deepfake detection.",
        examples=["How does FedAvg handle non-IID data distributions?"],
        min_length=3,
        max_length=1000,
    )
    k: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Number of document chunks to retrieve from the vector store. Higher values provide more context but increase token usage.",
        examples=[5],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "summary": "Federated learning convergence",
                    "value": {
                        "question": "How does FedAvg handle non-IID data distributions?",
                        "k": 5,
                    },
                },
                {
                    "summary": "Differential privacy",
                    "value": {
                        "question": "What is the privacy-utility tradeoff when epsilon < 1 in DP-SGD?",
                        "k": 7,
                    },
                },
                {
                    "summary": "Deepfake detection",
                    "value": {
                        "question": "Which neural architectures are most effective for detecting GAN-generated deepfakes?",
                        "k": 5,
                    },
                },
            ]
        }
    }


class RootResponse(BaseModel):
    status: str = Field(description="Always 'ok' when the service is running.")
    message: str = Field(description="Human-readable summary of available endpoints.")


class HealthResponse(BaseModel):
    status: str = Field(description="Always 'healthy' when the process is alive.")
    timestamp: float = Field(description="Unix epoch timestamp of the health check.")


class SourceItem(BaseModel):
    arxiv_id: str = Field(description="arXiv paper identifier, e.g. '2301.12345'.")
    title: str = Field(description="Full title of the paper.")
    year: str = Field(description="Publication year.")
    similarity: float = Field(description="Cosine similarity score (0–1) to the query embedding.")


class RetrievalEvent(BaseModel):
    type: str = Field(default="retrieval", description="Event discriminator — always 'retrieval'.")
    sources: list[SourceItem] = Field(description="Ordered list of retrieved paper chunks.")
    num_chunks: int = Field(description="Number of chunks retrieved (equals k unless fewer exist).")


class TokenEvent(BaseModel):
    type: str = Field(default="token", description="Event discriminator — always 'token'.")
    content: str = Field(description="One streamed piece of the generated answer.")


class DoneEvent(BaseModel):
    type: str = Field(default="done", description="Event discriminator — always 'done'.")
    full_answer: str = Field(description="Complete concatenated answer text.")
    elapsed_seconds: float = Field(description="Wall-clock time from query receipt to stream completion.")


# ── Custom Swagger UI ─────────────────────────────────────────────────────────

SWAGGER_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

/* ── BASE ── */
body, html {
    background: #070B14 !important;
    font-family: 'Inter', sans-serif !important;
    color: #CBD5E1 !important;
}

/* top bar */
.swagger-ui .topbar {
    background: linear-gradient(135deg, #0D1220 0%, #070B14 100%) !important;
    border-bottom: 1px solid rgba(99,102,241,0.25) !important;
    padding: 10px 0 !important;
}

.swagger-ui .topbar .download-url-wrapper { display: none !important; }

.swagger-ui .topbar-wrapper .link {
    display: flex !important;
    align-items: center !important;
    gap: 10px !important;
}

.swagger-ui .topbar-wrapper .link::before {
    content: '⚡ RAG Research Assistant';
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.06em !important;
    color: #A5B4FC !important;
}

.swagger-ui .topbar-wrapper img { display: none !important; }

/* ── INFO BLOCK ── */
.swagger-ui .info {
    background: rgba(10,14,26,0.8) !important;
    border: 1px solid rgba(99,102,241,0.2) !important;
    border-radius: 16px !important;
    padding: 32px 36px !important;
    margin: 24px 0 !important;
    position: relative !important;
    overflow: hidden !important;
}

.swagger-ui .info::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #6366F1, #22D3EE, #6366F1) !important;
}

.swagger-ui .info .title {
    font-family: 'Inter', sans-serif !important;
    font-size: 1.9rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.03em !important;
    color: #F8FAFC !important;
    margin-bottom: 4px !important;
}

.swagger-ui .info .title small {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.7rem !important;
    background: rgba(99,102,241,0.15) !important;
    border: 1px solid rgba(99,102,241,0.3) !important;
    color: #A5B4FC !important;
    padding: 2px 10px !important;
    border-radius: 20px !important;
    vertical-align: middle !important;
    margin-left: 10px !important;
    font-weight: 500 !important;
    letter-spacing: 0.05em !important;
}

.swagger-ui .info .description p,
.swagger-ui .info .description li {
    font-size: 0.9rem !important;
    color: #94A3B8 !important;
    line-height: 1.7 !important;
}

.swagger-ui .info .description h2,
.swagger-ui .info .description h3 {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
    color: #6366F1 !important;
    margin: 24px 0 10px !important;
    border: none !important;
}

/* ── MARKDOWN TABLE in description ── */
.swagger-ui .info .description table {
    width: 100% !important;
    border-collapse: collapse !important;
    margin: 16px 0 !important;
    font-size: 0.83rem !important;
}

.swagger-ui .info .description table th {
    background: rgba(99,102,241,0.1) !important;
    color: #A5B4FC !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.68rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    padding: 8px 12px !important;
    border: 1px solid rgba(99,102,241,0.15) !important;
    text-align: left !important;
}

.swagger-ui .info .description table td {
    padding: 8px 12px !important;
    border: 1px solid rgba(99,102,241,0.1) !important;
    color: #CBD5E1 !important;
}

.swagger-ui .info .description table tr:hover td {
    background: rgba(99,102,241,0.05) !important;
}

/* ── CODE BLOCKS in description ── */
.swagger-ui .info .description pre,
.swagger-ui .info .description code {
    background: rgba(7,11,20,0.9) !important;
    border: 1px solid rgba(99,102,241,0.2) !important;
    border-radius: 8px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important;
    color: #22D3EE !important;
    padding: 2px 6px !important;
}

.swagger-ui .info .description pre {
    padding: 16px 18px !important;
    line-height: 1.65 !important;
    overflow-x: auto !important;
}

/* ── OPERATION TAGS (section headers) ── */
.swagger-ui .opblock-tag {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.2em !important;
    text-transform: uppercase !important;
    color: #6366F1 !important;
    border-bottom: 1px solid rgba(99,102,241,0.15) !important;
    padding: 12px 0 10px !important;
    margin-top: 32px !important;
    background: transparent !important;
}

.swagger-ui .opblock-tag small {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.8rem !important;
    color: #475569 !important;
    font-weight: 400 !important;
    letter-spacing: 0 !important;
    text-transform: none !important;
}

/* ── OPERATION BLOCKS ── */
.swagger-ui .opblock {
    background: rgba(10,14,26,0.7) !important;
    border: 1px solid rgba(99,102,241,0.15) !important;
    border-radius: 14px !important;
    margin: 10px 0 !important;
    overflow: hidden !important;
    transition: border-color 0.2s !important;
    box-shadow: none !important;
}

.swagger-ui .opblock:hover {
    border-color: rgba(99,102,241,0.35) !important;
}

/* POST */
.swagger-ui .opblock.opblock-post {
    border-color: rgba(99,102,241,0.3) !important;
}
.swagger-ui .opblock.opblock-post .opblock-summary-method {
    background: linear-gradient(135deg, #6366F1, #4F46E5) !important;
    color: #fff !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 700 !important;
    border-radius: 8px !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.05em !important;
    padding: 6px 12px !important;
    box-shadow: 0 2px 10px rgba(99,102,241,0.35) !important;
}

/* GET */
.swagger-ui .opblock.opblock-get .opblock-summary-method {
    background: linear-gradient(135deg, rgba(34,211,238,0.2), rgba(34,211,238,0.1)) !important;
    color: #22D3EE !important;
    border: 1px solid rgba(34,211,238,0.3) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 700 !important;
    border-radius: 8px !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.05em !important;
    padding: 6px 12px !important;
}

.swagger-ui .opblock-summary {
    background: transparent !important;
    padding: 14px 18px !important;
    align-items: center !important;
}

.swagger-ui .opblock-summary-path {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    color: #E2E8F0 !important;
}

.swagger-ui .opblock-summary-description {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.82rem !important;
    color: #64748B !important;
}

/* ── EXPANDED OPERATION BODY ── */
.swagger-ui .opblock-body {
    background: rgba(7,11,20,0.5) !important;
    border-top: 1px solid rgba(99,102,241,0.1) !important;
}

.swagger-ui .opblock-section-header {
    background: rgba(10,14,26,0.6) !important;
    border-bottom: 1px solid rgba(99,102,241,0.1) !important;
    padding: 10px 18px !important;
}

.swagger-ui .opblock-section-header h4 {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.62rem !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
    color: #6366F1 !important;
    font-weight: 600 !important;
}

/* ── PARAMETERS TABLE ── */
.swagger-ui table thead tr th,
.swagger-ui table thead tr td {
    background: rgba(99,102,241,0.08) !important;
    color: #64748B !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    border-bottom: 1px solid rgba(99,102,241,0.15) !important;
    padding: 8px 12px !important;
}

.swagger-ui table tbody tr td {
    background: transparent !important;
    border-bottom: 1px solid rgba(99,102,241,0.07) !important;
    padding: 10px 12px !important;
    color: #94A3B8 !important;
    font-size: 0.84rem !important;
}

.swagger-ui .parameter__name {
    font-family: 'JetBrains Mono', monospace !important;
    color: #A5B4FC !important;
    font-weight: 600 !important;
    font-size: 0.83rem !important;
}

.swagger-ui .parameter__type {
    font-family: 'JetBrains Mono', monospace !important;
    color: #22D3EE !important;
    font-size: 0.75rem !important;
}

.swagger-ui .parameter__deprecated {
    color: #F87171 !important;
    font-size: 0.7rem !important;
}

/* ── SCHEMA / MODEL ── */
.swagger-ui .model-box {
    background: rgba(7,11,20,0.8) !important;
    border: 1px solid rgba(99,102,241,0.15) !important;
    border-radius: 10px !important;
    padding: 14px !important;
}

.swagger-ui .model {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.8rem !important;
    color: #94A3B8 !important;
}

.swagger-ui .model .property.primitive { color: #22D3EE !important; }
.swagger-ui .model span.model-title { color: #A5B4FC !important; font-weight: 600 !important; }
.swagger-ui .model-toggle:after { border-color: #6366F1 !important; }

/* ── JSON / CODE AREAS ── */
.swagger-ui .highlight-code pre,
.swagger-ui .microlight,
.swagger-ui pre.microlight {
    background: rgba(7,11,20,0.9) !important;
    border: 1px solid rgba(99,102,241,0.15) !important;
    border-radius: 10px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important;
    color: #CBD5E1 !important;
    line-height: 1.65 !important;
    padding: 14px 16px !important;
}

/* ── EXAMPLES DROPDOWN ── */
.swagger-ui .examples-select select,
.swagger-ui select {
    background: rgba(10,14,26,0.9) !important;
    border: 1px solid rgba(99,102,241,0.25) !important;
    border-radius: 8px !important;
    color: #CBD5E1 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.8rem !important;
    padding: 6px 10px !important;
}

.swagger-ui .examples-select select:focus,
.swagger-ui select:focus {
    border-color: #6366F1 !important;
    outline: none !important;
    box-shadow: 0 0 0 2px rgba(99,102,241,0.15) !important;
}

/* ── TEXTAREA (try-it-out) ── */
.swagger-ui textarea {
    background: rgba(7,11,20,0.9) !important;
    border: 1px solid rgba(99,102,241,0.2) !important;
    border-radius: 10px !important;
    color: #CBD5E1 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important;
    padding: 12px 14px !important;
    line-height: 1.6 !important;
}

.swagger-ui textarea:focus {
    border-color: #6366F1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.12) !important;
    outline: none !important;
}

/* ── BUTTONS ── */
.swagger-ui .btn {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    border-radius: 8px !important;
    padding: 8px 18px !important;
    transition: all 0.2s !important;
    letter-spacing: 0.02em !important;
}

.swagger-ui .btn.execute {
    background: linear-gradient(135deg, #6366F1, #4F46E5) !important;
    border: none !important;
    color: #fff !important;
    box-shadow: 0 3px 14px rgba(99,102,241,0.4) !important;
}

.swagger-ui .btn.execute:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(99,102,241,0.5) !important;
}

.swagger-ui .btn.try-out__btn {
    background: rgba(99,102,241,0.1) !important;
    border: 1px solid rgba(99,102,241,0.3) !important;
    color: #A5B4FC !important;
}

.swagger-ui .btn.try-out__btn:hover {
    background: rgba(99,102,241,0.2) !important;
}

.swagger-ui .btn.cancel {
    background: rgba(248,113,113,0.1) !important;
    border: 1px solid rgba(248,113,113,0.3) !important;
    color: #F87171 !important;
}

/* ── RESPONSES TABLE ── */
.swagger-ui .responses-inner h4,
.swagger-ui .responses-inner h5 {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.62rem !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    color: #475569 !important;
    font-weight: 600 !important;
}

.swagger-ui .response-col_status {
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    color: #34D399 !important;
}

.swagger-ui .response-col_description {
    font-size: 0.84rem !important;
    color: #94A3B8 !important;
}

/* ── CURL BLOCK ── */
.swagger-ui .curl-command {
    background: rgba(7,11,20,0.9) !important;
    border: 1px solid rgba(99,102,241,0.15) !important;
    border-radius: 10px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important;
    color: #22D3EE !important;
    padding: 14px 16px !important;
}

/* ── LOADING ── */
.swagger-ui .loading-container .loading::before {
    border-color: rgba(99,102,241,0.2) !important;
    border-top-color: #6366F1 !important;
}

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: rgba(7,11,20,0.5); }
::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.3); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(99,102,241,0.5); }

/* ── MISC ── */
.swagger-ui .wrapper { max-width: 1100px !important; padding: 0 24px !important; }

.swagger-ui a { color: #A5B4FC !important; }
.swagger-ui a:hover { color: #22D3EE !important; }

.swagger-ui .info a { color: #22D3EE !important; }

/* hide the ugly default authorize lock for now */
.swagger-ui .auth-wrapper { display: none !important; }

/* example value label */
.swagger-ui .example__section-header {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.65rem !important;
    color: #475569 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
}
"""

SWAGGER_JS_EXTRA = """
// Patch the page title
document.title = "RAG Research Assistant · API Docs";

// Inject a subtle animated gradient behind the topbar
const topbar = document.querySelector('.swagger-ui .topbar');
if (topbar) {
    topbar.style.background = 'linear-gradient(135deg, #0D1220 0%, #070B14 100%)';
}
"""


@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui():
    html = get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="RAG Research Assistant · API Docs",
        swagger_favicon_url="https://em-content.zobj.net/source/twitter/376/high-voltage_26a1.png",
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js",
    )
    # inject our custom CSS and JS into the returned HTML
    custom_head = f"""
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css">
    <style>{SWAGGER_CSS}</style>
    """
    custom_body_end = f"<script>{SWAGGER_JS_EXTRA}</script>"

    raw = html.body.decode("utf-8")
    raw = raw.replace("</head>", custom_head + "</head>", 1)
    raw = raw.replace("</body>", custom_body_end + "</body>", 1)
    return HTMLResponse(raw)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get(
    "/",
    response_model=RootResponse,
    summary="Service root",
    description="Returns a status confirmation and a pointer to the main `/query` endpoint.",
    tags=["ops"],
)
def root():
    return {
        "status": "ok",
        "message": "RAG Research Assistant API — POST /query for streaming answers",
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description=(
        "Lightweight liveness probe. Returns `healthy` and the current Unix timestamp. "
        "Use this for container orchestration readiness/liveness checks (e.g. Kubernetes, Railway, Render)."
    ),
    tags=["ops"],
)
def health():
    return {"status": "healthy", "timestamp": time.time()}


# ── SSE helpers ───────────────────────────────────────────────────────────────

def sse_event(data: dict) -> str:
    """Format a dict as a Server-Sent Events line."""
    return f"data: {json.dumps(data)}\n\n"


async def stream_rag_response(question: str, k: int):
    """
    Generator that yields SSE-formatted events:
    1. A 'retrieval' event with the sources found
    2. Multiple 'token' events as the answer streams in
    3. A final 'done' event with usage stats
    """
    start_time = time.time()

    # ── retrieval phase ─────────────────────────────────
    chunks = retrieve_dense(question, k=k)
    sources = [
        {
            "arxiv_id": c.arxiv_id,
            "title": c.title,
            "year": c.year,
            "similarity": c.similarity,
        }
        for c in chunks
    ]
    yield sse_event({"type": "retrieval", "sources": sources, "num_chunks": len(chunks)})

    # ── generation phase (streamed) ─────────────────────
    full_answer = ""
    for token in generate_answer_stream(question, chunks):
        full_answer += token
        yield sse_event({"type": "token", "content": token})

    # ── done ─────────────────────────────────────────────
    elapsed = time.time() - start_time
    yield sse_event({
        "type": "done",
        "full_answer": full_answer,
        "elapsed_seconds": round(elapsed, 2),
    })


# ── Main query endpoint ───────────────────────────────────────────────────────

@app.post(
    "/query",
    summary="Stream a RAG answer",
    description="""
Submit a natural-language question and receive a **streaming Server-Sent Events** response.

**Event sequence:**

1. **`retrieval`** — emitted immediately after vector search completes; contains the ranked source list.
2. **`token`** — emitted repeatedly as the LLM generates text; concatenate `content` fields to reconstruct the answer.
3. **`done`** — emitted once when generation finishes; contains the full answer and wall-clock latency.

**Python client example:**

```python
import httpx

with httpx.stream("POST", "http://localhost:8000/query",
                  json={"question": "How does FedAvg handle non-IID data?", "k": 5}) as r:
    for line in r.iter_lines():
        if line.startswith("data: "):
            event = json.loads(line[6:])
            if event["type"] == "token":
                print(event["content"], end="", flush=True)
```

**Browser `EventSource` example:**

```javascript
// Note: EventSource only supports GET; use fetch() with a ReadableStream for POST
const res = await fetch('/query', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({ question: 'What is differential privacy?', k: 5 })
});
const reader = res.body.getReader();
// ... read chunks and parse SSE lines
```
""",
    tags=["core"],
    responses={
        200: {
            "description": "SSE stream of retrieval → token* → done events.",
            "content": {
                "text/event-stream": {
                    "example": (
                        'data: {"type": "retrieval", "sources": [...], "num_chunks": 5}\n\n'
                        'data: {"type": "token", "content": "Federated"}\n\n'
                        'data: {"type": "done", "full_answer": "...", "elapsed_seconds": 1.83}\n\n'
                    )
                }
            },
        },
        400: {"description": "Question is empty or whitespace-only."},
        422: {"description": "Request body failed schema validation (e.g. `k` out of range 1–10)."},
    },
)
async def query_endpoint(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    return StreamingResponse(
        stream_rag_response(request.question, request.k),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
class AgentQueryRequest(BaseModel):
    question: str


@app.post("/query/agent")
def query_agent(request: AgentQueryRequest):
    start = time.time()

    classification = classify_query(request.question)
    sub_questions = decompose_query(request.question) if classification == "complex" else []

    response = research_agent.invoke({
        "messages": [
            SystemMessage(content=system_prompt),
            HumanMessage(content=request.question),
        ]
    })

    final_answer = response["messages"][-1].content

    sources = [
        str(msg.content)[:300]
        for msg in response["messages"]
        if getattr(msg, "name", "") == "search_research_papers"
    ]

    latency_ms = round((time.time() - start) * 1000, 1)

    return {
        "answer": final_answer,
        "routing_decision": classification,
        "sub_questions": sub_questions,
        "sources": sources,
        "latency_ms": latency_ms,
    }