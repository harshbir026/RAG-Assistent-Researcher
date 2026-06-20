import streamlit as st
import time
import html

from src.retrieve import retrieve_dense
from src.generate import generate_answer

# ── page config ───────────────────────────────────────────
st.set_page_config(
    page_title="RAG Research Assistant",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

/* ── RESET & BASE ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background: #070B14 !important;
    color: #E2E8F0 !important;
    font-family: 'Inter', sans-serif !important;
}

[data-testid="stAppViewContainer"] {
    background: radial-gradient(ellipse 80% 50% at 50% -10%, rgba(99,102,241,0.18) 0%, transparent 70%),
                #070B14 !important;
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: rgba(10,14,26,0.95) !important;
    border-right: 1px solid rgba(99,102,241,0.2) !important;
    backdrop-filter: blur(20px) !important;
}

[data-testid="stSidebar"] * { color: #CBD5E1 !important; }

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #F8FAFC !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
}

/* sidebar badge pills */
.sidebar-pill {
    display: inline-block;
    background: rgba(99,102,241,0.15);
    border: 1px solid rgba(99,102,241,0.3);
    color: #A5B4FC !important;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.72rem;
    font-family: 'JetBrains Mono', monospace;
    margin: 3px 2px;
    white-space: nowrap;
}

.sidebar-stat {
    background: rgba(15,20,40,0.8);
    border: 1px solid rgba(99,102,241,0.15);
    border-radius: 10px;
    padding: 12px 14px;
    margin: 8px 0;
}
.sidebar-stat .label {
    font-size: 0.65rem;
    color: #64748B !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-family: 'JetBrains Mono', monospace;
}
.sidebar-stat .value {
    font-size: 1.4rem;
    font-weight: 700;
    color: #6366F1 !important;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1.2;
}

/* ── HERO HEADER ── */
.hero-container {
    position: relative;
    padding: 48px 0 36px;
    text-align: center;
    overflow: hidden;
}

.hero-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: #6366F1;
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 10px;
}
.hero-eyebrow::before, .hero-eyebrow::after {
    content: '';
    display: inline-block;
    width: 40px;
    height: 1px;
    background: linear-gradient(90deg, transparent, #6366F1);
}
.hero-eyebrow::after {
    background: linear-gradient(90deg, #6366F1, transparent);
}

.hero-title {
    font-family: 'Inter', sans-serif;
    font-size: clamp(2rem, 5vw, 3.2rem);
    font-weight: 700;
    line-height: 1.1;
    letter-spacing: -0.03em;
    color: #F8FAFC;
    margin: 0 0 12px;
}
.hero-title .accent {
    background: linear-gradient(135deg, #6366F1 0%, #22D3EE 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.hero-sub {
    font-size: 0.95rem;
    color: #64748B;
    max-width: 560px;
    margin: 0 auto 32px;
    line-height: 1.6;
    font-weight: 400;
}

/* floating orbs behind hero */
.orb {
    position: absolute;
    border-radius: 50%;
    filter: blur(80px);
    pointer-events: none;
    z-index: 0;
}
.orb-1 {
    width: 300px; height: 300px;
    background: rgba(99,102,241,0.12);
    top: -60px; left: -80px;
}
.orb-2 {
    width: 200px; height: 200px;
    background: rgba(34,211,238,0.08);
    top: 20px; right: -60px;
}

/* ── SEARCH CARD ── */
.search-card {
    background: rgba(15,20,40,0.7);
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 20px;
    padding: 28px 32px;
    backdrop-filter: blur(20px);
    box-shadow: 0 0 0 1px rgba(99,102,241,0.05), 0 20px 60px rgba(0,0,0,0.4);
    margin-bottom: 28px;
    position: relative;
    z-index: 1;
}

/* ── INPUT OVERRIDE ── */
[data-testid="stTextInput"] input {
    background: rgba(7,11,20,0.8) !important;
    border: 1px solid rgba(99,102,241,0.3) !important;
    border-radius: 12px !important;
    color: #F8FAFC !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.97rem !important;
    padding: 14px 18px !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: #6366F1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.15) !important;
    outline: none !important;
}
[data-testid="stTextInput"] input::placeholder { color: #475569 !important; }

[data-testid="stTextInput"] label {
    color: #94A3B8 !important;
    font-size: 0.78rem !important;
    font-family: 'JetBrains Mono', monospace !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    margin-bottom: 6px !important;
}

/* ── PRIMARY BUTTON ── */
[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, #6366F1 0%, #4F46E5 100%) !important;
    border: none !important;
    border-radius: 12px !important;
    color: #fff !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 12px 32px !important;
    letter-spacing: 0.02em !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 20px rgba(99,102,241,0.35) !important;
}
[data-testid="stButton"] > button[kind="primary"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 30px rgba(99,102,241,0.5) !important;
}

/* secondary button */
[data-testid="stButton"] > button:not([kind="primary"]) {
    background: rgba(99,102,241,0.1) !important;
    border: 1px solid rgba(99,102,241,0.25) !important;
    border-radius: 10px !important;
    color: #A5B4FC !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    transition: all 0.2s !important;
}
[data-testid="stButton"] > button:not([kind="primary"]):hover {
    background: rgba(99,102,241,0.2) !important;
    border-color: rgba(99,102,241,0.4) !important;
}

/* ── ANSWER CARD ── */
.answer-card {
    background: rgba(12,17,35,0.85);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 18px;
    padding: 28px 32px;
    margin: 24px 0;
    backdrop-filter: blur(16px);
    position: relative;
    overflow: hidden;
}
.answer-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #6366F1, #22D3EE, #6366F1);
}
.answer-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #6366F1;
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.answer-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: rgba(99,102,241,0.2);
}
.answer-text {
    font-size: 0.97rem;
    line-height: 1.75;
    color: #CBD5E1;
    font-weight: 400;
}

/* ── METRIC STRIP ── */
.metric-strip {
    display: flex;
    gap: 16px;
    margin: 20px 0;
}
.metric-chip {
    background: rgba(10,14,26,0.8);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 12px;
    padding: 14px 20px;
    flex: 1;
    text-align: center;
    transition: border-color 0.2s;
}
.metric-chip:hover { border-color: rgba(99,102,241,0.4); }
.metric-chip .m-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.5rem;
    font-weight: 700;
    color: #6366F1;
    line-height: 1.1;
}
.metric-chip .m-label {
    font-size: 0.68rem;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 4px;
    font-family: 'JetBrains Mono', monospace;
}

/* ── SECTION HEADER ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 32px 0 16px;
}
.section-header .sh-icon {
    width: 32px; height: 32px;
    background: rgba(99,102,241,0.15);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 14px;
}
.section-header .sh-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #94A3B8;
}
.section-header::after {
    content: '';
    flex: 1;
    height: 1px;
    background: rgba(99,102,241,0.12);
}

/* ── SOURCE EXPANDER OVERRIDES ── */
[data-testid="stExpander"] {
    background: rgba(10,14,26,0.7) !important;
    border: 1px solid rgba(99,102,241,0.15) !important;
    border-radius: 14px !important;
    margin-bottom: 10px !important;
    overflow: hidden !important;
    transition: border-color 0.2s !important;
}
[data-testid="stExpander"]:hover {
    border-color: rgba(99,102,241,0.3) !important;
}
[data-testid="stExpander"] summary {
    background: transparent !important;
    padding: 14px 18px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.88rem !important;
    color: #CBD5E1 !important;
    font-weight: 500 !important;
}
[data-testid="stExpander"] summary:hover { color: #F8FAFC !important; }

[data-testid="stExpanderDetails"] {
    background: rgba(7,11,20,0.5) !important;
    padding: 16px 18px !important;
}

/* ── SIMILARITY BADGE ── */
.sim-badge {
    display: inline-block;
    background: linear-gradient(135deg, rgba(99,102,241,0.2), rgba(34,211,238,0.15));
    border: 1px solid rgba(99,102,241,0.3);
    color: #A5B4FC;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    padding: 2px 9px;
    border-radius: 20px;
    font-weight: 500;
}

/* ── META ROW ── */
.meta-row {
    display: flex;
    gap: 10px;
    align-items: center;
    flex-wrap: wrap;
    margin-bottom: 14px;
}
.meta-tag {
    background: rgba(15,20,40,0.9);
    border: 1px solid rgba(99,102,241,0.15);
    border-radius: 6px;
    padding: 3px 10px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: #64748B;
}
.meta-tag span { color: #94A3B8; }

/* ── CHUNK TEXT BOX ── */
.chunk-text {
    background: rgba(7,11,20,0.8);
    border-left: 2px solid rgba(99,102,241,0.4);
    border-radius: 0 8px 8px 0;
    padding: 14px 16px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    line-height: 1.7;
    color: #64748B;
    margin-top: 12px;
    white-space: pre-wrap;
    word-break: break-word;
}

/* ── HISTORY ITEM ── */
.history-query {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    color: #6366F1;
    margin-bottom: 4px;
}
.history-answer {
    font-size: 0.88rem;
    color: #94A3B8;
    line-height: 1.6;
}
.history-meta {
    display: flex;
    gap: 14px;
    margin-top: 10px;
}
.history-chip {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: #475569;
    background: rgba(99,102,241,0.08);
    border: 1px solid rgba(99,102,241,0.12);
    padding: 2px 8px;
    border-radius: 6px;
}

/* ── SLIDER ── */
[data-testid="stSlider"] > div > div {
    background: rgba(99,102,241,0.3) !important;
}
[data-testid="stSlider"] [data-testid="stTickBar"] { color: #475569 !important; }

/* ── WARNING ── */
[data-testid="stAlert"] {
    background: rgba(245,158,11,0.1) !important;
    border: 1px solid rgba(245,158,11,0.3) !important;
    border-radius: 12px !important;
    color: #FCD34D !important;
}

/* ── SPINNER ── */
[data-testid="stSpinner"] { color: #6366F1 !important; }

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.3); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(99,102,241,0.5); }

/* ── HIDE STREAMLIT CHROME ── */
#MainMenu, footer, [data-testid="stDecoration"],
[data-testid="stStatusWidget"] { display: none !important; }

header[data-testid="stHeader"] {
    background: transparent !important;
    border-bottom: none !important;
}

/* ── LINK BUTTON ── */
[data-testid="stLinkButton"] a {
    background: rgba(34,211,238,0.1) !important;
    border: 1px solid rgba(34,211,238,0.3) !important;
    border-radius: 8px !important;
    color: #22D3EE !important;
    font-size: 0.8rem !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    padding: 8px 14px !important;
    transition: all 0.2s !important;
    text-decoration: none !important;
}
[data-testid="stLinkButton"] a:hover {
    background: rgba(34,211,238,0.18) !important;
    border-color: rgba(34,211,238,0.5) !important;
    transform: translateY(-1px) !important;
}

/* ── COPY TEXT AREA ── */
[data-testid="stTextArea"] textarea {
    background: rgba(7,11,20,0.8) !important;
    border: 1px solid rgba(99,102,241,0.2) !important;
    border-radius: 10px !important;
    color: #94A3B8 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important;
}

/* ── COLUMNS ── */
[data-testid="stColumns"] { gap: 16px !important; }

/* ── DIVIDER ── */
hr {
    border: none !important;
    border-top: 1px solid rgba(99,102,241,0.12) !important;
    margin: 32px 0 !important;
}
</style>
""", unsafe_allow_html=True)


# ── session state init ───────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []


# ── SIDEBAR ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("### System")

    st.markdown("""
    <div class='sidebar-stat'>
        <div class='label'>Indexed Papers</div>
        <div class='value'>251</div>
    </div>
    <div class='sidebar-stat'>
        <div class='label'>Embedding Model</div>
        <div class='value' style='font-size:0.9rem;color:#22D3EE !important;'>BGE-768</div>
    </div>
    <div class='sidebar-stat'>
        <div class='label'>Vector Store</div>
        <div class='value' style='font-size:0.9rem;color:#A5B4FC !important;'>ChromaDB</div>
    </div>
    <div class='sidebar-stat'>
        <div class='label'>Generator</div>
        <div class='value' style='font-size:0.9rem;color:#34D399 !important;'>GPT-4o-mini</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Domains")
    st.markdown("""
    <div>
        <span class='sidebar-pill'>Federated Learning</span>
        <span class='sidebar-pill'>Privacy ML</span>
        <span class='sidebar-pill'>Deepfake Detection</span>
        <span class='sidebar-pill'>Differential Privacy</span>
        <span class='sidebar-pill'>Secure Aggregation</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Retrieval")
    k = st.slider("Chunks to retrieve (k)", min_value=1, max_value=10, value=5)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("⟳  Clear history", use_container_width=True):
        st.session_state.history = []
        st.rerun()


# ── HERO ──────────────────────────────────────────────────
st.markdown("""
<div class='hero-container'>
    <div class='orb orb-1'></div>
    <div class='orb orb-2'></div>
    <div class='hero-eyebrow'>Dense Retrieval · arXiv · GPT-4o-mini</div>
    <h1 class='hero-title'>RAG <span class='accent'>Research</span> Assistant</h1>
    <p class='hero-sub'>Semantic search over 251 curated arXiv papers. Ask anything about federated learning, privacy-preserving ML, or deepfake detection.</p>
</div>
""", unsafe_allow_html=True)


# ── SEARCH CARD ───────────────────────────────────────────
st.markdown("<div class='search-card'>", unsafe_allow_html=True)

query = st.text_input(
    "query",
    placeholder="e.g.  How does FedAvg handle non-IID data distributions?",
    label_visibility="collapsed",
)

col_btn, col_hint = st.columns([1, 4])
with col_btn:
    ask_clicked = st.button("⚡  Search", type="primary", use_container_width=True)
with col_hint:
    st.markdown(
        "<p style='color:#334155;font-size:0.78rem;padding-top:10px;font-family:JetBrains Mono,monospace;'>"
        "Try: differential privacy noise mechanisms · FedProx convergence · deepfake GAN detection</p>",
        unsafe_allow_html=True,
    )

st.markdown("</div>", unsafe_allow_html=True)


# ── QUERY EXECUTION ───────────────────────────────────────
if ask_clicked and query.strip():
    with st.spinner("Embedding query · Searching 251 papers · Generating answer…"):
        start_time = time.time()
        chunks = retrieve_dense(query, k=k)
        result = generate_answer(query, chunks)
        elapsed = time.time() - start_time

    st.session_state.history.insert(0, {
        "query": query,
        "result": result,
        "chunks": chunks,
        "elapsed": elapsed,
    })

elif ask_clicked and not query.strip():
    st.warning("Enter a question to search the research corpus.")


# ── LATEST RESULT ─────────────────────────────────────────
if st.session_state.history:
    latest = st.session_state.history[0]

    # ── METRICS ──
    st.markdown(f"""
    <div class='metric-strip'>
        <div class='metric-chip'>
            <div class='m-val'>{latest['result']['num_chunks_used']}</div>
            <div class='m-label'>Chunks Used</div>
        </div>
        <div class='metric-chip'>
            <div class='m-val'>{latest['result']['usage']['total_tokens']}</div>
            <div class='m-label'>Tokens</div>
        </div>
        <div class='metric-chip'>
            <div class='m-val'>{latest['elapsed']:.1f}s</div>
            <div class='m-label'>Latency</div>
        </div>
        <div class='metric-chip'>
            <div class='m-val'>{len(latest['chunks'])}</div>
            <div class='m-label'>Sources</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── ANSWER ──
    st.markdown(f"""
    <div class='answer-card'>
        <div class='answer-label'>⚡ Generated Answer</div>
        <div class='answer-text'>{html.escape(latest['result']['answer'])}</div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("📋  Copy answer text"):
        st.text_area(
            "answer_copy",
            value=latest["result"]["answer"],
            height=140,
            label_visibility="collapsed",
        )

    # ── SOURCES ──
    st.markdown("""
    <div class='section-header'>
        <div class='sh-icon'>📄</div>
        <div class='sh-title'>Retrieved Sources</div>
    </div>
    """, unsafe_allow_html=True)

    for i, chunk in enumerate(latest["chunks"], 1):
        sim_pct = int(chunk.similarity * 100)
        with st.expander(
            f"[{i}]  {chunk.title[:65]}{'…' if len(chunk.title) > 65 else ''}"
        ):
            st.markdown(f"""
            <div class='meta-row'>
                <span class='sim-badge'>sim {chunk.similarity:.3f}</span>
                <span class='meta-tag'>arXiv: <span>{chunk.arxiv_id}</span></span>
                <span class='meta-tag'>year: <span>{chunk.year}</span></span>
                <span class='meta-tag'>chunk: <span>{chunk.chunk_index}/{chunk.total_chunks}</span></span>
            </div>
            """, unsafe_allow_html=True)

            col_a, col_b = st.columns([4, 1])
            with col_b:
                st.link_button(
                    "↗  arXiv",
                    f"https://arxiv.org/abs/{chunk.arxiv_id}",
                    use_container_width=True,
                )

            preview = chunk.text[:500] + ("…" if len(chunk.text) > 500 else "")
            st.markdown(
                f"<div class='chunk-text'>{html.escape(preview)}</div>",
                unsafe_allow_html=True,
            )


# ── HISTORY ───────────────────────────────────────────────
if len(st.session_state.history) > 1:
    st.markdown("""
    <div class='section-header'>
        <div class='sh-icon'>🕐</div>
        <div class='sh-title'>Query History</div>
    </div>
    """, unsafe_allow_html=True)

    for item in st.session_state.history[1:]:
        with st.expander(f"{item['query']}"):
            st.markdown(
                f"<div class='history-query'>// {html.escape(item['query'])}</div>"
                f"<div class='history-answer'>{html.escape(item['result']['answer'])}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(f"""
            <div class='history-meta'>
                <span class='history-chip'>{item['result']['num_chunks_used']} chunks</span>
                <span class='history-chip'>{item['result']['usage']['total_tokens']} tokens</span>
                <span class='history-chip'>{item['elapsed']:.1f}s</span>
            </div>
            """, unsafe_allow_html=True)