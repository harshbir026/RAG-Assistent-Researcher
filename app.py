import streamlit as st
import time
import json
import html
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
from src.agent import research_agent, system_prompt, classify_query, decompose_query
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import Client
from langchain_core.tracers.context import collect_runs

from src.retrieve import retrieve_dense, retrieve_bm25, retrieve_hybrid
from src.generate import generate_answer_stream


# ── page config ───────────────────────────────────────────
st.set_page_config(
    page_title="RAG Research Assistant",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── session state ─────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ══════════════════════════════════════════════════════════
# DESIGN SYSTEM CSS
# ══════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stApp"] {
    background: #070B14 !important;
    color: #E2E8F0 !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 90% 40% at 50% -5%, rgba(99,102,241,0.14) 0%, transparent 65%),
        #070B14 !important;
}

/* ── HIDE CHROME ── */
#MainMenu, footer,
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] { display: none !important; }
header[data-testid="stHeader"] {
    background: transparent !important;
    border-bottom: none !important;
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: rgba(8,12,24,0.97) !important;
    border-right: 1px solid rgba(99,102,241,0.18) !important;
}
[data-testid="stSidebar"] * { color: #CBD5E1 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
    color: #475569 !important;
}
.mode-card {
    border-radius: 12px;
    padding: 14px 16px;
    margin: 6px 0;
    border: 1px solid rgba(99,102,241,0.18);
    background: rgba(99,102,241,0.07);
}
.mode-card.agentic {
    border-color: rgba(167,139,250,0.35);
    background: rgba(167,139,250,0.08);
}
.mode-card .mc-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    font-weight: 600;
    color: #A5B4FC;
    margin-bottom: 3px;
}
.mode-card.agentic .mc-title { color: #C4B5FD; }
.mode-card .mc-sub { font-size: 0.72rem; color: #475569; line-height: 1.45; }

.sidebar-stat {
    background: rgba(10,14,26,0.8);
    border: 1px solid rgba(99,102,241,0.12);
    border-radius: 10px;
    padding: 11px 14px;
    margin: 6px 0;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.sidebar-stat .ss-label {
    font-size: 0.68rem; color: #475569;
    font-family: 'JetBrains Mono', monospace;
    text-transform: uppercase; letter-spacing: 0.08em;
}
.sidebar-stat .ss-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem; font-weight: 600; color: #A5B4FC;
}

/* ── TABS ── */
[data-testid="stTabs"] [role="tablist"] {
    background: rgba(10,14,26,0.7) !important;
    border: 1px solid rgba(99,102,241,0.15) !important;
    border-radius: 14px !important;
    padding: 4px !important; gap: 4px !important;
    margin-bottom: 28px !important;
}
[data-testid="stTabs"] [role="tab"] {
    background: transparent !important; border: none !important;
    border-radius: 10px !important; color: #475569 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.87rem !important; font-weight: 500 !important;
    padding: 9px 22px !important; transition: all 0.18s !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    background: rgba(99,102,241,0.18) !important;
    color: #A5B4FC !important; font-weight: 600 !important;
}
[data-testid="stTabs"] [role="tab"]:hover { color: #CBD5E1 !important; }
[data-testid="stTabs"] [data-baseweb="tab-highlight"] { display: none !important; }

/* ── HERO ── */
.hero-wrap { text-align: center; padding: 40px 0 24px; }
.hero-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem; letter-spacing: 0.28em; text-transform: uppercase;
    color: #6366F1; margin-bottom: 14px;
    display: flex; align-items: center; justify-content: center; gap: 12px;
}
.hero-eyebrow::before, .hero-eyebrow::after {
    content: ''; display: inline-block; width: 48px; height: 1px;
    background: linear-gradient(90deg, transparent, #6366F1);
}
.hero-eyebrow::after { background: linear-gradient(90deg, #6366F1, transparent); }
.hero-title {
    font-size: clamp(1.9rem, 4.5vw, 3rem); font-weight: 700;
    letter-spacing: -0.03em; color: #F8FAFC; margin: 0 0 10px; line-height: 1.1;
}
.hero-title .grad {
    background: linear-gradient(135deg, #6366F1 10%, #22D3EE 90%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.hero-sub { font-size: 0.92rem; color: #475569; max-width: 560px; margin: 0 auto; line-height: 1.65; }

/* ── STANDARD PIPELINE VISUALIZER ── */
.pipeline-wrap {
    display: flex; align-items: center; justify-content: center;
    gap: 0; margin: 28px 0 20px;
}
.pipe-stage { display: flex; flex-direction: column; align-items: center; gap: 7px; min-width: 110px; }
.pipe-icon {
    width: 46px; height: 46px; border-radius: 12px;
    display: flex; align-items: center; justify-content: center; font-size: 19px;
    border: 1px solid rgba(99,102,241,0.18); background: rgba(10,14,26,0.8); transition: all 0.3s;
}
.pipe-icon.active { border-color: #6366F1; background: rgba(99,102,241,0.14); box-shadow: 0 0 20px rgba(99,102,241,0.32); }
.pipe-icon.done   { border-color: rgba(52,211,153,0.45); background: rgba(52,211,153,0.09); }
.pipe-label {
    font-family: 'JetBrains Mono', monospace; font-size: 0.6rem;
    letter-spacing: 0.12em; text-transform: uppercase; color: #334155; transition: color 0.3s;
}
.pipe-label.active { color: #A5B4FC; }
.pipe-label.done   { color: #34D399; }
.pipe-connector {
    width: 52px; height: 1px; background: rgba(99,102,241,0.14);
    position: relative; top: -22px; flex-shrink: 0; transition: background 0.3s;
}
.pipe-connector.active { background: linear-gradient(90deg, #34D399, #6366F1); }

/* ── AGENTIC LOOP VISUALIZER ── */
.agent-loop-wrap {
    display: flex; align-items: center; justify-content: center;
    gap: 0; margin: 28px 0 20px;
    padding: 20px 28px;
    background: rgba(167,139,250,0.04);
    border: 1px solid rgba(167,139,250,0.15);
    border-radius: 16px;
    position: relative;
    overflow: hidden;
}
.agent-loop-wrap::before {
    content: 'LangGraph ReAct Loop';
    position: absolute; top: 8px; left: 50%; transform: translateX(-50%);
    font-family: 'JetBrains Mono', monospace; font-size: 0.55rem;
    letter-spacing: 0.18em; text-transform: uppercase; color: rgba(167,139,250,0.5);
}
.agent-node {
    display: flex; flex-direction: column; align-items: center; gap: 7px; min-width: 90px;
}
.agent-icon {
    width: 46px; height: 46px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center; font-size: 18px;
    border: 1px solid rgba(167,139,250,0.2); background: rgba(10,14,26,0.85); transition: all 0.3s;
}
.agent-icon.active {
    border-color: #A78BFA;
    background: rgba(167,139,250,0.14);
    box-shadow: 0 0 22px rgba(167,139,250,0.35);
}
.agent-icon.done { border-color: rgba(52,211,153,0.45); background: rgba(52,211,153,0.09); }
.agent-label {
    font-family: 'JetBrains Mono', monospace; font-size: 0.58rem;
    letter-spacing: 0.1em; text-transform: uppercase; color: #334155; transition: color 0.3s; white-space: nowrap;
}
.agent-label.active { color: #C4B5FD; }
.agent-label.done   { color: #34D399; }
.agent-arrow {
    font-size: 16px; color: rgba(167,139,250,0.25);
    position: relative; top: -22px; flex-shrink: 0; transition: color 0.3s;
}
.agent-arrow.active { color: #A78BFA; }

/* ── AGENT TRACE ── */
.trace-wrap {
    background: rgba(7,11,20,0.9);
    border: 1px solid rgba(167,139,250,0.2);
    border-radius: 14px;
    padding: 18px 20px;
    margin: 14px 0;
    position: relative;
    overflow: hidden;
}
.trace-wrap::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, #A78BFA, #6366F1);
}
.trace-label {
    font-family: 'JetBrains Mono', monospace; font-size: 0.6rem;
    letter-spacing: 0.2em; text-transform: uppercase; color: #A78BFA;
    margin-bottom: 12px; display: flex; align-items: center; gap: 8px;
}
.trace-label::after { content: ''; flex: 1; height: 1px; background: rgba(167,139,250,0.15); }
.trace-row {
    display: flex; align-items: flex-start; gap: 10px;
    padding: 8px 10px; border-radius: 8px; margin-bottom: 6px;
    background: rgba(167,139,250,0.05); border: 1px solid rgba(167,139,250,0.1);
    font-family: 'JetBrains Mono', monospace; font-size: 0.78rem;
}
.trace-row .tr-icon { flex-shrink: 0; margin-top: 1px; }
.trace-row .tr-action { color: #A78BFA; font-weight: 600; min-width: 70px; }
.trace-row .tr-detail { color: #64748B; flex: 1; word-break: break-word; }
.trace-row.tr-search { border-color: rgba(34,211,238,0.15); background: rgba(34,211,238,0.04); }
.trace-row.tr-search .tr-action { color: #22D3EE; }
.trace-row.tr-done   { border-color: rgba(52,211,153,0.15); background: rgba(52,211,153,0.04); }
.trace-row.tr-done .tr-action   { color: #34D399; }
.agent-count-badge {
    display: inline-flex; align-items: center; gap: 5px;
    background: rgba(167,139,250,0.12); border: 1px solid rgba(167,139,250,0.28);
    border-radius: 20px; padding: 3px 10px;
    font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; color: #C4B5FD;
    margin-left: 8px;
}
.routing-banner {
    font-family: 'JetBrains Mono', monospace; font-size: 0.72rem;
    padding: 8px 14px; border-radius: 10px; margin-bottom: 14px;
    border: 1px solid rgba(167,139,250,0.25); background: rgba(167,139,250,0.06); color: #C4B5FD;
}
.routing-banner.simple { border-color: rgba(34,211,238,0.25); background: rgba(34,211,238,0.05); color: #22D3EE; }
.subq-wrap {
    background: rgba(167,139,250,0.04); border: 1px solid rgba(167,139,250,0.15);
    border-radius: 12px; padding: 14px 16px; margin-bottom: 14px;
}
.subq-label {
    font-family: 'JetBrains Mono', monospace; font-size: 0.65rem;
    letter-spacing: 0.1em; text-transform: uppercase; color: #A78BFA; margin-bottom: 8px;
}
.subq-row { font-size: 0.84rem; color: #CBD5E1; padding: 5px 0; display: flex; gap: 8px; }
.subq-num { font-family: 'JetBrains Mono', monospace; color: #6366F1; font-weight: 600; flex-shrink: 0; }
/* ── SEARCH CARD ── */
.search-card {
    background: rgba(12,17,35,0.75); border: 1px solid rgba(99,102,241,0.22);
    border-radius: 18px; padding: 24px 28px;
    backdrop-filter: blur(18px); box-shadow: 0 20px 60px rgba(0,0,0,0.35); margin-bottom: 24px;
}
.search-card.agentic { border-color: rgba(167,139,250,0.3); }

/* ── INPUT ── */
[data-testid="stTextInput"] input {
    background: rgba(7,11,20,0.85) !important;
    border: 1px solid rgba(99,102,241,0.28) !important;
    border-radius: 12px !important; color: #F8FAFC !important;
    font-family: 'Inter', sans-serif !important; font-size: 0.97rem !important;
    padding: 13px 18px !important; transition: border-color 0.2s, box-shadow 0.2s !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: #6366F1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.14) !important; outline: none !important;
}
[data-testid="stTextInput"] input::placeholder { color: #334155 !important; }
[data-testid="stTextInput"] label {
    color: #475569 !important; font-size: 0.72rem !important;
    font-family: 'JetBrains Mono', monospace !important;
    letter-spacing: 0.1em !important; text-transform: uppercase !important;
}

/* ── BUTTONS ── */
[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, #6366F1, #4F46E5) !important;
    border: none !important; border-radius: 11px !important; color: #fff !important;
    font-family: 'Inter', sans-serif !important; font-weight: 600 !important;
    font-size: 0.88rem !important; padding: 12px 28px !important;
    box-shadow: 0 4px 18px rgba(99,102,241,0.38) !important; transition: all 0.2s !important;
}
[data-testid="stButton"] > button[kind="primary"]:hover {
    transform: translateY(-1px) !important; box-shadow: 0 8px 28px rgba(99,102,241,0.52) !important;
}
[data-testid="stButton"] > button:not([kind="primary"]) {
    background: rgba(99,102,241,0.08) !important;
    border: 1px solid rgba(99,102,241,0.22) !important;
    border-radius: 10px !important; color: #A5B4FC !important;
    font-family: 'Inter', sans-serif !important; font-weight: 500 !important;
    font-size: 0.85rem !important; transition: all 0.18s !important;
}
[data-testid="stButton"] > button:not([kind="primary"]):hover { background: rgba(99,102,241,0.15) !important; }

/* ── ANSWER CARD ── */
.answer-card {
    background: rgba(10,14,26,0.9); border: 1px solid rgba(99,102,241,0.18);
    border-radius: 16px; padding: 24px 28px; margin: 16px 0; position: relative; overflow: hidden;
}
.answer-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, #6366F1, #22D3EE);
}
.answer-card.agentic { border-color: rgba(167,139,250,0.25); }
.answer-card.agentic::before { background: linear-gradient(90deg, #A78BFA, #6366F1); }

.answer-label {
    font-family: 'JetBrains Mono', monospace; font-size: 0.6rem;
    letter-spacing: 0.2em; text-transform: uppercase; color: #6366F1;
    margin-bottom: 14px; display: flex; align-items: center; gap: 8px;
}
.answer-label::after { content: ''; flex: 1; height: 1px; background: rgba(99,102,241,0.15); }
.answer-label.agentic { color: #A78BFA; }
.answer-label.agentic::after { background: rgba(167,139,250,0.15); }

.answer-text {
    font-size: 0.95rem; line-height: 1.78; color: #CBD5E1;
    font-family: 'Inter', sans-serif; white-space: pre-wrap; word-break: break-word;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0} }
.cursor { color: #6366F1; animation: blink 0.75s step-end infinite; }

/* ── TIMING STRIP ── */
.timing-strip { display: flex; gap: 10px; margin: 12px 0 4px; flex-wrap: wrap; }
.timing-chip {
    background: rgba(10,14,26,0.8); border: 1px solid rgba(99,102,241,0.13);
    border-radius: 8px; padding: 6px 13px;
    font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; color: #475569;
    display: flex; align-items: center; gap: 5px;
}
.timing-chip .tc-val { color: #A5B4FC; font-weight: 600; }
.timing-chip .tc-mode {
    background: rgba(99,102,241,0.14); border: 1px solid rgba(99,102,241,0.24);
    color: #6366F1; padding: 1px 7px; border-radius: 20px; font-size: 0.62rem;
}
.timing-chip .tc-mode.agentic {
    background: rgba(167,139,250,0.14); border-color: rgba(167,139,250,0.28); color: #A78BFA;
}

/* ── SECTION HEADER ── */
.section-header { display: flex; align-items: center; gap: 10px; margin: 26px 0 13px; }
.sh-icon {
    width: 30px; height: 30px; background: rgba(99,102,241,0.11);
    border: 1px solid rgba(99,102,241,0.22); border-radius: 8px;
    display: flex; align-items: center; justify-content: center; font-size: 13px; flex-shrink: 0;
}
.sh-icon.agentic { background: rgba(167,139,250,0.1); border-color: rgba(167,139,250,0.28); }
.sh-title {
    font-family: 'JetBrains Mono', monospace; font-size: 0.62rem;
    letter-spacing: 0.18em; text-transform: uppercase; color: #64748B;
}
.section-header::after { content: ''; flex: 1; height: 1px; background: rgba(99,102,241,0.1); }

/* ── EXPANDERS ── */
[data-testid="stExpander"] {
    background: rgba(10,14,26,0.7) !important; border: 1px solid rgba(99,102,241,0.13) !important;
    border-radius: 13px !important; margin-bottom: 8px !important;
    overflow: hidden !important; transition: border-color 0.18s !important;
}
[data-testid="stExpander"]:hover { border-color: rgba(99,102,241,0.28) !important; }
[data-testid="stExpander"] summary {
    background: transparent !important; padding: 13px 16px !important;
    font-family: 'Inter', sans-serif !important; font-size: 0.86rem !important;
    color: #CBD5E1 !important; font-weight: 500 !important;
}
[data-testid="stExpanderDetails"] { background: rgba(7,11,20,0.55) !important; padding: 14px 16px !important; }

/* ── st.status override ── */
[data-testid="stStatusContainer"] {
    background: rgba(10,14,26,0.85) !important;
    border: 1px solid rgba(167,139,250,0.22) !important;
    border-radius: 13px !important; overflow: hidden !important;
}
[data-testid="stStatusContainer"] [data-testid="stMarkdownContainer"] p {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important; color: #94A3B8 !important;
}

/* ── SOURCE META ── */
.sim-badge {
    display: inline-block;
    background: linear-gradient(135deg, rgba(99,102,241,0.18), rgba(34,211,238,0.12));
    border: 1px solid rgba(99,102,241,0.28); color: #A5B4FC;
    font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; padding: 2px 8px; border-radius: 20px;
}
.meta-row { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; margin-bottom: 12px; }
.meta-tag {
    background: rgba(15,20,40,0.9); border: 1px solid rgba(99,102,241,0.12);
    border-radius: 5px; padding: 2px 8px;
    font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; color: #475569;
}
.meta-tag span { color: #94A3B8; }
.chunk-text {
    background: rgba(7,11,20,0.8); border-left: 2px solid rgba(99,102,241,0.35);
    border-radius: 0 8px 8px 0; padding: 12px 14px;
    font-family: 'JetBrains Mono', monospace; font-size: 0.78rem;
    line-height: 1.7; color: #64748B; margin-top: 10px; white-space: pre-wrap; word-break: break-word;
}

/* ── LINK BUTTON ── */
[data-testid="stLinkButton"] a {
    background: rgba(34,211,238,0.08) !important; border: 1px solid rgba(34,211,238,0.25) !important;
    border-radius: 8px !important; color: #22D3EE !important; font-size: 0.78rem !important;
    font-family: 'Inter', sans-serif !important; font-weight: 500 !important;
    padding: 7px 12px !important; transition: all 0.18s !important; text-decoration: none !important;
}
[data-testid="stLinkButton"] a:hover { background: rgba(34,211,238,0.15) !important; transform: translateY(-1px) !important; }

/* ── HISTORY ── */
.history-meta { display: flex; gap: 8px; margin-top: 10px; flex-wrap: wrap; }
.hchip {
    font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; color: #475569;
    background: rgba(99,102,241,0.07); border: 1px solid rgba(99,102,241,0.12);
    padding: 2px 8px; border-radius: 5px;
}
.hchip.agentic { background: rgba(167,139,250,0.08); border-color: rgba(167,139,250,0.2); color: #A78BFA; }

/* ── EVAL TAB ── */
.eval-header {
    background: rgba(10,14,26,0.7); border: 1px solid rgba(99,102,241,0.15);
    border-radius: 16px; padding: 22px 26px; margin-bottom: 24px; position: relative; overflow: hidden;
}
.eval-header::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, #22D3EE, #6366F1);
}
.eval-title {
    font-family: 'JetBrains Mono', monospace; font-size: 0.65rem;
    letter-spacing: 0.2em; text-transform: uppercase; color: #22D3EE; margin-bottom: 8px;
}
.eval-desc { font-size: 0.88rem; color: #64748B; line-height: 1.6; }
.insight-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin-top: 20px; }
.insight-card {
    background: rgba(10,14,26,0.8); border: 1px solid rgba(99,102,241,0.14);
    border-radius: 13px; padding: 18px 20px; transition: border-color 0.18s;
}
.insight-card:hover { border-color: rgba(99,102,241,0.3); }
.insight-retriever { font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; letter-spacing: 0.14em; text-transform: uppercase; margin-bottom: 6px; }
.insight-retriever.dense  { color: #6366F1; }
.insight-retriever.bm25   { color: #22D3EE; }
.insight-retriever.hybrid { color: #34D399; }
.insight-verdict { font-size: 0.82rem; color: #94A3B8; line-height: 1.55; }
.insight-tag { display: inline-block; margin-top: 8px; padding: 2px 8px; border-radius: 20px; font-family: 'JetBrains Mono', monospace; font-size: 0.62rem; }
.insight-tag.best     { background: rgba(52,211,153,0.12); border: 1px solid rgba(52,211,153,0.3); color: #34D399; }
.insight-tag.fast     { background: rgba(34,211,238,0.1);  border: 1px solid rgba(34,211,238,0.25); color: #22D3EE; }
.insight-tag.balanced { background: rgba(99,102,241,0.12); border: 1px solid rgba(99,102,241,0.28); color: #A5B4FC; }

/* ── WARNING ── */
[data-testid="stAlert"] {
    background: rgba(245,158,11,0.08) !important; border: 1px solid rgba(245,158,11,0.28) !important;
    border-radius: 11px !important; color: #FCD34D !important;
}

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.25); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(99,102,241,0.45); }
hr { border: none !important; border-top: 1px solid rgba(99,102,241,0.1) !important; margin: 28px 0 !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### Retrieval Mode")

    retrieval_mode = st.radio(
        "retrieval_mode",
        [
            "Agentic (Multi-Hop ReAct)",
            "Hybrid (RRF + Reranker)",
            "Dense (BGE Embeddings)",
            "Sparse (BM25)",
        ],
        label_visibility="collapsed",
        help="Switch retrieval algorithm live.",
    )

    mode_descriptions = {
        "Agentic (Multi-Hop ReAct)": ("Agentic", "LangGraph ReAct loop. Autonomously triggers multiple searches for complex queries.", True),
        "Hybrid (RRF + Reranker)":   ("Hybrid",  "RRF fusion + Cross-Encoder reranker. Best precision & faithfulness.", False),
        "Dense (BGE Embeddings)":    ("Dense",   "BGE-768 cosine search. Strong semantic recall, minimal latency.", False),
        "Sparse (BM25)":             ("BM25",    "Term-frequency matching. Exact keyword precision, lower recall.", False),
    }
    mode_key, mode_desc, is_agentic = mode_descriptions[retrieval_mode]
    card_class = "mode-card agentic" if is_agentic else "mode-card"

    st.markdown(f"""
    <div class='{card_class}'>
        <div class='mc-title'>{mode_key}</div>
        <div class='mc-sub'>{mode_desc}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Pipeline")
    st.markdown("""
    <div class='sidebar-stat'><span class='ss-label'>Papers</span><span class='ss-val'>251</span></div>
    <div class='sidebar-stat'><span class='ss-label'>Embeddings</span><span class='ss-val' style='color:#22D3EE !important;'>BGE-768</span></div>
    <div class='sidebar-stat'><span class='ss-label'>Vector store</span><span class='ss-val' style='color:#A5B4FC !important;'>ChromaDB</span></div>
    <div class='sidebar-stat'><span class='ss-label'>Generator</span><span class='ss-val' style='color:#34D399 !important;'>GPT-4o-mini</span></div>
    <div class='sidebar-stat'><span class='ss-label'>Agent</span><span class='ss-val' style='color:#C4B5FD !important;'>LangGraph</span></div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Retrieval")
    k = st.slider("Chunks (k)", min_value=1, max_value=10, value=5)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("⟳  Clear history", use_container_width=True):
        st.session_state.history = []
        st.rerun()


# ══════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════
def pipeline_bar(embed_state="idle", retrieve_state="idle", generate_state="idle"):
    def _cls(s):  return {"idle": "", "active": "active", "done": "done"}.get(s, "")
    def _conn(s): return "active" if s == "done" else ""
    st.markdown(f"""
    <div class='pipeline-wrap'>
        <div class='pipe-stage'>
            <div class='pipe-icon {_cls(embed_state)}'>🔢</div>
            <div class='pipe-label {_cls(embed_state)}'>Embed</div>
        </div>
        <div class='pipe-connector {_conn(embed_state)}'></div>
        <div class='pipe-stage'>
            <div class='pipe-icon {_cls(retrieve_state)}'>🔍</div>
            <div class='pipe-label {_cls(retrieve_state)}'>Retrieve</div>
        </div>
        <div class='pipe-connector {_conn(retrieve_state)}'></div>
        <div class='pipe-stage'>
            <div class='pipe-icon {_cls(generate_state)}'>⚡</div>
            <div class='pipe-label {_cls(generate_state)}'>Generate</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def agent_loop_bar(plan_state="idle", search_state="idle", reason_state="idle", answer_state="idle"):
    """Circular-node visualizer for the ReAct agent — visually distinct from the linear pipeline."""
    def _cls(s):  return {"idle": "", "active": "active", "done": "done"}.get(s, "")
    def _arr(s):  return "active" if s == "done" else ""
    st.markdown(f"""
    <div class='agent-loop-wrap'>
        <div class='agent-node'>
            <div class='agent-icon {_cls(plan_state)}'>🧠</div>
            <div class='agent-label {_cls(plan_state)}'>Plan</div>
        </div>
        <div class='agent-arrow {_arr(plan_state)}'>→</div>
        <div class='agent-node'>
            <div class='agent-icon {_cls(search_state)}'>🔍</div>
            <div class='agent-label {_cls(search_state)}'>Search</div>
        </div>
        <div class='agent-arrow {_arr(search_state)}'>→</div>
        <div class='agent-node'>
            <div class='agent-icon {_cls(reason_state)}'>⚙️</div>
            <div class='agent-label {_cls(reason_state)}'>Reason</div>
        </div>
        <div class='agent-arrow {_arr(reason_state)}'>→</div>
        <div class='agent-node'>
            <div class='agent-icon {_cls(answer_state)}'>✅</div>
            <div class='agent-label {_cls(answer_state)}'>Answer</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


@st.cache_data
def load_eval_data():
    results_dir = Path("results")
    data = []
    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    for mode, filename in [
        ("Dense",  "dense_eval.json"),
        ("BM25",   "bm25_eval.json"),
        ("Hybrid", "hybrid_eval.json"),
    ]:
        path = results_dir / filename
        if path.exists():
            with open(path) as f:
                scores = json.load(f)["scores"]
                for m in metrics:
                    data.append({"Metric": m, "Retriever": mode, "Score": scores.get(m, 0)})
    return pd.DataFrame(data) if data else None


# ══════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════
tab_chat, tab_eval = st.tabs(["💬  Assistant", "📊  Ablation Dashboard"])


# ──────────────────────────────────────────────────────────
# TAB 1 — CHAT
# ──────────────────────────────────────────────────────────
with tab_chat:

    eyebrow_text = "Dense · Sparse · Hybrid · <span style='color:#A78BFA'>Agentic ReAct</span>" if is_agentic else "Dense · Sparse · Hybrid · Agentic"

    st.markdown(f"""
    <div class='hero-wrap'>
        <div class='hero-eyebrow'>{eyebrow_text}</div>
        <h1 class='hero-title'>RAG <span class='grad'>Research</span> Assistant</h1>
        <p class='hero-sub'>Semantic search over 251 curated arXiv papers. Switch retrieval strategies live — including a LangGraph ReAct agent for multi-hop complex queries.</p>
    </div>
    """, unsafe_allow_html=True)

    pipe_placeholder = st.empty()
    with pipe_placeholder:
        if is_agentic:
            agent_loop_bar()
        else:
            pipeline_bar()

    # Search card
    card_cls = "search-card agentic" if is_agentic else "search-card"
    placeholder_text = (
        "e.g.  How do federated learning approaches in healthcare differ from general IoT deployments?"
        if is_agentic else
        "e.g.  How does FedAvg handle non-IID data distributions?"
    )
    st.markdown(f"<div class='{card_cls}'>", unsafe_allow_html=True)
    query = st.text_input("query", placeholder=placeholder_text, label_visibility="collapsed")
    col_btn, col_hint = st.columns([1, 5])
    with col_btn:
        btn_label = "🧠  Run Agent" if is_agentic else "⚡  Search"
        ask_clicked = st.button(btn_label, type="primary", use_container_width=True)
    with col_hint:
        hint = (
            "Agent will plan, search multiple times, and synthesise a grounded answer."
            if is_agentic else
            "Try: FedProx convergence · DP noise mechanisms · deepfake GAN forensics"
        )
        st.markdown(
            f"<p style='color:#334155;font-size:0.76rem;padding-top:10px;font-family:JetBrains Mono,monospace;'>{hint}</p>",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # ── QUERY EXECUTION ──────────────────────────────────
    if ask_clicked and query.strip():
        mode_short = retrieval_mode.split()[0]
        start_time = time.time()

        # ── AGENTIC PATH ──────────────────────────────────
        if mode_short == "Agentic":

            with pipe_placeholder:
                agent_loop_bar(plan_state="active")

            # ── NEW: routing decision + sub-questions ──
            classification = classify_query(query)
            is_complex = classification == "complex"

            st.markdown(f"""
            <div class='routing-banner {'complex' if is_complex else 'simple'}'>
                Query classified as: <strong>{classification}</strong>
                {' → decomposing into sub-questions' if is_complex else ' → answering directly'}
            </div>
            """, unsafe_allow_html=True)

            sub_questions = []
            if is_complex:
                sub_questions = decompose_query(query)
                subq_html = "".join(
                    f"<div class='subq-row'><span class='subq-num'>{i+1}</span>{html.escape(q)}</div>"
                    for i, q in enumerate(sub_questions)
                )
                st.markdown(f"""
                <div class='subq-wrap'>
                    <div class='subq-label'>🧩 Breaking into {len(sub_questions)} sub-questions</div>
                    {subq_html}
                </div>
                """, unsafe_allow_html=True)
            

            # Collect trace rows as agent runs
            trace_rows = []

            st.markdown("""
            <div class='section-header'>
                <div class='sh-icon agentic'>🧠</div>
                <div class='sh-title'>Agent Reasoning Trace</div>
            </div>
            """, unsafe_allow_html=True)

            trace_placeholder = st.empty()

            def render_trace(rows, running=True):
                inner = ""
                for row in rows:
                    inner += f"<div class='trace-row {row['cls']}'><span class='tr-icon'>{row['icon']}</span><span class='tr-action'>{row['action']}</span><span class='tr-detail'>{html.escape(row['detail'])}</span></div>"
                suffix = "<div class='trace-row' style='border:none;background:none;'><span class='tr-icon'>⏳</span><span class='tr-detail' style='color:#475569;'>Agent is thinking…</span></div>" if running else ""
                trace_placeholder.markdown(
                    f"<div class='trace-wrap'><div class='trace-label'>🧠 ReAct Trace</div>{inner}{suffix}</div>",
                    unsafe_allow_html=True,
                )

            render_trace([], running=True)

            with pipe_placeholder:
                agent_loop_bar(plan_state="done", search_state="active")

            
            # Run agent (wrapped to capture the LangSmith run for the trace link)
            ls_client = Client()
            with collect_runs() as run_collector:
                response = research_agent.invoke({
                    "messages": [
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=query),
                    ]
                })

            trace_url = None
            if run_collector.traced_runs:
                run_id = run_collector.traced_runs[0].id
                try:
                    trace_url = ls_client.share_run(run_id)
                except Exception:
                    trace_url = None

            with pipe_placeholder:
                agent_loop_bar(plan_state="done", search_state="done", reason_state="active")

            # Parse trace
            tool_calls_made = 0
            for msg in response["messages"]:
                if getattr(msg, "name", "") == "search_research_papers":
                    tool_calls_made += 1
                    trace_rows.append({
                        "cls": "tr-search",
                        "icon": "🔍",
                        "action": f"Search {tool_calls_made}",
                        "detail": str(msg.content)[:200],
                    })

            if tool_calls_made == 0:
                trace_rows.append({
                    "cls": "tr-done",
                    "icon": "⚡",
                    "action": "Direct",
                    "detail": "Answered directly without tool calls.",
                })

            trace_rows.append({
                "cls": "tr-done",
                "icon": "✅",
                "action": "Done",
                "detail": f"{tool_calls_made} search{'es' if tool_calls_made != 1 else ''} executed · synthesising final answer",
            })
            render_trace(trace_rows, running=False)

            with pipe_placeholder:
                agent_loop_bar(plan_state="done", search_state="done", reason_state="done", answer_state="done")

            final_answer = response["messages"][-1].content
            generation_elapsed = time.time() - start_time
            retrieval_elapsed = 0.0

            st.markdown("""
            <div class='section-header'>
                <div class='sh-icon agentic'>⚡</div>
                <div class='sh-title'>Final Agent Answer</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(
                f"<div class='answer-card agentic'>"
                f"<div class='answer-label agentic'>🧠 Agent Output</div>"
                f"<div class='answer-text'>{html.escape(final_answer)}</div></div>",
                unsafe_allow_html=True,
            )
            full_answer = final_answer
            if trace_url:
                st.link_button("🔗  View trace →", trace_url, use_container_width=False)
            else:
                st.caption("Trace not available — check LANGCHAIN_TRACING_V2 is set to true.")

        # ── STANDARD PIPELINE PATH ────────────────────────
        else:
            if retrieval_mode == "Dense (BGE Embeddings)":
                retriever_fn = retrieve_dense
            elif retrieval_mode == "Sparse (BM25)":
                retriever_fn = retrieve_bm25
            else:
                retriever_fn = retrieve_hybrid

            with pipe_placeholder: pipeline_bar(embed_state="active")
            time.sleep(0.05)
            with pipe_placeholder: pipeline_bar(embed_state="done", retrieve_state="active")
            with st.spinner(f"Running {mode_short} retrieval…"):
                chunks = retriever_fn(query, k=k)
            retrieval_elapsed = time.time() - start_time
            with pipe_placeholder: pipeline_bar(embed_state="done", retrieve_state="done", generate_state="active")

            st.markdown("""
            <div class='section-header'>
                <div class='sh-icon'>📄</div>
                <div class='sh-title'>Retrieved Sources</div>
            </div>
            """, unsafe_allow_html=True)

            if not chunks:
                st.warning("No chunks passed the similarity threshold for this query.")
            else:
                for i, chunk in enumerate(chunks, 1):
                    with st.expander(f"[{i}]  {chunk.title[:68]}{'…' if len(chunk.title)>68 else ''}"):
                        st.markdown(f"""
                        <div class='meta-row'>
                            <span class='sim-badge'>sim {chunk.similarity:.3f}</span>
                            <span class='meta-tag'>arXiv: <span>{chunk.arxiv_id}</span></span>
                            <span class='meta-tag'>year: <span>{chunk.year}</span></span>
                        </div>
                        """, unsafe_allow_html=True)
                        col_a, col_b = st.columns([4, 1])
                        with col_b:
                            st.link_button("↗  arXiv", f"https://arxiv.org/abs/{chunk.arxiv_id}", use_container_width=True)
                        preview = chunk.text[:500] + ("…" if len(chunk.text) > 500 else "")
                        st.markdown(f"<div class='chunk-text'>{html.escape(preview)}</div>", unsafe_allow_html=True)

            st.markdown("""
            <div class='section-header'>
                <div class='sh-icon'>⚡</div>
                <div class='sh-title'>Generated Answer</div>
            </div>
            """, unsafe_allow_html=True)

            answer_placeholder = st.empty()
            full_answer = ""
            gen_start = time.time()

            for token in generate_answer_stream(query, chunks):
                full_answer += token
                answer_placeholder.markdown(
                    f"<div class='answer-card'><div class='answer-label'>⚡ Generated Answer</div>"
                    f"<div class='answer-text'>{html.escape(full_answer)}<span class='cursor'>▋</span></div></div>",
                    unsafe_allow_html=True,
                )

            generation_elapsed = time.time() - gen_start
            answer_placeholder.markdown(
                f"<div class='answer-card'><div class='answer-label'>⚡ Generated Answer</div>"
                f"<div class='answer-text'>{html.escape(full_answer)}</div></div>",
                unsafe_allow_html=True,
            )
            with pipe_placeholder: pipeline_bar(embed_state="done", retrieve_state="done", generate_state="done")

        # ── SHARED: timing strip ──────────────────────────
        elapsed = time.time() - start_time
        mode_cls = "agentic" if mode_short == "Agentic" else ""
        searches_html = f"<div class='timing-chip'>Searches <span class='tc-val'>{tool_calls_made}</span></div>" if mode_short == "Agentic" else f"<div class='timing-chip'>Retrieve <span class='tc-val'>{retrieval_elapsed:.2f}s</span></div><div class='timing-chip'>Generate <span class='tc-val'>{generation_elapsed:.2f}s</span></div>"
        st.markdown(f"""
        <div class='timing-strip'>
            {searches_html}
            <div class='timing-chip'>Total <span class='tc-val'>{elapsed:.1f}s</span></div>
            <div class='timing-chip'>Mode <span class='tc-mode {mode_cls}'>{mode_short}</span></div>
            <div class='timing-chip'>k = <span class='tc-val'>{k}</span></div>
        </div>
        """, unsafe_allow_html=True)

        st.session_state.history.insert(0, {
            "query": query,
            "answer": full_answer,
            "mode": mode_short,
            "retrieval_time": retrieval_elapsed,
            "gen_time": generation_elapsed,
            "elapsed": elapsed,
            "tool_calls": tool_calls_made if mode_short == "Agentic" else None,
        })

    elif ask_clicked and not query.strip():
        st.warning("Enter a question to search the research corpus.")

    # Show last result when idle
    if st.session_state.history and not (ask_clicked and query.strip()):
        latest = st.session_state.history[0]
        is_prev_agentic = latest["mode"] == "Agentic"
        card_cls = "answer-card agentic" if is_prev_agentic else "answer-card"
        lbl_cls = "answer-label agentic" if is_prev_agentic else "answer-label"
        lbl_txt = "🧠 Agent Output" if is_prev_agentic else "⚡ Generated Answer"
        st.markdown(
            f"<div class='{card_cls}'><div class='{lbl_cls}'>{lbl_txt}</div>"
            f"<div class='answer-text'>{html.escape(latest['answer'])}</div></div>",
            unsafe_allow_html=True,
        )
        mode_cls = "agentic" if is_prev_agentic else ""
        extra = f"<div class='timing-chip'>Searches <span class='tc-val'>{latest['tool_calls']}</span></div>" if is_prev_agentic and latest.get("tool_calls") is not None else ""
        st.markdown(f"""
        <div class='timing-strip'>
            {extra}
            <div class='timing-chip'>Total <span class='tc-val'>{latest['elapsed']:.1f}s</span></div>
            <div class='timing-chip'>Mode <span class='tc-mode {mode_cls}'>{latest['mode']}</span></div>
        </div>
        """, unsafe_allow_html=True)

    # History
    if len(st.session_state.history) > 1:
        st.markdown("""
        <div class='section-header'>
            <div class='sh-icon'>🕐</div>
            <div class='sh-title'>Query History</div>
        </div>
        """, unsafe_allow_html=True)
        for item in st.session_state.history[1:]:
            m = item["mode"]
            with st.expander(f"{item['query']}  ·  {m}"):
                st.markdown(
                    f"<div style='font-size:0.88rem;color:#94A3B8;line-height:1.65;'>{html.escape(item['answer'])}</div>",
                    unsafe_allow_html=True,
                )
                hchip_m = "hchip agentic" if m == "Agentic" else "hchip"
                searches_chip = f"<span class='hchip agentic'>{item['tool_calls']} searches</span>" if m == "Agentic" and item.get("tool_calls") is not None else ""
                st.markdown(f"""
                <div class='history-meta'>
                    <span class='{hchip_m}'>{m}</span>
                    {searches_chip}
                    <span class='hchip'>total {item['elapsed']:.1f}s</span>
                </div>
                """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────
# TAB 2 — ABLATION DASHBOARD
# ──────────────────────────────────────────────────────────
with tab_eval:

    st.markdown("""
    <div class='eval-header'>
        <div class='eval-title'>📊 Retrieval Ablation Study</div>
        <div class='eval-desc'>
            RAGAS evaluation across 25 golden Q&amp;A pairs — comparing Dense (BGE-768),
            Sparse (BM25), and Hybrid (RRF + Cross-Encoder) retrieval architectures
            on Faithfulness, Answer Relevancy, Context Precision, and Context Recall.
        </div>
    </div>
    """, unsafe_allow_html=True)

    df_eval = load_eval_data()

    if df_eval is not None:
        COLOR_MAP = {"Dense": "#6366F1", "BM25": "#22D3EE", "Hybrid": "#34D399"}
        fig = go.Figure()
        for retriever, color in COLOR_MAP.items():
            subset = df_eval[df_eval["Retriever"] == retriever]
            fig.add_trace(go.Bar(
                name=retriever, x=subset["Metric"], y=subset["Score"],
                marker=dict(color=color, opacity=0.82, line=dict(color=color, width=1)),
                text=[f"{v:.3f}" for v in subset["Score"]], textposition="outside",
                textfont=dict(family="JetBrains Mono", size=11, color=color),
            ))
        fig.update_layout(
            barmode="group", height=420,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(7,11,20,0.6)",
            font=dict(family="Inter", color="#64748B", size=12),
            legend=dict(bgcolor="rgba(10,14,26,0.8)", bordercolor="rgba(99,102,241,0.2)", borderwidth=1,
                        font=dict(family="JetBrains Mono", size=11, color="#94A3B8"),
                        orientation="h", x=0.5, xanchor="center", y=1.08, yanchor="bottom"),
            xaxis=dict(tickfont=dict(family="JetBrains Mono", size=11, color="#64748B"),
                       gridcolor="rgba(99,102,241,0.07)", linecolor="rgba(99,102,241,0.15)", title=None),
            yaxis=dict(range=[0.5, 1.05],
                       tickfont=dict(family="JetBrains Mono", size=10, color="#475569"),
                       gridcolor="rgba(99,102,241,0.07)", linecolor="rgba(99,102,241,0.1)",
                       title=dict(text="RAGAS Score", font=dict(size=11, color="#475569")), tickformat=".2f"),
            bargap=0.25, bargroupgap=0.06, margin=dict(t=40, b=20, l=50, r=20),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        <div class='insight-grid'>
            <div class='insight-card'>
                <div class='insight-retriever dense'>Dense · BGE-768</div>
                <div class='insight-verdict'>Strong semantic recall — surfaces conceptually related chunks even without keyword overlap. Occasionally retrieves semantically similar noise at the margin.</div>
                <span class='insight-tag fast'>Fastest latency</span>
            </div>
            <div class='insight-card'>
                <div class='insight-retriever bm25'>Sparse · BM25</div>
                <div class='insight-verdict'>High precision on exact terminology (arXiv IDs, algorithm names). Context Recall drops ~20% on paraphrased questions that lack surface-form overlap.</div>
                <span class='insight-tag fast'>No GPU required</span>
            </div>
            <div class='insight-card'>
                <div class='insight-retriever hybrid'>Hybrid · RRF + Reranker</div>
                <div class='insight-verdict'>Cross-Encoder reranker acts as an aggressive filter — maximises Faithfulness and Precision at the cost of ~150ms overhead and a marginal recall dip.</div>
                <span class='insight-tag best'>Best overall</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.warning(
            "Evaluation results not found. "
            "Run `python -m src.evaluate` to generate `results/dense_eval.json`, "
            "`bm25_eval.json`, and `hybrid_eval.json`."
        )