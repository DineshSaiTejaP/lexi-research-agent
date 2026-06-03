"""
Lexi — Legal Precedent Research Agent
Streamlit UI
"""

import logging
import time
import streamlit as st
from agent import run_agent

# ── Logging — all output captured by Streamlit Cloud logs ─────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("lexi.app")

st.set_page_config(
    page_title="Lexi — Legal Research Agent",
    page_icon="⚖️",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.header {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 60%, #0f3460 100%);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 14px;
    padding: 28px 36px;
    margin-bottom: 24px;
}
.header h1 { color: #e2e8f0; font-size: 1.8rem; font-weight: 700; margin: 0 0 6px; }
.header p  { color: #94a3b8; font-size: 0.9rem; margin: 0; }
.badge {
    display: inline-block;
    background: rgba(99,102,241,0.15);
    border: 1px solid rgba(99,102,241,0.35);
    color: #818cf8;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    margin-bottom: 10px;
}
.brief {
    background: #1e1b4b;
    border-left: 4px solid #8b5cf6;
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 18px;
}
.brief h4 { color: #a78bfa; margin: 0 0 8px; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; }
.brief p  { color: #cbd5e1; font-size: 0.88rem; margin: 0; line-height: 1.6; }

.step-thought { background: rgba(251,191,36,0.06); border: 1px solid rgba(251,191,36,0.2); border-radius: 7px; padding: 10px 14px; margin: 6px 0; color: #fbbf24; font-size: 0.83rem; }
.step-tool    { background: rgba(99,102,241,0.06); border: 1px solid rgba(99,102,241,0.2); border-radius: 7px; padding: 10px 14px; margin: 6px 0; }
.step-tool .tool-name { color: #818cf8; font-weight: 600; font-size: 0.76rem; text-transform: uppercase; letter-spacing: 0.5px; }
.step-tool .tool-input { color: #cbd5e1; font-size: 0.83rem; margin-top: 3px; }
.step-result  { background: rgba(16,185,129,0.05); border: 1px solid rgba(16,185,129,0.18); border-radius: 7px; padding: 10px 14px; margin: 6px 0; color: #6ee7b7; font-size: 0.8rem; max-height: 180px; overflow-y: auto; white-space: pre-wrap; }
.step-error   { background: rgba(239,68,68,0.05); border: 1px solid rgba(239,68,68,0.2); border-radius: 7px; padding: 10px 14px; margin: 6px 0; color: #fca5a5; font-size: 0.83rem; }

.answer-wrap {
    background: #0f1117;
    border: 1px solid rgba(99,102,241,0.18);
    border-radius: 14px;
    padding: 24px 28px;
    margin-top: 14px;
}
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header">
  <div class="badge">⚖️ LEGAL RESEARCH AI</div>
  <h1>Lexi — Precedent Research Agent</h1>
  <p>50+ Indian court judgments · LangGraph agent · Hybrid vector + BM25 retrieval · Full reasoning trace</p>
</div>
""", unsafe_allow_html=True)

# ── Session state init ─────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "query_input" not in st.session_state:
    st.session_state.query_input = ""

col_main, col_steps = st.columns([3, 2])

# ── Left column ────────────────────────────────────────────────────────────────
with col_main:
    # ── Sample query buttons ── each click populates the text area and reruns
    SAMPLE_QUERIES = [
        "Find precedents where an insurer was held liable despite the driver being unlicensed.",
        "What adverse precedents exist for a policy breach due to an invalid license?",
        "Which judgments involve Section 149 of the Motor Vehicles Act?",
        "Which of these judgments involve commercial vehicles?",
        "What compensation range is realistic for a 42-year-old earning ₹35,000/month?",
        "Discuss cases involving the 'pay and recover' doctrine.",
    ]

    st.markdown("**Try a sample query:**")
    c1, c2 = st.columns(2)
    for i, q in enumerate(SAMPLE_QUERIES):
        label = q[:58] + "…" if len(q) > 60 else q
        if (c1 if i % 2 == 0 else c2).button(label, key=f"sq_{i}", use_container_width=True):
            # Set session state THEN rerun so text_area picks up the new value
            st.session_state.query_input = q
            logger.info(f"Sample query selected: {q[:60]}")
            st.rerun()

    st.divider()

    # ── Text area — driven by session state so buttons reliably populate it ──
    query = st.text_area(
        "Your research query:",
        key="query_input",
        height=90,
        placeholder="Ask anything about the corpus…",
    )
    go = st.button("🔍 Research", type="primary", use_container_width=True)

# ── Right column placeholder ───────────────────────────────────────────────────
with col_steps:
    st.markdown("### 🧠 Reasoning Trace")
    trace_slot = st.empty()
    trace_slot.info("Run a query to see how the agent reasons step by step.")

# ── Run agent ──────────────────────────────────────────────────────────────────
if go and query.strip():
    logger.info(f"[QUERY START] {query.strip()[:120]}")
    t0 = time.time()

    with col_main:
        with st.spinner("Researching the corpus…"):
            result = run_agent(query.strip())

    elapsed = round(time.time() - t0, 1)
    n_steps  = len(result["steps"])
    n_tools  = sum(1 for s in result["steps"] if s.get("type") == "tool_call")
    docs_set = {
        w for s in result["steps"]
        for w in str(s.get("content", "")).split()
        if w.startswith("DOC_")
    }

    if result["error"]:
        logger.error(f"[QUERY ERROR] {result['error'][:200]}")
    else:
        logger.info(
            f"[QUERY DONE] elapsed={elapsed}s steps={n_steps} "
            f"tool_calls={n_tools} docs_seen={len(docs_set)}"
        )

    st.session_state.history.append(result | {"query": query.strip(), "elapsed": elapsed})

    # ── Render reasoning trace ──
    with col_steps:
        trace_slot.empty()
        steps = result["steps"]
        st.markdown(f"**{len(steps)} steps · {elapsed}s**")
        for i, step in enumerate(steps, 1):
            t = step.get("type", "")
            if t == "thought":
                with st.expander(f"💭 Thought {i}", expanded=False):
                    st.markdown(
                        f'<div class="step-thought">{step["content"]}</div>',
                        unsafe_allow_html=True,
                    )
            elif t == "tool_call":
                with st.expander(f"🔧 {step['tool_name']}", expanded=True):
                    st.markdown(
                        f'<div class="step-tool">'
                        f'<div class="tool-name">🔧 {step["tool_name"]}</div>'
                        f'<div class="tool-input">{step["content"]}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
            elif t == "tool_result":
                with st.expander(f"📄 Result from {step.get('tool_name', 'tool')}", expanded=True):
                    st.markdown(
                        f'<div class="step-result">{step["content"]}</div>',
                        unsafe_allow_html=True,
                    )
            elif t == "retrieval_summary":
                with st.expander("📊 Retrieval Summary", expanded=True):
                    st.markdown(
                        f'<div class="step-thought">{step["content"]}</div>', # Reuse thought style
                        unsafe_allow_html=True,
                    )
            elif t == "error":
                st.markdown(
                    f'<div class="step-error">⚠️ {step["content"]}</div>',
                    unsafe_allow_html=True,
                )

    # ── Render answer ──
    with col_main:
        m1, m2, m3 = st.columns(3)
        m1.metric("Tool Calls", n_tools)
        m2.metric("Docs Referenced", len(docs_set))
        m3.metric("Total Steps", n_steps)

        st.markdown("### Research Findings")
        if result["error"]:
            st.error(result["error"])
        else:
            # Use st.markdown inside a styled container so ### headers render
            st.markdown('<div class="answer-wrap">', unsafe_allow_html=True)
            st.markdown(result["answer"])
            st.markdown('</div>', unsafe_allow_html=True)

elif go:
    with col_main:
        st.warning("Please enter a query.")

# ── Query history ──────────────────────────────────────────────────────────────
past = st.session_state.history[:-1] if st.session_state.history else []
if past:
    with col_main:
        st.divider()
        st.markdown("### Query History")
        for item in reversed(past):
            label = f"{item['query'][:70]}  ·  {item.get('elapsed', '?')}s"
            with st.expander(label):
                st.markdown(item["answer"])