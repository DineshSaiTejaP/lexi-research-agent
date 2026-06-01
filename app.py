"""
Lexi — Legal Precedent Research Agent
Streamlit UI
"""

import streamlit as st
from agent import run_agent

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
.step-result  { background: rgba(16,185,129,0.05); border: 1px solid rgba(16,185,129,0.18); border-radius: 7px; padding: 10px 14px; margin: 6px 0; color: #6ee7b7; font-size: 0.8rem; max-height: 180px; overflow-y: auto; }
.step-error   { background: rgba(239,68,68,0.05); border: 1px solid rgba(239,68,68,0.2); border-radius: 7px; padding: 10px 14px; margin: 6px 0; color: #fca5a5; font-size: 0.83rem; }

.answer-box {
    background: #0f1117;
    border: 1px solid rgba(99,102,241,0.18);
    border-radius: 14px;
    padding: 24px 28px;
    margin-top: 14px;
    color: #e2e8f0;
    line-height: 1.7;
}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header">
  <div class="badge">⚖️ LEGAL RESEARCH AI</div>
  <h1>Lexi — Precedent Research Agent</h1>
  <p>56 Indian court judgments · ReAct agent · Hybrid vector + BM25 retrieval · Full reasoning trace</p>
</div>
""", unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = []

col_main, col_steps = st.columns([3, 2])

with col_main:
    st.markdown("""
    <div class="brief">
      <h4>📋 Reference Case Brief</h4>
      <p>
        <strong style="color:#e2e8f0">Client:</strong> Mrs. Lakshmi Devi — motor accident death claim<br>
        <strong style="color:#e2e8f0">Facts:</strong> Husband killed by commercial truck; driver had no valid licence.
        National Insurance Co. is denying the claim, arguing the policy is void due to the unlicensed driver.<br>
        <strong style="color:#e2e8f0">Profile:</strong> Deceased age 42, ₹35,000/month income, wife + 2 minor children (ages 8 & 12)
      </p>
    </div>
    """, unsafe_allow_html=True)

    sample_queries = [
        "Find precedents supporting Mrs. Lakshmi Devi's claim against the insurance company.",
        "What adverse precedents could National Insurance use, and how do we counter them?",
        "Which judgments involve Section 149 of the Motor Vehicles Act?",
        "Which of these judgments involve commercial vehicles?",
        "What compensation range is realistic for a 42-year-old earning ₹35,000/month?",
        "Find cases where an insurer was held liable despite the driver being unlicensed.",
    ]

    st.markdown("**Try a sample query:**")
    c1, c2 = st.columns(2)
    for i, q in enumerate(sample_queries):
        btn_label = q[:58] + "…" if len(q) > 60 else q
        if (c1 if i % 2 == 0 else c2).button(btn_label, key=f"s{i}", use_container_width=True):
            st.session_state["queued"] = q

    st.divider()

    query = st.text_area(
        "Your research query:",
        value=st.session_state.pop("queued", ""),
        height=90,
        placeholder="Ask anything about the corpus…",
    )
    go = st.button("🔍 Research", type="primary", use_container_width=True)

with col_steps:
    st.markdown("### 🧠 Reasoning Trace")
    trace_slot = st.empty()
    trace_slot.info("Run a query to see how the agent reasons step by step.")

# ── Run ───────────────────────────────────────────────────────────────────────
if go and query.strip():
    with col_main:
        with st.spinner("Researching the corpus…"):
            result = run_agent(query.strip())

    st.session_state.history.append(result | {"query": query.strip()})

    # Render reasoning trace
    with col_steps:
        trace_slot.empty()
        steps = result["steps"]
        st.markdown(f"**{len(steps)} steps captured**")
        for i, step in enumerate(steps, 1):
            t = step.get("type", "")
            if t == "thought":
                with st.expander(f"💭 Thought {i}", expanded=False):
                    st.markdown(f'<div class="step-thought">{step["content"]}</div>', unsafe_allow_html=True)
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
                with st.expander(f"📄 Result {i}", expanded=False):
                    st.markdown(f'<div class="step-result">{step["content"]}</div>', unsafe_allow_html=True)
            elif t == "error":
                st.markdown(f'<div class="step-error">⚠️ {step["content"]}</div>', unsafe_allow_html=True)

    # Render answer
    with col_main:
        tool_calls = sum(1 for s in result["steps"] if s.get("type") == "tool_call")
        docs_cited = len({w for s in result["steps"] for w in str(s.get("content", "")).split() if w.startswith("DOC_")})

        m1, m2, m3 = st.columns(3)
        m1.metric("Tool Calls", tool_calls)
        m2.metric("Docs Referenced", docs_cited)
        m3.metric("Total Steps", len(result["steps"]))

        st.markdown("### Research Findings")
        if result["error"]:
            st.error(result["error"])
        else:
            st.markdown(f'<div class="answer-box">{result["answer"]}</div>', unsafe_allow_html=True)

elif go:
    with col_main:
        st.warning("Please enter a query.")

# ── History ───────────────────────────────────────────────────────────────────
past = st.session_state.history[:-1] if st.session_state.history else []
if past:
    with col_main:
        st.divider()
        st.markdown("### Query History")
        for item in reversed(past):
            with st.expander(item["query"][:80]):
                st.markdown(item["answer"])
