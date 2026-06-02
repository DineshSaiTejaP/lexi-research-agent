"""
LangGraph legal precedent research agent.

Graph topology:
    START
      │
   [router]            ← LLM classifies query: "general" or "deep_research"
      │
   [retriever]         ← Always runs vector + keyword + adversarial pass
      │
    ┌─┴──────────────────┐
[general_answerer]  [supporting_analyzer]
      │                   │
     END           [adverse_analyzer]
                          │
                  [strategy_synthesizer]
                          │
                         END

Routing is LLM-driven — no hardcoded keyword matching.
Simple queries (1 LLM call + retrieval) vs deep research (4 LLM calls + retrieval).
"""

import os
import re
import time
import logging
from typing import TypedDict, Literal

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage

from tools.vector_search import vector_search
from tools.keyword_search import keyword_search

load_dotenv()
logger = logging.getLogger(__name__)

# ── Streamlit secret injection ─────────────────────────────────────────────────
try:
    import streamlit as st
    for key in ["GROQ_API_KEY", "GOOGLE_API_KEY", "LLM_PROVIDER", "LLM_MODEL",
                "CHROMA_PERSIST_DIR", "TOP_K_RETRIEVAL"]:
        if key in st.secrets and not os.environ.get(key):
            os.environ[key] = str(st.secrets[key])
except Exception:
    pass


# ── Rate limit helpers ─────────────────────────────────────────────────────────
_last_call_at: float = 0.0
MIN_CALL_GAP = 2.0


def _throttle():
    global _last_call_at
    elapsed = time.time() - _last_call_at
    if elapsed < MIN_CALL_GAP:
        time.sleep(MIN_CALL_GAP - elapsed)
    _last_call_at = time.time()


def _parse_retry_delay(error_msg: str, default: float = 10.0) -> float:
    patterns = [
        r"retry[_ ](?:after|in|delay)[:\s]+(\d+(?:\.\d+)?)\s*s",
        r"please retry in (\d+(?:\.\d+)?)s",
        r"\"retryDelay\":\s*\"(\d+(?:\.\d+)?)s\"",
        r"(\d+(?:\.\d+)?)\s*seconds",
    ]
    for pat in patterns:
        m = re.search(pat, str(error_msg), re.IGNORECASE)
        if m:
            return min(float(m.group(1)) + 2, 90.0)
    return default


def _is_rate_limit(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(k in msg for k in ["429", "resource_exhausted", "rate limit",
                                   "too many requests", "quota", "503",
                                   "service unavailable"])


def _invoke_with_retry(llm, messages, max_retries: int = 4):
    backoff = [10, 20, 40, 60]
    for attempt in range(max_retries + 1):
        _throttle()
        try:
            return llm.invoke(messages)
        except Exception as exc:
            if not _is_rate_limit(exc):
                raise
            if attempt == max_retries:
                raise
            wait = _parse_retry_delay(str(exc), default=backoff[min(attempt, len(backoff) - 1)])
            logger.warning(f"Rate limit (attempt {attempt+1}/{max_retries}) — waiting {wait:.0f}s")
            try:
                st.toast(f"⏳ Rate limit — retrying in {wait:.0f}s ({attempt+1}/{max_retries})…")
            except Exception:
                pass
            time.sleep(wait)


# ── LLM factory ────────────────────────────────────────────────────────────────
def build_llm():
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    model = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
    max_tokens = int(os.getenv("LLM_MAX_TOKENS", "1500"))

    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=model, temperature=0.1, max_tokens=max_tokens)
    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model, temperature=0.1, max_tokens=max_tokens)
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, temperature=0.1, max_tokens=max_tokens)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER '{provider}'. Use groq, google, or openai.")


# ── Graph State ────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    query: str
    query_type: str          # "general" | "deep_research"
    retrieved_docs: str      # formatted retrieval output passed to all LLM nodes
    supporting_section: str
    adverse_section: str
    strategy_section: str
    final_answer: str
    steps: list              # reasoning trace consumed by app.py UI
    error: str | None


# ── Node 1: router ─────────────────────────────────────────────────────────────
def router_node(state: AgentState) -> AgentState:
    """
    LLM classifies the query as 'general' or 'deep_research'.
    This drives the conditional edge after retrieval — no hardcoded keywords.
    """
    query = state["query"]
    steps = list(state.get("steps", []))

    steps.append({"type": "thought", "content": f"Classifying query intent: '{query[:80]}'"})

    llm = build_llm()
    prompt = f"""You are a legal research assistant. Classify the user query below.

Query: {query}

Respond with EXACTLY one word:
- "deep_research" — if the query asks for precedent research, case analysis, 
  legal strategy, supporting/adverse cases, compensation calculation, or any 
  task that requires synthesising information across multiple judgments.
- "general" — if the query asks for a simple factual list, a filter, or a 
  direct lookup (e.g. "which cases involve trucks", "show Supreme Court cases").

Classification:"""

    try:
        response = _invoke_with_retry(llm, [HumanMessage(content=prompt)])
        raw = response.content.strip().lower()
        query_type = "deep_research" if "deep" in raw else "general"
    except Exception as exc:
        query_type = "deep_research"   # safe default — never under-delivers
        steps.append({"type": "error", "content": f"Router fallback to deep_research: {exc}"})

    steps.append({"type": "thought", "content": f"Query classified as: **{query_type}**"})
    return {**state, "query_type": query_type, "steps": steps}


# ── Node 2: retriever ──────────────────────────────────────────────────────────
def retriever_node(state: AgentState) -> AgentState:
    """
    Runs vector search + keyword search on every query.
    For deep_research, adds a second keyword pass targeting adversarial patterns
    so adverse precedents are never crowded out by supporting ones.
    """
    query = state["query"]
    query_type = state["query_type"]
    steps = list(state.get("steps", []))
    top_k = int(os.getenv("TOP_K_RETRIEVAL", "8"))

    # ── Vector search ──
    steps.append({"type": "tool_call", "tool_name": "vector_search", "content": query})
    try:
        vec_results = vector_search(query, top_k=top_k)
        vec_text = _format_hits(vec_results, "vector")
    except Exception as exc:
        vec_text = f"Vector search error: {exc}"
    steps.append({"type": "tool_result", "content": vec_text[:1500]})

    # ── Keyword (BM25) search ──
    steps.append({"type": "tool_call", "tool_name": "keyword_search", "content": query})
    try:
        kw_results = keyword_search(query, top_k=top_k)
        kw_text = _format_hits(kw_results, "keyword")
    except Exception as exc:
        kw_text = f"Keyword search error: {exc}"
    steps.append({"type": "tool_result", "content": kw_text[:1500]})

    # ── Adversarial pass (deep_research only) ──
    adv_text = ""
    if query_type == "deep_research":
        adv_query = "insurer absolved policy void no liability fundamental breach unlicensed"
        steps.append({
            "type": "tool_call",
            "tool_name": "keyword_search (adversarial pass)",
            "content": adv_query,
        })
        try:
            adv_results = keyword_search(adv_query, top_k=5)
            adv_text = _format_hits(adv_results, "adversarial-keyword")
        except Exception as exc:
            adv_text = ""
        steps.append({"type": "tool_result", "content": adv_text[:1000]})

    retrieved = f"{vec_text}\n\n{kw_text}"
    if adv_text:
        retrieved += f"\n\n[Adversarial pass — insurer-favouring judgments]\n{adv_text}"

    return {**state, "retrieved_docs": retrieved, "steps": steps}


def _format_hits(results: list, source: str) -> str:
    if not results:
        return f"No results from {source} search."
    lines = [f"Results from {source} search ({len(results)} hits):\n"]
    for i, r in enumerate(results, 1):
        lines.append(
            f"{i}. [{r['doc_id']}] {r.get('case_name','')[:55]} "
            f"({r.get('court','')[:30]}, {r.get('year','')}) | "
            f"{r.get('text','')[:300]}…"
        )
    return "\n".join(lines)


# ── Node 3a: general_answerer ──────────────────────────────────────────────────
def general_answerer_node(state: AgentState) -> AgentState:
    """
    Single LLM call for simple/factual queries.
    Path: START → router → retriever → general_answerer → END
    """
    steps = list(state.get("steps", []))
    steps.append({"type": "thought", "content": "Formulating direct answer from retrieved documents."})

    llm = build_llm()
    prompt = f"""You are Lexi, an AI legal research assistant specialising in Indian court judgments.
Answer the following query using ONLY the retrieved documents below.

Query: {state['query']}

Retrieved Documents:
{state['retrieved_docs'][:4000]}

Rules:
- Cite documents as [DOC_XXX]
- Do not fabricate any case or legal principle not in the retrieved list
- Be concise and direct

Answer:"""

    try:
        response = _invoke_with_retry(llm, [HumanMessage(content=prompt)])
        answer = response.content
    except Exception as exc:
        answer = f"Error generating answer: {exc}"
        steps.append({"type": "error", "content": str(exc)})

    return {**state, "final_answer": answer, "steps": steps}


# ── Node 3b: supporting_analyzer ──────────────────────────────────────────────
def supporting_analyzer_node(state: AgentState) -> AgentState:
    """
    Extracts judgments that support the client's case.
    Path: retriever → supporting_analyzer → adverse_analyzer → strategy_synthesizer → END
    """
    steps = list(state.get("steps", []))
    steps.append({"type": "thought", "content": "Extracting SUPPORTING precedents from retrieved judgments."})

    llm = build_llm()
    prompt = f"""You are Lexi, an AI legal research assistant specialising in Indian court judgments.

Research Query: {state['query']}

Retrieved Judgments:
{state['retrieved_docs'][:4000]}

Write the Supporting Precedents section. For each supporting judgment include:
- Citation: [DOC_XXX] Case Name (Court, Year)
- Legal principle the judgment establishes
- Which specific facts align with the client's situation
- How this judgment strengthens the argument

Only cite documents that appear in the retrieved list above. Do not fabricate cases.

### Supporting Precedents"""

    try:
        response = _invoke_with_retry(llm, [HumanMessage(content=prompt)])
        supporting = response.content
    except Exception as exc:
        supporting = f"[Error generating supporting analysis: {exc}]"
        steps.append({"type": "error", "content": str(exc)})

    steps.append({"type": "thought", "content": "Supporting precedents section complete."})
    return {**state, "supporting_section": supporting, "steps": steps}


# ── Node 4: adverse_analyzer ───────────────────────────────────────────────────
def adverse_analyzer_node(state: AgentState) -> AgentState:
    """
    Identifies and honestly assesses judgments that work AGAINST the client.
    Dedicated node ensures adverse cases are never crowded out by supporting ones.
    """
    steps = list(state.get("steps", []))
    steps.append({"type": "thought", "content": "Extracting ADVERSE precedents — cases the opposing counsel could use."})

    llm = build_llm()
    prompt = f"""You are Lexi, an AI legal research assistant.

Research Query: {state['query']}

Retrieved Judgments (including adversarial pass results):
{state['retrieved_docs'][:4000]}

Write the Adverse Precedents section. A well-prepared legal team must know both sides.
For each adverse judgment include:
- Citation: [DOC_XXX] Case Name (Court, Year)
- Risk Level: HIGH / MEDIUM / LOW
- Why this precedent hurts the client's case
- How to distinguish or counter it in argument

Be honest — do not suppress unfavourable cases. Only cite from the retrieved list.

### Adverse Precedents"""

    try:
        response = _invoke_with_retry(llm, [HumanMessage(content=prompt)])
        adverse = response.content
    except Exception as exc:
        adverse = f"[Error generating adverse analysis: {exc}]"
        steps.append({"type": "error", "content": str(exc)})

    steps.append({"type": "thought", "content": "Adverse precedents section complete."})
    return {**state, "adverse_section": adverse, "steps": steps}


# ── Node 5: strategy_synthesizer ──────────────────────────────────────────────
def strategy_synthesizer_node(state: AgentState) -> AgentState:
    """
    Final synthesis node. Combines supporting + adverse into actionable strategy memo.
    """
    steps = list(state.get("steps", []))
    steps.append({"type": "thought", "content": "Synthesising final strategy recommendation."})

    llm = build_llm()
    prompt = f"""You are Lexi, an AI legal research assistant.

Research Query: {state['query']}

Supporting Precedents Identified:
{state['supporting_section'][:2000]}

Adverse Precedents Identified:
{state['adverse_section'][:2000]}

Write the Strategy Recommendation section:
- Priority arguments to make (ranked by strength)
- Realistic compensation range based on cited judgments
- Key risks the client must be aware of
- Recommended next steps for the legal team

### Strategy Recommendation"""

    try:
        response = _invoke_with_retry(llm, [HumanMessage(content=prompt)])
        strategy = response.content
    except Exception as exc:
        strategy = f"[Error generating strategy: {exc}]"
        steps.append({"type": "error", "content": str(exc)})

    final_answer = (
        f"### Supporting Precedents\n\n{state['supporting_section']}\n\n"
        f"---\n\n"
        f"### Adverse Precedents\n\n{state['adverse_section']}\n\n"
        f"---\n\n"
        f"### Strategy Recommendation\n\n{strategy}"
    )

    steps.append({"type": "thought", "content": "Research memo complete — all three sections generated."})
    return {**state, "strategy_section": strategy, "final_answer": final_answer, "steps": steps}


# ── Conditional edge function ──────────────────────────────────────────────────
def route_after_retrieval(state: AgentState) -> str:
    """
    The only branching point in the graph.
    Returns node name based on the router's LLM classification.
    """
    return "general_answerer" if state["query_type"] == "general" else "supporting_analyzer"


# ── Build & compile the graph ──────────────────────────────────────────────────
def build_graph():
    graph = StateGraph(AgentState)

    # Nodes
    graph.add_node("router",               router_node)
    graph.add_node("retriever",            retriever_node)
    graph.add_node("general_answerer",     general_answerer_node)
    graph.add_node("supporting_analyzer",  supporting_analyzer_node)
    graph.add_node("adverse_analyzer",     adverse_analyzer_node)
    graph.add_node("strategy_synthesizer", strategy_synthesizer_node)

    # Edges
    graph.add_edge(START,       "router")
    graph.add_edge("router",    "retriever")

    graph.add_conditional_edges(
        "retriever",
        route_after_retrieval,
        {
            "general_answerer":    "general_answerer",
            "supporting_analyzer": "supporting_analyzer",
        },
    )

    graph.add_edge("general_answerer",     END)
    graph.add_edge("supporting_analyzer",  "adverse_analyzer")
    graph.add_edge("adverse_analyzer",     "strategy_synthesizer")
    graph.add_edge("strategy_synthesizer", END)

    return graph.compile()


# ── Public API (unchanged interface — app.py needs no edits) ───────────────────
_graph = None


def run_agent(query: str) -> dict:
    """Run the LangGraph agent. Returns {answer, steps, error}."""
    global _graph
    if _graph is None:
        _graph = build_graph()

    initial_state: AgentState = {
        "query": query,
        "query_type": "",
        "retrieved_docs": "",
        "supporting_section": "",
        "adverse_section": "",
        "strategy_section": "",
        "final_answer": "",
        "steps": [],
        "error": None,
    }

    try:
        final_state = _graph.invoke(initial_state)
        return {
            "answer": final_state["final_answer"],
            "steps": final_state["steps"],
            "error": final_state.get("error"),
        }
    except Exception as exc:
        msg = str(exc)
        if _is_rate_limit(exc):
            wait = _parse_retry_delay(msg, default=30)
            friendly = (
                f"**Rate limit reached.**\n\n"
                f"Please wait **{wait:.0f} seconds** and resubmit your query."
            )
            return {"answer": friendly, "steps": [], "error": friendly}
        return {"answer": f"Error: {msg}", "steps": [], "error": msg}