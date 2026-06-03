"""
LangGraph legal precedent research agent.

Graph topology:
    START → [agent] ⟷ [tools] → END

Uses a ReAct (Tool-Calling) architecture to handle varying complexities 
naturally without any if-else branching or hard-coded routing paths. The LLM 
dynamically calls tools and formats responses based on the query context.
"""

import os
import re
import time
import logging
import operator
from typing import TypedDict, Annotated

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from tools.vector_search import vector_search
from tools.keyword_search import keyword_search

load_dotenv()
logger = logging.getLogger(__name__)

# ── Helper: Check if running inside Streamlit ──────────────────────────────────
def _in_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False

# ── Streamlit secret injection ─────────────────────────────────────────────────
try:
    if _in_streamlit():
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
                if _in_streamlit():
                    import streamlit as st
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
    messages: Annotated[list, operator.add]
    steps: list              # reasoning trace consumed by app.py UI
    error: str | None

# ── Tools ──────────────────────────────────────────────────────────────────────
@tool
def search_cases(query: str) -> str:
    """Search Indian court judgments using semantic and keyword search. Use this for general queries or finding supporting precedents."""
    try:
        top_k = int(os.getenv("TOP_K_RETRIEVAL", "8"))
        vec = vector_search(query, top_k=top_k)
        kw = keyword_search(query, top_k=top_k)
        
        all_hits = {r["doc_id"]: r for r in vec + kw if r}
        return _format_hits(list(all_hits.values()), "hybrid search")
    except Exception as e:
        return f"Search error: {e}"

@tool
def search_adverse_cases(query: str) -> str:
    """Search specifically for adverse/unfavourable judgments (e.g. insurer absolved, policy voided, claim dismissed)."""
    try:
        adv_query = f"{query} insurer absolved policy void no liability fundamental breach dismissed"
        results = keyword_search(adv_query, top_k=5)
        return _format_hits(results, "adversarial search")
    except Exception as e:
        return f"Adverse search error: {e}"

def _format_hits(results: list, source: str) -> str:
    if not results:
        return f"No results from {source}."
    lines = [f"Results from {source} ({len(results)} hits):\n"]
    for i, r in enumerate(results, 1):
        score_info = f"Score: {r.get('score', 0.0):.3f}"
        lines.append(
            f"{i}. [{r['doc_id']}] ({score_info}) {r.get('case_name','')[:55]} "
            f"({r.get('court','')[:30]}, {r.get('year','')}) | {r.get('text','')[:300]}…"
        )
    return "\n".join(lines)


# ── Graph Nodes ────────────────────────────────────────────────────────────────
def agent_node(state: AgentState) -> dict:
    llm = build_llm()
    tools = [search_cases, search_adverse_cases]
    llm_with_tools = llm.bind_tools(tools)
    
    sys_msg = SystemMessage(content="""You are Lexi, an AI legal research assistant specializing in Indian court judgments.
You must handle user queries flexibly and naturally:
- If the user asks a simple or general question (e.g. "Which cases involve commercial vehicles?"), use search_cases and answer directly.
- If the user asks for deep legal research or precedents, you must dynamically research both sides. Use search_cases for supporting precedents, and ALWAYS use search_adverse_cases to check for opposing precedents. Then output your final answer formatted strictly with three sections:
  ### Supporting Precedents
  ### Adverse Precedents
  ### Strategy Recommendation

Always cite documents exactly as [DOC_XXX].""")
    
    try:
        response = _invoke_with_retry(llm_with_tools, [sys_msg] + state["messages"])
        
        steps = []
        if response.tool_calls:
            for tc in response.tool_calls:
                steps.append({"type": "tool_call", "tool_name": tc["name"], "content": str(tc["args"])})
        else:
            steps.append({"type": "thought", "content": "Synthesising final legal response."})
            
        return {"messages": [response], "steps": steps}
    except Exception as exc:
        return {"error": str(exc), "steps": [{"type": "error", "content": str(exc)}]}

def tools_node(state: AgentState) -> dict:
    last_msg = state["messages"][-1]
    tool_msgs = []
    steps = []
    
    tools_map = {"search_cases": search_cases, "search_adverse_cases": search_adverse_cases}
    
    for tc in last_msg.tool_calls:
        tool_fn = tools_map.get(tc["name"])
        if tool_fn:
            result = tool_fn.invoke(tc["args"])
            tool_msgs.append(ToolMessage(content=result, tool_call_id=tc["id"]))
            steps.append({"type": "tool_result", "tool_name": tc["name"], "content": result[:500] + "..."})
            
    return {"messages": tool_msgs, "steps": steps}

def should_continue(state: AgentState) -> str:
    if state.get("error"):
        return END
    last_msg = state["messages"][-1]
    if getattr(last_msg, "tool_calls", None):
        return "tools"
    return END

def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue)
    graph.add_edge("tools", "agent")
    return graph.compile()


# ── Public API ─────────────────────────────────────────────────────────────────
_graph = None


def run_agent(query: str) -> dict:
    """Run the LangGraph agent. Returns {answer, steps, error}."""
    global _graph
    if _graph is None:
        _graph = build_graph()

    initial_state: AgentState = {
        "messages": [HumanMessage(content=query)],
        "steps": [],
        "error": None,
    }

    try:
        final_state = _graph.invoke(initial_state)
        if final_state.get("error"):
            return {"answer": f"Error: {final_state['error']}", "steps": final_state["steps"], "error": final_state["error"]}
            
        answer = final_state["messages"][-1].content
        return {
            "answer": answer,
            "steps": final_state["steps"],
            "error": None,
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