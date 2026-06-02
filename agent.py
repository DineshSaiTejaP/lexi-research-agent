"""
ReAct agent for legal precedent research.

The agent uses the Reason + Act loop to dynamically decide how many retrieval
steps are needed. Simple queries get a direct answer; research queries trigger
multiple tool calls followed by a structured Supporting / Adverse / Strategy memo.

Rate limit strategy (Groq free tier: 6,000 TPM / 30 RPM / 14,400 RPD):
- max_tokens=1500 per call keeps each request well under the per-minute token cap
- min 2s gap enforced between consecutive LLM calls
- Up to 4 retries with server-suggested wait on 429/503
- Falls back gracefully with a user-friendly message if all retries fail
"""

import os
import re
import time
import logging

from dotenv import load_dotenv
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate

from callbacks import StepCaptureCallback
from tools import ALL_TOOLS

load_dotenv()
logger = logging.getLogger(__name__)

# On Streamlit Cloud, secrets come from st.secrets rather than .env
try:
    import streamlit as st
    for key in ["GROQ_API_KEY", "GOOGLE_API_KEY", "LLM_PROVIDER", "LLM_MODEL",
                "CHROMA_PERSIST_DIR", "TOP_K_RETRIEVAL"]:
        if key in st.secrets and not os.environ.get(key):
            os.environ[key] = str(st.secrets[key])
except Exception:
    pass


# ── Rate limit helpers ────────────────────────────────────────────────────────

_last_call_at: float = 0.0
MIN_CALL_GAP = 2.0  # seconds between LLM calls — keeps us under 30 RPM


def _throttle():
    """Block until enough time has passed since the last LLM call."""
    global _last_call_at
    elapsed = time.time() - _last_call_at
    if elapsed < MIN_CALL_GAP:
        time.sleep(MIN_CALL_GAP - elapsed)
    _last_call_at = time.time()


def _parse_retry_delay(error_msg: str, default: float = 10.0) -> float:
    """Extract suggested retry delay (seconds) from an API error message."""
    patterns = [
        r"retry[_ ](?:after|in|delay)[:\s]+(\d+(?:\.\d+)?)\s*s",
        r"please retry in (\d+(?:\.\d+)?)s",
        r"\"retryDelay\":\s*\"(\d+(?:\.\d+)?)s\"",
        r"(\d+(?:\.\d+)?)\s*seconds",
    ]
    for pat in patterns:
        m = re.search(pat, str(error_msg), re.IGNORECASE)
        if m:
            return min(float(m.group(1)) + 2, 90.0)  # add 2s buffer, cap at 90s
    return default


def _is_rate_limit(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(k in msg for k in ["429", "resource_exhausted", "rate limit",
                                   "too many requests", "quota", "503",
                                   "service unavailable"])


def _run_with_retry(executor: AgentExecutor, query: str, callbacks: list,
                    max_retries: int = 4) -> dict:
    """
    Invoke the agent with automatic retry on rate-limit errors.
    Respects the server's own suggested retry delay when provided.
    Falls back to exponential backoff: 10s → 20s → 40s → 60s.
    """
    backoff = [10, 20, 40, 60]

    for attempt in range(max_retries + 1):
        _throttle()
        try:
            return executor.invoke({"input": query}, config={"callbacks": callbacks})
        except Exception as exc:
            if not _is_rate_limit(exc):
                raise

            if attempt == max_retries:
                raise

            wait = _parse_retry_delay(str(exc), default=backoff[min(attempt, len(backoff) - 1)])
            logger.warning(f"Rate limit (attempt {attempt + 1}/{max_retries}) — waiting {wait:.0f}s")

            try:
                st.toast(f"⏳ Rate limit hit — waiting {wait:.0f}s then retrying ({attempt + 1}/{max_retries})…")
            except Exception:
                pass

            time.sleep(wait)


# ── Prompt ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are Lexi, an AI legal research assistant specialising in Indian court judgments.
You have access to a corpus of court judgments and three search tools.

Tools available:
{tools}

## Deciding how to respond

For **general or factual queries** (e.g. "Which judgments involve trucks?"):
- Run 1–2 tool calls, then give a clear, direct answer.

For **precedent research queries** (e.g. "Find supporting precedents for our client"):
- Use both vector_search AND keyword_search to ensure broad coverage.
- Run additional calls to refine if initial results are thin.
- Always look for BOTH supporting AND adverse precedents — one-sided research is dangerous.
- Structure your final answer as three sections:

  ### Supporting Precedents
  For each: [DOC_XXX] Case Name (Court, Year) | Legal principle | Why it helps

  ### Adverse Precedents
  For each: [DOC_XXX] Case Name (Court, Year) | Risk: HIGH/MEDIUM/LOW | How to counter

  ### Strategy Recommendation
  Priority arguments, realistic compensation range, key risks, next steps.

## Rules
- Only cite documents that were actually returned by your tools. Never fabricate cases.
- Always include [DOC_XXX] identifiers so citations can be verified.
- If retrieval returns nothing useful, say so honestly rather than guessing.
- Keep responses focused — you have a token budget, so be precise not verbose.

## Format
Thought: <your reasoning>
Action: <one of: {tool_names}>
Action Input: <input to the tool>
Observation: <tool result>
... repeat as needed ...
Thought: I have enough information to answer.
Final Answer: <your complete response>

Question: {input}
{agent_scratchpad}\
"""


# ── LLM factory ───────────────────────────────────────────────────────────────

def build_llm():
    """
    Build the LLM from env config.
    max_tokens is kept conservative (1500) to stay within Groq's 6,000 TPM free limit.
    """
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


def build_executor() -> AgentExecutor:
    llm = build_llm()

    prompt = PromptTemplate(
        template=SYSTEM_PROMPT,
        input_variables=["input", "agent_scratchpad"],
        partial_variables={
            "tools": "\n".join(f"- {t.name}: {t.description}" for t in ALL_TOOLS),
            "tool_names": ", ".join(t.name for t in ALL_TOOLS),
        },
    )

    agent = create_react_agent(llm=llm, tools=ALL_TOOLS, prompt=prompt)
    return AgentExecutor(
        agent=agent,
        tools=ALL_TOOLS,
        verbose=True,
        max_iterations=6,           # 6 steps × ~1500 tokens ≈ 9,000 tokens total, well under daily budget
        early_stopping_method="generate",
        handle_parsing_errors=True,
        return_intermediate_steps=True,
    )


# ── Public API ────────────────────────────────────────────────────────────────

_callback = StepCaptureCallback()


def run_agent(query: str) -> dict:
    """Run the agent with rate-limit retry. Returns answer + reasoning steps."""
    _callback.reset()
    try:
        result = _run_with_retry(build_executor(), query, callbacks=[_callback])
        return {"answer": result.get("output", ""), "steps": _callback.steps, "error": None}
    except Exception as exc:
        msg = str(exc)
        if _is_rate_limit(exc):
            wait = _parse_retry_delay(msg, default=30)
            friendly = (
                f"⏳ **Rate limit reached after all retries.**\n\n"
                f"The API suggests waiting **{wait:.0f} seconds** before trying again.\n"
                f"Please wait a moment and resubmit your query."
            )
            return {"answer": friendly, "steps": _callback.steps, "error": friendly}
        return {"answer": f"Error: {msg}", "steps": _callback.steps, "error": msg}
