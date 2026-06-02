"""
ReAct agent for legal precedent research.

The agent uses the Reason + Act loop to dynamically decide how many retrieval
steps are needed. Simple queries get a direct answer; research queries trigger
multiple tool calls followed by a structured Supporting / Adverse / Strategy memo.
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
    for key in ["GROQ_API_KEY", "GOOGLE_API_KEY", "LLM_PROVIDER", "LLM_MODEL", "CHROMA_PERSIST_DIR", "TOP_K_RETRIEVAL"]:
        if key in st.secrets and not os.environ.get(key):
            os.environ[key] = st.secrets[key]
except Exception:
    pass  # Running locally without Streamlit context


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


def _parse_retry_delay(error_msg: str, default: float = 10.0) -> float:
    """
    Extract the suggested retry delay from a rate-limit error message.
    APIs often embed 'retry after X seconds' or 'retryDelay: Xs' in the error.
    Falls back to the provided default if nothing is found.
    """
    patterns = [
        r"retry[_ ](?:after|in|delay)[:\s]+(\d+(?:\.\d+)?)\s*s",
        r"please retry in (\d+(?:\.\d+)?)s",
        r"\"retryDelay\":\s*\"(\d+(?:\.\d+)?)s\"",
        r"(\d+(?:\.\d+)?)\s*seconds",
    ]
    for pat in patterns:
        m = re.search(pat, str(error_msg), re.IGNORECASE)
        if m:
            return min(float(m.group(1)), 60.0)  # cap at 60s
    return default


def _is_rate_limit(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(k in msg for k in ["429", "resource_exhausted", "rate limit", "too many requests", "quota", "503"])


def _run_with_retry(executor: AgentExecutor, query: str, callbacks: list, max_retries: int = 4) -> dict:
    """
    Run the agent with exponential backoff on rate-limit errors.
    Waits the server-suggested delay when available, otherwise uses
    exponential backoff: 5s → 15s → 30s → 60s.
    """
    backoff = [5, 15, 30, 60]

    for attempt in range(max_retries + 1):
        try:
            return executor.invoke({"input": query}, config={"callbacks": callbacks})
        except Exception as exc:
            if not _is_rate_limit(exc):
                raise  # non-rate-limit errors propagate immediately

            if attempt == max_retries:
                raise

            wait = _parse_retry_delay(str(exc), default=backoff[min(attempt, len(backoff) - 1)])
            logger.warning(f"Rate limit hit (attempt {attempt + 1}/{max_retries}). Waiting {wait:.0f}s...")

            # Surface the wait to the Streamlit UI if available
            try:
                st.toast(f"⏳ Rate limit — waiting {wait:.0f}s before retry {attempt + 1}/{max_retries}...", icon="⏳")
            except Exception:
                pass

            time.sleep(wait)


def build_llm():
    """
    Build the LLM client from environment config.
    Supported providers: groq (default), google, openai
    """
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    model = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=model,
            temperature=0.1,
            max_tokens=4096,
            groq_api_key=os.getenv("GROQ_API_KEY"),
        )
    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model, temperature=0.1, max_tokens=4096)
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, temperature=0.1, max_tokens=4096)
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
        max_iterations=10,
        early_stopping_method="generate",
        handle_parsing_errors=True,
        return_intermediate_steps=True,
    )


_callback = StepCaptureCallback()


def run_agent(query: str) -> dict:
    """Run the agent with automatic rate-limit retry. Returns answer + reasoning steps."""
    _callback.reset()
    try:
        result = _run_with_retry(build_executor(), query, callbacks=[_callback])
        return {"answer": result.get("output", ""), "steps": _callback.steps, "error": None}
    except Exception as exc:
        msg = str(exc)
        if _is_rate_limit(exc):
            wait = _parse_retry_delay(msg)
            friendly = (
                f"⏳ The LLM API is rate-limited. "
                f"The server suggests waiting {wait:.0f}s. "
                f"Please try again in a moment."
            )
            return {"answer": friendly, "steps": _callback.steps, "error": friendly}
        return {"answer": msg, "steps": _callback.steps, "error": msg}
