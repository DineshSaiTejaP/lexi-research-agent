"""Captures intermediate ReAct agent steps for display in Streamlit."""

import sys
from typing import Any, Union
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.agents import AgentAction, AgentFinish


class StepCaptureCallback(BaseCallbackHandler):
    """
    Records every thought, tool call, and tool result as the agent runs,
    so the UI can show the full reasoning chain — not just the final answer.
    """

    def __init__(self):
        super().__init__()
        self.steps: list[dict] = []

    def reset(self):
        self.steps = []
        print("\n\n" + "="*50 + "\n[NEW QUERY STARTED]\n" + "="*50, flush=True)

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> None:
        if action.log:
            self.steps.append({"type": "thought", "content": action.log.strip()})
            print(f"\n[THOUGHT]\n{action.log.strip()}\n", flush=True)
            
        self.steps.append({
            "type": "tool_call",
            "tool_name": action.tool,
            "content": action.tool_input,
        })
        print(f"\n[TOOL CALL] {action.tool}\nInput: {action.tool_input}\n", flush=True)

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        content = output[:2000]
        self.steps.append({"type": "tool_result", "content": content})
        print(f"\n[TOOL RESULT]\n{content}...\n", flush=True)

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> None:
        log = finish.log.strip() if finish.log else ""
        final = finish.return_values.get("output", "")
        if log and log != final:
            self.steps.append({"type": "thought", "content": log})
            print(f"\n[THOUGHT]\n{log}\n", flush=True)
            
        print(f"\n[AGENT FINISH]\n{final}\n" + "="*50 + "\n", flush=True)

    def on_chain_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> None:
        self.steps.append({"type": "error", "content": str(error)})
        print(f"\n[CHAIN ERROR]\n{error}\n", flush=True)

    def on_tool_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> None:
        self.steps.append({"type": "error", "content": f"Tool error: {error}"})
        print(f"\n[TOOL ERROR]\n{error}\n", flush=True)

