"""Captures intermediate ReAct agent steps for display in Streamlit."""

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

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> None:
        if action.log:
            self.steps.append({"type": "thought", "content": action.log.strip()})
        self.steps.append({
            "type": "tool_call",
            "tool_name": action.tool,
            "content": action.tool_input,
        })

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        self.steps.append({"type": "tool_result", "content": output[:2000]})

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> None:
        log = finish.log.strip() if finish.log else ""
        final = finish.return_values.get("output", "")
        if log and log != final:
            self.steps.append({"type": "thought", "content": log})

    def on_chain_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> None:
        self.steps.append({"type": "error", "content": str(error)})

    def on_tool_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> None:
        self.steps.append({"type": "error", "content": f"Tool error: {error}"})
