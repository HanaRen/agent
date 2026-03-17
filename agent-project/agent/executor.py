"""Executor runs tool calls and streams results back to the agent."""

from tools import registry
from config import Settings


class Executor:
    def __init__(self, settings: Settings, logger):
        self.logger = logger
        self.max_steps = settings.max_steps
        self.registry = registry

    def execute(self, plan, trace_id: str):
        if plan.tool_name not in self.registry:
            return {"status": "error", "output": f"Unknown tool {plan.tool_name}"}

        tool = self.registry[plan.tool_name]
        result = tool.run(plan.tool_input or {}, trace_id=trace_id)
        self.logger.info(
            "tool.run",
            extra={"trace_id": trace_id, "tool": plan.tool_name, "result": result},
        )
        return result
