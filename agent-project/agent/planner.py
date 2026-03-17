"""Planner decides the next action for the agent."""

from dataclasses import dataclass
from typing import Any, Dict

from agent.prompt import build_planner_prompt
from llm.client import LLMClient
from config import Settings


@dataclass
class Plan:
    action: str  # "tool" or "final"
    tool_name: str | None = None
    tool_input: Dict[str, Any] | None = None
    output: str | None = None
    thoughts: str | None = None


class Planner:
    def __init__(self, settings: Settings, logger):
        self.llm = LLMClient(settings=settings, logger=logger)
        self.logger = logger
        self.available_tools = settings.tool_registry.keys()

    def decide(self, memory, trace_id: str) -> Plan:
        prompt = build_planner_prompt(memory.messages, self.available_tools)
        response = self.llm.chat(prompt, trace_id=trace_id)
        return self._parse_response(response)

    def _parse_response(self, response: dict) -> Plan:
        # Placeholder parser, to be replaced with a robust schema-based parser.
        if response.get("action") == "final":
            return Plan(action="final", output=response.get("output", ""))
        return Plan(
            action="tool",
            tool_name=response.get("tool"),
            tool_input=response.get("tool_input", {}),
            thoughts=response.get("thoughts"),
        )
