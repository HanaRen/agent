"""Lightweight LLM client placeholder."""

from typing import List

from llm.schemas import Message
from config import Settings


class LLMClient:
    def __init__(self, settings: Settings, logger):
        self.settings = settings
        self.logger = logger

    def chat(self, prompt: str, trace_id: str) -> dict:
        # Placeholder: echo back a dummy plan that calls calculator if "calc" present
        if "calc" in prompt.lower():
            return {"action": "tool", "tool": "calculator", "tool_input": {"expression": "1+1"}}
        return {"action": "final", "output": "Default response from LLM stub."}
