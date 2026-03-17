"""Prompt templates for planner and executor."""

from typing import List


def build_planner_prompt(messages: List[dict], tools) -> str:
    tool_list = ", ".join(tools)
    return (
        "You are a helpful suggestion agent. "
        "Decide the next action. Tools available: "
        f"{tool_list}. "
        "Respond with JSON: {action: 'tool'|'final', tool: <name>, tool_input: {}, output: ''}"
    )
