"""Calculator tool for basic arithmetic."""

from typing import Dict, Any

from tools.base import Tool


class Calculator(Tool):
    name = "calculator"
    description = "Evaluate simple arithmetic expressions using Python eval (safe subset)."

    def run(self, tool_input: Dict[str, Any], trace_id: str) -> Dict[str, Any]:
        expr = tool_input.get("expression", "")
        try:
            # Basic safety: allow digits and operators only.
            allowed = set("0123456789+-*/(). ")
            if not set(expr) <= allowed:
                raise ValueError("Invalid characters in expression.")
            value = eval(expr)  # noqa: S307 - controlled input above
            return {"status": "ok", "output": str(value)}
        except Exception as exc:  # pylint: disable=broad-except
            return {"status": "error", "output": str(exc)}
