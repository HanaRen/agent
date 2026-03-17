"""Mock search tool returning canned results."""

from typing import Dict, Any, List

from tools.base import Tool


MOCK_RESULTS = [
    {"title": "Result 1", "url": "https://example.com/1", "snippet": "Example snippet 1"},
    {"title": "Result 2", "url": "https://example.com/2", "snippet": "Example snippet 2"},
]


class Search(Tool):
    name = "search"
    description = "Mock web search returning canned results."

    def run(self, tool_input: Dict[str, Any], trace_id: str) -> Dict[str, Any]:
        query = tool_input.get("query", "")
        hits: List[dict] = MOCK_RESULTS
        return {"status": "ok", "output": f"Search results for '{query}': {hits}"}
