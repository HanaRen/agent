from tools.base import ToolRegistry
from tools.calculator import Calculator
from tools.search import Search

registry = ToolRegistry()
registry.register(Calculator())
registry.register(Search())

__all__ = ["registry", "Calculator", "Search"]
