"""Project-wide settings."""

from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class Settings:
    model: str = "stub-llm"
    max_steps: int = 6
    tool_registry: Dict[str, Any] = field(default_factory=dict)


# Runtime singleton; tools register separately
settings = Settings()
