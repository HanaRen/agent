"""Shared schemas for LLM messages."""

from dataclasses import dataclass
from typing import Literal


Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class Message:
    role: Role
    content: str
