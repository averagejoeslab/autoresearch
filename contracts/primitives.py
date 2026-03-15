"""
Shared interfaces for agent runtime primitives.
Every harness.py must export a class implementing the Primitive protocol.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class AgentMessage:
    """Universal message type flowing between primitives."""

    role: str  # "user", "assistant", "system", "tool_result"
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class AgentState:
    """Snapshot of the agent state at a point in time."""

    messages: list[AgentMessage] = field(default_factory=list)
    memory: dict[str, Any] = field(default_factory=dict)
    plan: list[dict[str, Any]] = field(default_factory=list)
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Primitive(Protocol):
    """Every primitive must implement this interface."""

    name: str

    def initialize(self, config: dict[str, Any]) -> None:
        """Called once at startup. Receive configuration."""
        ...

    def process(self, state: AgentState) -> AgentState:
        """Core method: receive state, return modified state."""
        ...

    def get_metrics(self) -> dict[str, float]:
        """Return current performance metrics."""
        ...
