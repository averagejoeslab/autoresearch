"""
Strategy Selection primitive.

Chooses the best planning strategy for a task based on its features.
"""

from __future__ import annotations

from typing import Any


VALID_STRATEGIES = ["linear", "tree", "iterative", "decompose"]


class StrategySelector:
    """Selects a planning strategy based on task text and features."""

    name = "strategy_selector"

    def select(self, task: str, task_features: dict[str, Any] | None = None) -> str:
        """Return the best strategy name for the given task.

        Parameters
        ----------
        task : str
            Natural-language description of the task.
        task_features : dict, optional
            Keys:
              - complexity      : int 1-5
              - requires_search : bool
              - has_dependencies: bool
              - is_creative     : bool
              - time_pressure   : bool

        Returns
        -------
        str
            One of: "linear", "tree", "iterative", "decompose".
        """
        # ── Baseline: always return "linear" ─────────────────────────
        return "linear"
