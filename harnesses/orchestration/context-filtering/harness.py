"""
Context Filter primitive.

Selects which context items to include when dispatching a subtask
to a sub-agent, respecting a budget constraint.
"""

from __future__ import annotations

from typing import Any


class ContextFilter:
    """Select relevant context for a subtask within a budget.

    Baseline strategy
    -----------------
    Take the *most recent* items that fit the budget, regardless of
    content relevance.  This is simple and preserves recency but
    ignores topical relevance entirely.
    """

    name: str = "context_filter"

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def filter_for_subtask(
        self,
        full_context: list[dict[str, Any]],
        subtask: str,
        budget: int,
    ) -> list[dict[str, Any]]:
        """Return at most *budget* context items relevant to *subtask*.

        Parameters
        ----------
        full_context : list[dict]
            Each dict has at least:
              - "id"      : str | int
              - "content" : str
              - "type"    : str  (e.g. "fact", "instruction", "observation")
              - "timestamp" : float | int  (higher = more recent)
        subtask : str
            Description of the subtask the sub-agent will perform.
        budget : int
            Maximum number of items to return.

        Returns
        -------
        list[dict]
            Subset of *full_context* (order preserved).
        """
        if not full_context:
            return []

        # Baseline: sort by timestamp descending, take top `budget`
        sorted_items = sorted(
            full_context,
            key=lambda x: x.get("timestamp", 0),
            reverse=True,
        )
        selected_ids = {item["id"] for item in sorted_items[:budget]}

        # Return in original order
        return [item for item in full_context if item["id"] in selected_ids]
