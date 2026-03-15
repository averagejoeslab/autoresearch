"""
Result Aggregation primitive.

Merges outputs from multiple sub-agents into a single coherent answer.
"""

from __future__ import annotations

from typing import Any


class ResultAggregator:
    """Merge sub-agent results into one answer.

    Baseline strategy
    -----------------
    Concatenate all results in order, prefixed with subtask labels.
    No deduplication or contradiction handling.
    """

    name: str = "result_aggregator"

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def aggregate(
        self,
        results: list[dict[str, Any]],
        original_task: str,
    ) -> str:
        """Produce a single merged answer from sub-agent outputs.

        Parameters
        ----------
        results : list[dict]
            Each dict has:
              - "agent"      : str
              - "subtask"    : str
              - "output"     : str
              - "confidence" : float  (0.0 - 1.0)
        original_task : str
            The original high-level task.

        Returns
        -------
        str
            Merged answer text.
        """
        if not results:
            return ""

        parts: list[str] = []
        for r in results:
            subtask = r.get("subtask", "unknown")
            output = r.get("output", "")
            parts.append(f"[{subtask}]: {output}")

        return "\n\n".join(parts)
