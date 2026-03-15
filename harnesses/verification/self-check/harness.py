"""
Self-Check primitive.

Assesses whether an agent's output correctly addresses a task,
returning a judgment, confidence score, and list of detected issues.
"""

from __future__ import annotations

from typing import Any


class SelfChecker:
    """Evaluates an output against its originating task."""

    name = "self_checker"

    def check(self, output: str, task: str) -> dict[str, Any]:
        """Assess *output* quality relative to *task*.

        Parameters
        ----------
        output : str
            The generated output to evaluate.
        task : str
            The original task description.

        Returns
        -------
        dict
            - "is_correct" : bool
            - "confidence" : float in [0, 1]
            - "issues"     : list[str]
        """
        # ── Baseline: always optimistic, no issue detection ──────────
        return {
            "is_correct": True,
            "confidence": 0.5,
            "issues": [],
        }
