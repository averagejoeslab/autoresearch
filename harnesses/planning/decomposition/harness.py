"""
Task Decomposition primitive.

Breaks complex tasks into ordered, dependency-aware execution steps.
"""

from __future__ import annotations

import re
from typing import Any


class TaskDecomposer:
    """Decomposes a high-level task into executable steps with dependencies."""

    name = "task_decomposer"

    # ── keyword → steps mapping for the baseline ──────────────────────
    # The baseline is intentionally simple: 3 generic steps for any task.
    # An improved version should produce task-specific, fine-grained plans.

    def decompose(self, task: str, context: dict[str, Any] | None = None) -> list[dict]:
        """Break *task* into ordered steps.

        Parameters
        ----------
        task : str
            Natural-language description of the task.
        context : dict, optional
            Extra information (domain, constraints, resources, etc.).

        Returns
        -------
        list[dict]
            Each dict has:
              - "action"      : short imperative label
              - "description" : one-sentence explanation
              - "depends_on"  : list[int] – indices of prerequisite steps
        """
        context = context or {}

        # ── Baseline strategy: 3 generic steps ───────────────────────
        steps = [
            {
                "action": "prepare",
                "description": f"Gather requirements and resources for: {task}",
                "depends_on": [],
            },
            {
                "action": "execute",
                "description": f"Carry out the core work for: {task}",
                "depends_on": [0],
            },
            {
                "action": "verify",
                "description": f"Validate the results of: {task}",
                "depends_on": [1],
            },
        ]
        return steps
