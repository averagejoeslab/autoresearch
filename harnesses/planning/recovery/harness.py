"""
Plan Recovery primitive.

Re-plans after a step failure so the original goal can still be achieved.
"""

from __future__ import annotations

import copy
from typing import Any


class PlanRecovery:
    """Produces a revised plan that works around a failed step."""

    name = "plan_recovery"

    def recover(
        self,
        original_plan: list[dict],
        failed_step: int,
        error: str,
    ) -> list[dict]:
        """Return a revised plan that avoids *failed_step*.

        Parameters
        ----------
        original_plan : list[dict]
            Each dict has "action", "description", "depends_on".
        failed_step : int
            Index (0-based) of the step that failed.
        error : str
            Description of the failure.

        Returns
        -------
        list[dict]
            New plan with the failed step removed (and dependency
            indices remapped).
        """
        # ── Baseline: remove the failed step, keep the rest ──────────
        revised: list[dict] = []
        index_map: dict[int, int] = {}

        for i, step in enumerate(original_plan):
            if i == failed_step:
                continue
            new_index = len(revised)
            index_map[i] = new_index
            new_step = copy.deepcopy(step)
            # Remap dependency indices, dropping references to the failed step
            new_step["depends_on"] = [
                index_map[d] for d in step.get("depends_on", [])
                if d != failed_step and d in index_map
            ]
            revised.append(new_step)

        return revised
