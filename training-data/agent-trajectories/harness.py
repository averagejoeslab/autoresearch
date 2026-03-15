"""
Agent trajectory curation for training data.

Scores and selects high-quality agent execution traces for fine-tuning.

Exports:
    TrajectoryCurator  -- score and select trajectories.
"""

from __future__ import annotations

from typing import Any


class TrajectoryCurator:
    """Score and select agent trajectories for training data curation.

    Baseline strategy
    -----------------
    Score by inverse step count (fewer steps = better quality).
    Select top-k by score within the budget.
    """

    name: str = "trajectory_curator"

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def score_trajectory(self, trajectory: list[dict[str, Any]]) -> float:
        """Rate the quality of a single agent trajectory.

        Parameters
        ----------
        trajectory : list[dict]
            Sequence of steps. Each step dict has at least:
              - "action"  : str  -- what the agent did
              - "result"  : str  -- outcome of the action
              - "success" : bool -- whether this step succeeded

        Returns
        -------
        float
            Quality score in [0.0, 1.0]. Higher = better trajectory.
        """
        if not trajectory:
            return 0.0

        n_steps = len(trajectory)

        # Baseline: inverse step count
        # 1 step -> 1.0, 2 steps -> 0.5, 10 steps -> 0.1, etc.
        score = 1.0 / n_steps

        return round(min(1.0, max(0.0, score)), 4)

    def select_trajectories(
        self,
        trajectories: list[list[dict[str, Any]]],
        budget: int,
    ) -> list[int]:
        """Select the best N trajectories within a budget.

        Parameters
        ----------
        trajectories : list[list[dict]]
            Pool of trajectories to select from.
        budget : int
            Maximum number of trajectories to select.

        Returns
        -------
        list[int]
            Indices of selected trajectories (into the input list).
        """
        if not trajectories:
            return []

        budget = min(budget, len(trajectories))

        # Score each trajectory
        scored = [
            (i, self.score_trajectory(t)) for i, t in enumerate(trajectories)
        ]

        # Sort by score descending, take top budget
        scored.sort(key=lambda x: -x[1])
        return [i for i, _ in scored[:budget]]
