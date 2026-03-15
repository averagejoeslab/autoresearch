"""
Delegation Decider primitive.

Decides when a task should be delegated to sub-agents and how to
assign subtasks among available agents based on their capabilities.
"""

from __future__ import annotations

from typing import Any


class DelegationDecider:
    """Decide whether to delegate and plan sub-agent assignments.

    Baseline strategy
    -----------------
    * Delegate if task_features["complexity"] > 3.
    * Assign subtasks to the first available agent that is not already
      assigned (round-robin fallback).
    """

    name: str = "delegation_decider"

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def should_delegate(
        self,
        task: str,
        task_features: dict[str, Any],
    ) -> bool:
        """Return True when *task* should be split across sub-agents.

        Parameters
        ----------
        task : str
            Natural-language description of the task.
        task_features : dict
            Structured features:
              - complexity     : int 1-5
              - scope          : str ("narrow", "medium", "broad")
              - requires_specialization : bool
              - estimated_time : float (minutes)
        """
        complexity = task_features.get("complexity", 1)
        return complexity > 3

    def plan_delegation(
        self,
        task: str,
        available_agents: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Return a list of subtask assignments.

        Parameters
        ----------
        task : str
            The high-level task to break down.
        available_agents : list[dict]
            Each dict has at least:
              - "name" : str
              - "capabilities" : list[str]

        Returns
        -------
        list[dict]
            Each dict has:
              - "subtask"    : str  – description of the sub-work
              - "agent"      : str  – name of the assigned agent
              - "priority"   : int  – 1 = highest
              - "depends_on" : list[str] – subtask descriptions this depends on
        """
        if not available_agents:
            return []

        # Baseline: create a single subtask assigned to the first agent
        return [
            {
                "subtask": task,
                "agent": available_agents[0]["name"],
                "priority": 1,
                "depends_on": [],
            }
        ]
