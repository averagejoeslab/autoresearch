"""
Reliability metrics for measuring agent consistency.

Measures pass@k, pass^k, and related reliability statistics
from repeated trial data.

Exports:
    ReliabilityMeasure  -- compute reliability metrics from trial data.
"""

from __future__ import annotations

import math
from typing import Any


class ReliabilityMeasure:
    """Compute reliability metrics from multi-trial agent results.

    Baseline strategy
    -----------------
    pass@1: just use single-trial results, ignoring consistency.
    """

    name: str = "reliability_measure"

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def measure(self, results: list[list[bool]]) -> dict[str, Any]:
        """Compute reliability metrics from trial data.

        Parameters
        ----------
        results : list[list[bool]]
            Outer list = tasks, inner list = trials per task.
            Each bool indicates pass (True) or fail (False).

        Returns
        -------
        dict with keys:
            "pass_at_1"        : float  -- average single-trial pass rate
            "pass_at_k"        : dict[int, float]  -- pass@k for k=1,3,5,10
            "consistency"      : float  -- fraction of tasks that are fully
                                           consistent (all pass or all fail)
            "flaky_rate"       : float  -- fraction of tasks with mixed results
            "per_task"         : list[dict]  -- per-task metrics
            "aggregate_reliability" : float  -- overall reliability score
        """
        if not results:
            return {
                "pass_at_1": 0.0,
                "pass_at_k": {},
                "consistency": 0.0,
                "flaky_rate": 0.0,
                "per_task": [],
                "aggregate_reliability": 0.0,
            }

        per_task = []
        for trials in results:
            if not trials:
                per_task.append({
                    "pass_rate": 0.0,
                    "n_trials": 0,
                    "consistent": True,
                    "flaky": False,
                })
                continue

            pass_rate = sum(trials) / len(trials)
            all_same = all(t == trials[0] for t in trials)
            per_task.append({
                "pass_rate": round(pass_rate, 4),
                "n_trials": len(trials),
                "consistent": all_same,
                "flaky": not all_same,
            })

        # Baseline: pass@1 is just the average pass rate
        pass_at_1 = sum(t["pass_rate"] for t in per_task) / len(per_task)

        # pass@k: probability of at least one pass in k trials
        # For baseline, we just report pass@1 for all k values
        pass_at_k = {}
        for k in [1, 3, 5, 10]:
            pass_at_k[k] = round(pass_at_1, 4)

        consistency = sum(1 for t in per_task if t["consistent"]) / len(per_task)
        flaky_rate = sum(1 for t in per_task if t["flaky"]) / len(per_task)

        # Aggregate reliability = pass@1 (baseline ignores variance)
        aggregate = round(pass_at_1, 4)

        return {
            "pass_at_1": round(pass_at_1, 4),
            "pass_at_k": pass_at_k,
            "consistency": round(consistency, 4),
            "flaky_rate": round(flaky_rate, 4),
            "per_task": per_task,
            "aggregate_reliability": aggregate,
        }

    def recommend_k(self, task_results: list[list[bool]]) -> int:
        """Suggest optimal number of trials k per task.

        Parameters
        ----------
        task_results : list[list[bool]]
            Outer list = tasks, inner list = trials.

        Returns
        -------
        int
            Recommended k value (1, 3, 5, or 10).

        Baseline: always recommend k=1 (ignore consistency data).
        """
        return 1
