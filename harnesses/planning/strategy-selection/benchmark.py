#!/usr/bin/env python3
"""
Benchmark: Strategy Selection

30 tasks with labelled optimal strategies.  Measures:
  - classification accuracy
  - macro F1 across the 4 strategy classes

Fitness = 0.7 * accuracy + 0.3 * macro_f1
Prints  : score: X.XXXXXX
"""

from __future__ import annotations

import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from harness import StrategySelector  # noqa: E402

VALID_STRATEGIES = ["linear", "tree", "iterative", "decompose"]

# ─────────────────────────────────────────────────────────────────────
# 30 labelled tasks
# Each has: task, task_features, expected_strategy
# ─────────────────────────────────────────────────────────────────────
TASKS: list[dict] = [
    # ── linear (8 tasks) ─────────────────────────────────────────────
    {"task": "Rename a file in the repository",
     "features": {"complexity": 1, "requires_search": False, "has_dependencies": False, "is_creative": False, "time_pressure": False},
     "expected": "linear"},
    {"task": "Update the README with the new version number",
     "features": {"complexity": 1, "requires_search": False, "has_dependencies": False, "is_creative": False, "time_pressure": False},
     "expected": "linear"},
    {"task": "Add a print statement for debugging",
     "features": {"complexity": 1, "requires_search": False, "has_dependencies": False, "is_creative": False, "time_pressure": True},
     "expected": "linear"},
    {"task": "Change the database connection string in config",
     "features": {"complexity": 1, "requires_search": False, "has_dependencies": False, "is_creative": False, "time_pressure": False},
     "expected": "linear"},
    {"task": "Bump the package version to 2.0.0",
     "features": {"complexity": 1, "requires_search": False, "has_dependencies": False, "is_creative": False, "time_pressure": False},
     "expected": "linear"},
    {"task": "Delete an unused import",
     "features": {"complexity": 1, "requires_search": False, "has_dependencies": False, "is_creative": False, "time_pressure": True},
     "expected": "linear"},
    {"task": "Fix a typo in an error message",
     "features": {"complexity": 1, "requires_search": False, "has_dependencies": False, "is_creative": False, "time_pressure": False},
     "expected": "linear"},
    {"task": "Set an environment variable for the build",
     "features": {"complexity": 1, "requires_search": False, "has_dependencies": False, "is_creative": False, "time_pressure": False},
     "expected": "linear"},

    # ── tree (7 tasks) ───────────────────────────────────────────────
    {"task": "Find the root cause of a flaky integration test",
     "features": {"complexity": 4, "requires_search": True, "has_dependencies": False, "is_creative": False, "time_pressure": False},
     "expected": "tree"},
    {"task": "Investigate why the API returns 500 errors intermittently",
     "features": {"complexity": 4, "requires_search": True, "has_dependencies": False, "is_creative": False, "time_pressure": True},
     "expected": "tree"},
    {"task": "Explore different caching strategies for the feed",
     "features": {"complexity": 3, "requires_search": True, "has_dependencies": False, "is_creative": False, "time_pressure": False},
     "expected": "tree"},
    {"task": "Determine the best sorting algorithm for this dataset",
     "features": {"complexity": 3, "requires_search": True, "has_dependencies": False, "is_creative": False, "time_pressure": False},
     "expected": "tree"},
    {"task": "Debug a memory leak in the application",
     "features": {"complexity": 5, "requires_search": True, "has_dependencies": False, "is_creative": False, "time_pressure": True},
     "expected": "tree"},
    {"task": "Evaluate three different authentication providers",
     "features": {"complexity": 3, "requires_search": True, "has_dependencies": False, "is_creative": False, "time_pressure": False},
     "expected": "tree"},
    {"task": "Diagnose slow query performance in the analytics dashboard",
     "features": {"complexity": 4, "requires_search": True, "has_dependencies": False, "is_creative": False, "time_pressure": False},
     "expected": "tree"},

    # ── iterative (8 tasks) ──────────────────────────────────────────
    {"task": "Write a compelling product description",
     "features": {"complexity": 3, "requires_search": False, "has_dependencies": False, "is_creative": True, "time_pressure": False},
     "expected": "iterative"},
    {"task": "Refine the UI design for the onboarding flow",
     "features": {"complexity": 3, "requires_search": False, "has_dependencies": False, "is_creative": True, "time_pressure": False},
     "expected": "iterative"},
    {"task": "Optimize the landing page copy for conversions",
     "features": {"complexity": 3, "requires_search": False, "has_dependencies": False, "is_creative": True, "time_pressure": False},
     "expected": "iterative"},
    {"task": "Improve the error messages to be more user-friendly",
     "features": {"complexity": 2, "requires_search": False, "has_dependencies": False, "is_creative": True, "time_pressure": False},
     "expected": "iterative"},
    {"task": "Tune hyperparameters for the ML model",
     "features": {"complexity": 4, "requires_search": False, "has_dependencies": False, "is_creative": False, "time_pressure": False},
     "expected": "iterative"},
    {"task": "Polish the presentation slides for the board meeting",
     "features": {"complexity": 2, "requires_search": False, "has_dependencies": False, "is_creative": True, "time_pressure": False},
     "expected": "iterative"},
    {"task": "Refactor the data processing module for clarity",
     "features": {"complexity": 3, "requires_search": False, "has_dependencies": False, "is_creative": False, "time_pressure": False},
     "expected": "iterative"},
    {"task": "A/B test different email subject lines",
     "features": {"complexity": 2, "requires_search": False, "has_dependencies": False, "is_creative": True, "time_pressure": False},
     "expected": "iterative"},

    # ── decompose (7 tasks) ──────────────────────────────────────────
    {"task": "Build a full-stack e-commerce application",
     "features": {"complexity": 5, "requires_search": False, "has_dependencies": True, "is_creative": False, "time_pressure": False},
     "expected": "decompose"},
    {"task": "Migrate the entire infrastructure to Kubernetes",
     "features": {"complexity": 5, "requires_search": False, "has_dependencies": True, "is_creative": False, "time_pressure": False},
     "expected": "decompose"},
    {"task": "Implement a real-time chat system with file sharing",
     "features": {"complexity": 5, "requires_search": False, "has_dependencies": True, "is_creative": False, "time_pressure": False},
     "expected": "decompose"},
    {"task": "Set up a complete CI/CD pipeline with staging and prod",
     "features": {"complexity": 4, "requires_search": False, "has_dependencies": True, "is_creative": False, "time_pressure": False},
     "expected": "decompose"},
    {"task": "Build a data warehouse with ETL from five sources",
     "features": {"complexity": 5, "requires_search": False, "has_dependencies": True, "is_creative": False, "time_pressure": False},
     "expected": "decompose"},
    {"task": "Redesign the company website with CMS integration",
     "features": {"complexity": 4, "requires_search": False, "has_dependencies": True, "is_creative": True, "time_pressure": False},
     "expected": "decompose"},
    {"task": "Create a microservices architecture from a monolith",
     "features": {"complexity": 5, "requires_search": False, "has_dependencies": True, "is_creative": False, "time_pressure": False},
     "expected": "decompose"},
]


# ─────────────────────────────────────────────────────────────────────
# Scoring helpers
# ─────────────────────────────────────────────────────────────────────

def _macro_f1(predictions: list[str], labels: list[str], classes: list[str]) -> float:
    """Compute macro-averaged F1 across *classes*."""
    f1s: list[float] = []
    for cls in classes:
        tp = sum(1 for p, l in zip(predictions, labels) if p == cls and l == cls)
        fp = sum(1 for p, l in zip(predictions, labels) if p == cls and l != cls)
        fn = sum(1 for p, l in zip(predictions, labels) if p != cls and l == cls)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        f1s.append(f1)
    return sum(f1s) / len(f1s) if f1s else 0.0


def run_benchmark() -> float:
    selector = StrategySelector()

    predictions: list[str] = []
    labels: list[str] = []

    for t in TASKS:
        pred = selector.select(t["task"], t["features"])
        # Clamp to valid strategies
        if pred not in VALID_STRATEGIES:
            pred = "linear"  # default fallback
        predictions.append(pred)
        labels.append(t["expected"])

    accuracy = sum(1 for p, l in zip(predictions, labels) if p == l) / len(labels)
    macro_f1 = _macro_f1(predictions, labels, VALID_STRATEGIES)

    score = 0.7 * accuracy + 0.3 * macro_f1

    print(f"  accuracy  : {accuracy:.6f}")
    print(f"  macro_f1  : {macro_f1:.6f}")
    return score


if __name__ == "__main__":
    final = run_benchmark()
    print(f"score: {final:.6f}")
