#!/usr/bin/env python3
"""
Benchmark: Delegation Decider
==============================
25 tasks testing when to delegate and how to assign sub-agents.

Fitness = 0.5 * decision_accuracy + 0.3 * assignment_quality + 0.2 * efficiency

Run:  python benchmark.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from harness import DelegationDecider  # noqa: E402

# ── Inline test data ────────────────────────────────────────────────

AGENTS = [
    {"name": "coder", "capabilities": ["code", "debug", "refactor"]},
    {"name": "writer", "capabilities": ["writing", "editing", "summarization"]},
    {"name": "researcher", "capabilities": ["search", "analysis", "data"]},
    {"name": "planner", "capabilities": ["planning", "scheduling", "coordination"]},
    {"name": "reviewer", "capabilities": ["review", "testing", "qa"]},
]

# Each case: (task, task_features, should_delegate, expected_agents, expected_subtask_count)
# expected_agents: list of agent names that should be involved (order doesn't matter)
# expected_subtask_count: minimum number of subtasks expected
CASES: list[dict] = [
    # --- Simple tasks: should NOT delegate ---
    {
        "task": "Fix a typo in the README file",
        "features": {"complexity": 1, "scope": "narrow", "requires_specialization": False, "estimated_time": 2.0},
        "should_delegate": False,
        "expected_agents": [],
        "min_subtasks": 0,
    },
    {
        "task": "Rename a variable from 'x' to 'count'",
        "features": {"complexity": 1, "scope": "narrow", "requires_specialization": False, "estimated_time": 1.0},
        "should_delegate": False,
        "expected_agents": [],
        "min_subtasks": 0,
    },
    {
        "task": "Add a print statement for debugging",
        "features": {"complexity": 1, "scope": "narrow", "requires_specialization": False, "estimated_time": 1.0},
        "should_delegate": False,
        "expected_agents": [],
        "min_subtasks": 0,
    },
    {
        "task": "Update the version number in setup.py",
        "features": {"complexity": 1, "scope": "narrow", "requires_specialization": False, "estimated_time": 1.0},
        "should_delegate": False,
        "expected_agents": [],
        "min_subtasks": 0,
    },
    {
        "task": "Delete an unused import",
        "features": {"complexity": 1, "scope": "narrow", "requires_specialization": False, "estimated_time": 0.5},
        "should_delegate": False,
        "expected_agents": [],
        "min_subtasks": 0,
    },
    {
        "task": "Write a short comment above a function",
        "features": {"complexity": 2, "scope": "narrow", "requires_specialization": False, "estimated_time": 2.0},
        "should_delegate": False,
        "expected_agents": [],
        "min_subtasks": 0,
    },
    {
        "task": "Add a simple unit test for an add function",
        "features": {"complexity": 2, "scope": "narrow", "requires_specialization": False, "estimated_time": 5.0},
        "should_delegate": False,
        "expected_agents": [],
        "min_subtasks": 0,
    },
    {
        "task": "Format code with black",
        "features": {"complexity": 1, "scope": "medium", "requires_specialization": False, "estimated_time": 1.0},
        "should_delegate": False,
        "expected_agents": [],
        "min_subtasks": 0,
    },
    {
        "task": "Look up what the max function does in Python",
        "features": {"complexity": 1, "scope": "narrow", "requires_specialization": False, "estimated_time": 2.0},
        "should_delegate": False,
        "expected_agents": [],
        "min_subtasks": 0,
    },
    {
        "task": "Change a button color from blue to green in CSS",
        "features": {"complexity": 2, "scope": "narrow", "requires_specialization": False, "estimated_time": 3.0},
        "should_delegate": False,
        "expected_agents": [],
        "min_subtasks": 0,
    },
    # --- Medium tasks: borderline (complexity 3) should NOT delegate ---
    {
        "task": "Refactor a 50-line function into smaller helpers",
        "features": {"complexity": 3, "scope": "medium", "requires_specialization": False, "estimated_time": 20.0},
        "should_delegate": False,
        "expected_agents": [],
        "min_subtasks": 0,
    },
    {
        "task": "Write integration tests for the login flow",
        "features": {"complexity": 3, "scope": "medium", "requires_specialization": False, "estimated_time": 30.0},
        "should_delegate": False,
        "expected_agents": [],
        "min_subtasks": 0,
    },
    # --- Complex tasks: should delegate ---
    {
        "task": "Build a REST API with authentication, write documentation, and deploy",
        "features": {"complexity": 5, "scope": "broad", "requires_specialization": True, "estimated_time": 120.0},
        "should_delegate": True,
        "expected_agents": ["coder", "writer"],
        "min_subtasks": 2,
    },
    {
        "task": "Research competitor products, write analysis report, and create presentation",
        "features": {"complexity": 4, "scope": "broad", "requires_specialization": True, "estimated_time": 60.0},
        "should_delegate": True,
        "expected_agents": ["researcher", "writer"],
        "min_subtasks": 2,
    },
    {
        "task": "Implement a new feature with code review and comprehensive test suite",
        "features": {"complexity": 4, "scope": "broad", "requires_specialization": True, "estimated_time": 90.0},
        "should_delegate": True,
        "expected_agents": ["coder", "reviewer"],
        "min_subtasks": 2,
    },
    {
        "task": "Design database schema, implement data layer, write migration scripts, and test",
        "features": {"complexity": 5, "scope": "broad", "requires_specialization": True, "estimated_time": 180.0},
        "should_delegate": True,
        "expected_agents": ["coder", "reviewer"],
        "min_subtasks": 3,
    },
    {
        "task": "Analyze user feedback data, identify trends, write report, and plan fixes",
        "features": {"complexity": 4, "scope": "broad", "requires_specialization": True, "estimated_time": 90.0},
        "should_delegate": True,
        "expected_agents": ["researcher", "planner"],
        "min_subtasks": 2,
    },
    {
        "task": "Rewrite the legacy module in modern Python, add type hints, and update docs",
        "features": {"complexity": 4, "scope": "broad", "requires_specialization": False, "estimated_time": 60.0},
        "should_delegate": True,
        "expected_agents": ["coder", "writer"],
        "min_subtasks": 2,
    },
    {
        "task": "Create a full marketing website with content, SEO optimization, and QA",
        "features": {"complexity": 5, "scope": "broad", "requires_specialization": True, "estimated_time": 240.0},
        "should_delegate": True,
        "expected_agents": ["coder", "writer", "reviewer"],
        "min_subtasks": 3,
    },
    {
        "task": "Set up CI/CD pipeline, write deployment docs, and test across environments",
        "features": {"complexity": 4, "scope": "broad", "requires_specialization": True, "estimated_time": 60.0},
        "should_delegate": True,
        "expected_agents": ["coder", "writer"],
        "min_subtasks": 2,
    },
    {
        "task": "Conduct security audit of the codebase and produce remediation plan",
        "features": {"complexity": 5, "scope": "broad", "requires_specialization": True, "estimated_time": 120.0},
        "should_delegate": True,
        "expected_agents": ["researcher", "reviewer", "planner"],
        "min_subtasks": 2,
    },
    # --- Edge cases ---
    {
        "task": "Translate error messages to 5 languages",
        "features": {"complexity": 4, "scope": "broad", "requires_specialization": True, "estimated_time": 45.0},
        "should_delegate": True,
        "expected_agents": ["writer"],
        "min_subtasks": 1,
    },
    {
        "task": "",
        "features": {"complexity": 1, "scope": "narrow", "requires_specialization": False, "estimated_time": 0.0},
        "should_delegate": False,
        "expected_agents": [],
        "min_subtasks": 0,
    },
    {
        "task": "Plan and execute a multi-phase data migration with rollback strategy",
        "features": {"complexity": 5, "scope": "broad", "requires_specialization": True, "estimated_time": 300.0},
        "should_delegate": True,
        "expected_agents": ["coder", "planner", "reviewer"],
        "min_subtasks": 3,
    },
    {
        "task": "Profile performance bottlenecks, optimize critical paths, and verify improvements",
        "features": {"complexity": 4, "scope": "medium", "requires_specialization": True, "estimated_time": 90.0},
        "should_delegate": True,
        "expected_agents": ["coder", "researcher"],
        "min_subtasks": 2,
    },
]


def _evaluate_assignment_quality(
    assignments: list[dict],
    expected_agents: list[str],
    min_subtasks: int,
) -> float:
    """Score how well the delegation plan matches expectations."""
    if not expected_agents and not assignments:
        return 1.0  # correctly produced no assignments
    if not expected_agents and assignments:
        return 0.0  # should not have delegated
    if expected_agents and not assignments:
        return 0.0  # should have delegated

    score = 0.0

    # 1. Agent coverage: are the right agents used? (0.5 weight)
    assigned_agents = {a.get("agent", "") for a in assignments}
    expected_set = set(expected_agents)
    if expected_set:
        coverage = len(assigned_agents & expected_set) / len(expected_set)
    else:
        coverage = 1.0
    score += 0.5 * coverage

    # 2. Subtask count: enough subtasks? (0.3 weight)
    if min_subtasks > 0:
        count_ratio = min(len(assignments) / min_subtasks, 1.0)
    else:
        count_ratio = 1.0 if not assignments else 0.0
    score += 0.3 * count_ratio

    # 3. Structural validity: each assignment has required fields (0.2 weight)
    required_fields = {"subtask", "agent", "priority"}
    valid_count = sum(
        1 for a in assignments if required_fields.issubset(a.keys())
    )
    validity = valid_count / len(assignments) if assignments else 1.0
    score += 0.2 * validity

    return score


def _evaluate_efficiency(
    should_delegate: bool,
    did_delegate: bool,
    assignments: list[dict],
) -> float:
    """Score efficiency: not over-delegating, not under-delegating."""
    if should_delegate == did_delegate:
        base = 1.0
    else:
        base = 0.0

    # Penalize excessive subtasks (more than 6 is suspicious)
    if assignments and len(assignments) > 6:
        base *= 0.8

    return base


def run_benchmark() -> float:
    decider = DelegationDecider()

    decision_correct = 0
    assignment_scores: list[float] = []
    efficiency_scores: list[float] = []

    for case in CASES:
        task = case["task"]
        features = case["features"]
        expected_delegate = case["should_delegate"]
        expected_agents = case["expected_agents"]
        min_subtasks = case["min_subtasks"]

        # --- Decision test ---
        try:
            did_delegate = decider.should_delegate(task, features)
        except Exception:
            did_delegate = False

        if did_delegate == expected_delegate:
            decision_correct += 1

        # --- Assignment test (only if delegation expected) ---
        if expected_delegate:
            try:
                assignments = decider.plan_delegation(task, AGENTS)
            except Exception:
                assignments = []
        else:
            # Even if not expected, test that non-delegation produces empty/minimal
            assignments = []

        aq = _evaluate_assignment_quality(assignments, expected_agents, min_subtasks)
        assignment_scores.append(aq)

        eff = _evaluate_efficiency(expected_delegate, did_delegate, assignments)
        efficiency_scores.append(eff)

    n = len(CASES)
    decision_accuracy = decision_correct / n
    assignment_quality = sum(assignment_scores) / n
    efficiency = sum(efficiency_scores) / n

    fitness = (
        0.5 * decision_accuracy
        + 0.3 * assignment_quality
        + 0.2 * efficiency
    )

    print(f"decision_accuracy:  {decision_accuracy:.4f} ({decision_correct}/{n})")
    print(f"assignment_quality: {assignment_quality:.4f}")
    print(f"efficiency:         {efficiency:.4f}")
    print(f"score: {fitness:.6f}")
    return fitness


if __name__ == "__main__":
    run_benchmark()
