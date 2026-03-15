#!/usr/bin/env python3
"""
Benchmark: Task Generation (Meta-benchmark)
=============================================
Tests whether generated tasks discriminate between agents of
different quality levels.

Fitness = 0.4 * discrimination + 0.3 * difficulty_correlation + 0.3 * validity

Run:  python benchmark.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from harness import TaskGenerator  # noqa: E402

# ── Simulated agents of varying quality ──────────────────────────────

def _perfect_agent(prompt: str, expected: str) -> str:
    """Always returns the exact expected answer."""
    return expected


def _mediocre_agent(prompt: str, expected: str) -> str:
    """Returns correct answer only for short/simple prompts."""
    # Gets it right if the expected answer is short (< 10 chars)
    # and the prompt is simple (< 40 chars)
    if len(expected) < 10 and len(prompt) < 40:
        return expected
    # Otherwise returns a plausible but wrong answer
    if expected.isdigit():
        return str(int(expected) + 1)  # off by one
    words = expected.split()
    if len(words) > 1:
        return " ".join(words[:-1])  # drops last word
    return expected[:len(expected) // 2] if len(expected) > 2 else "unknown"


def _bad_agent(prompt: str, expected: str) -> str:
    """Almost always wrong."""
    # Only gets trivially obvious answers right
    if expected.lower() in ("yes", "no", "true", "false"):
        return expected
    return "I don't know"


def _score_agent_on_tasks(
    agent_fn,
    tasks: list[dict],
) -> float:
    """Score an agent against a task set. Returns fraction correct."""
    if not tasks:
        return 0.0
    correct = 0
    for task in tasks:
        prompt = task["prompt"]
        expected = task["expected_answer"]
        response = agent_fn(prompt, expected)
        # Exact match (case-insensitive, stripped)
        if response.strip().lower() == expected.strip().lower():
            correct += 1
    return correct / len(tasks)


# ── Test configurations ─────────────────────────────────────────────

# We test generation across multiple domains and difficulty levels
GENERATION_CONFIGS = [
    {"domain": "geography", "difficulty": 1, "n": 10},
    {"domain": "geography", "difficulty": 3, "n": 10},
    {"domain": "geography", "difficulty": 5, "n": 10},
    {"domain": "arithmetic", "difficulty": 1, "n": 10},
    {"domain": "arithmetic", "difficulty": 3, "n": 10},
    {"domain": "arithmetic", "difficulty": 5, "n": 10},
    {"domain": "color-theory", "difficulty": 2, "n": 5},
    {"domain": "general", "difficulty": 2, "n": 10},
    {"domain": "general", "difficulty": 4, "n": 10},
]


def run_benchmark() -> float:
    generator = TaskGenerator()

    discrimination_scores: list[float] = []
    difficulty_corr_scores: list[float] = []
    validity_scores: list[float] = []

    for config in GENERATION_CONFIGS:
        domain = config["domain"]
        difficulty = config["difficulty"]
        n = config["n"]

        try:
            tasks = generator.generate(domain, difficulty, n)
        except Exception:
            tasks = []

        if not tasks:
            discrimination_scores.append(0.0)
            difficulty_corr_scores.append(0.0)
            validity_scores.append(0.0)
            continue

        # ── Validity: are the generated tasks well-formed? ───────────
        valid_count = 0
        for task in tasks:
            has_prompt = isinstance(task.get("prompt"), str) and len(task["prompt"]) > 0
            has_answer = isinstance(task.get("expected_answer"), str) and len(task["expected_answer"]) > 0
            has_diff = isinstance(task.get("difficulty"), int) and 1 <= task["difficulty"] <= 5
            has_tags = isinstance(task.get("tags"), list) and len(task["tags"]) > 0
            if has_prompt and has_answer and has_diff and has_tags:
                valid_count += 1
        validity = valid_count / len(tasks)
        validity_scores.append(validity)

        # ── Discrimination: do tasks separate good from bad agents? ──
        score_perfect = _score_agent_on_tasks(_perfect_agent, tasks)
        score_mediocre = _score_agent_on_tasks(_mediocre_agent, tasks)
        score_bad = _score_agent_on_tasks(_bad_agent, tasks)

        # Ideal: perfect >> mediocre >> bad
        # Score spread: perfect - bad (should be large)
        spread = score_perfect - score_bad

        # Ordering bonus: perfect > mediocre > bad
        ordering = 0.0
        if score_perfect > score_mediocre:
            ordering += 0.5
        if score_mediocre > score_bad:
            ordering += 0.5

        discrimination = 0.5 * spread + 0.5 * ordering
        discrimination_scores.append(discrimination)

        # ── Difficulty correlation ───────────────────────────────────
        # Check if tasks have the requested difficulty
        avg_task_diff = sum(t.get("difficulty", 0) for t in tasks) / len(tasks)
        # How close is the average to what was requested?
        diff_error = abs(avg_task_diff - difficulty) / 4.0  # normalize to 0-1
        difficulty_corr = max(0.0, 1.0 - diff_error)
        difficulty_corr_scores.append(difficulty_corr)

    # Aggregate across all configs
    avg_discrimination = sum(discrimination_scores) / len(discrimination_scores)
    avg_difficulty = sum(difficulty_corr_scores) / len(difficulty_corr_scores)
    avg_validity = sum(validity_scores) / len(validity_scores)

    fitness = (
        0.4 * avg_discrimination
        + 0.3 * avg_difficulty
        + 0.3 * avg_validity
    )

    print(f"discrimination:         {avg_discrimination:.4f}")
    print(f"difficulty_correlation: {avg_difficulty:.4f}")
    print(f"validity:               {avg_validity:.4f}")
    print(f"score: {fitness:.6f}")
    return fitness


if __name__ == "__main__":
    run_benchmark()
