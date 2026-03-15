"""
Benchmark: Grading Strategies

Evaluates how well different grading strategies correlate with human judgment.
30 tasks with human-assigned gold-standard scores.

Fitness signal: 0.5 * correlation_with_human + 0.3 * consistency + 0.2 * speed

Usage:
    python benchmark.py
"""

from __future__ import annotations

import math
import os
import sys
import time

# Allow importing from contracts/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from harness import GradingStrategy

# ── 30 evaluation tasks with human gold-standard scores ──────────────
# Each task: (output, expected, task_description, human_score)
# human_score is the gold standard: 0.0 = completely wrong, 1.0 = perfect
TASKS: list[tuple[str, str, str, float]] = [
    # --- Exact matches (human: 1.0) ---
    ("Paris", "Paris", "What is the capital of France?", 1.0),
    ("42", "42", "What is 6 * 7?", 1.0),
    ("True", "True", "Is 2 < 3?", 1.0),
    ("H2O", "H2O", "Chemical formula for water?", 1.0),
    ("def foo(): pass", "def foo(): pass", "Write a Python no-op function.", 1.0),

    # --- Paraphrases / semantically equivalent (human: 0.8-1.0) ---
    ("The capital of France is Paris", "Paris",
     "What is the capital of France?", 0.9),
    ("It equals 42", "42", "What is 6 * 7?", 0.85),
    ("Yes, 2 is less than 3", "True",
     "Is 2 < 3?", 0.8),
    ("Water's formula is H2O", "H2O",
     "Chemical formula for water?", 0.9),
    ("The answer is approximately 3.14159", "3.14159",
     "What is pi to 5 decimal places?", 0.95),

    # --- Partially correct (human: 0.3-0.7) ---
    ("London", "Paris", "What is the capital of France?", 0.0),
    ("43", "42", "What is 6 * 7?", 0.3),
    ("The capital is either Paris or Lyon", "Paris",
     "What is the capital of France?", 0.6),
    ("H2O2", "H2O", "Chemical formula for water?", 0.3),
    ("3.14", "3.14159", "What is pi to 5 decimal places?", 0.5),
    ("def foo():\n    return None", "def foo(): pass",
     "Write a Python no-op function.", 0.7),
    ("About 40 or so", "42", "What is 6 * 7?", 0.4),
    ("Paris, France", "Paris", "What is the capital of France?", 0.9),
    ("The result is 42.", "42", "What is 6 * 7?", 0.85),
    ("Water is made of hydrogen and oxygen", "H2O",
     "Chemical formula for water?", 0.5),

    # --- Wrong answers (human: 0.0-0.1) ---
    ("Berlin", "Paris", "What is the capital of France?", 0.0),
    ("56", "42", "What is 6 * 7?", 0.0),
    ("False", "True", "Is 2 < 3?", 0.0),
    ("NaCl", "H2O", "Chemical formula for water?", 0.0),
    ("I don't know", "Paris", "What is the capital of France?", 0.0),
    ("2.718", "3.14159", "What is pi to 5 decimal places?", 0.0),
    ("print('hello')", "def foo(): pass",
     "Write a Python no-op function.", 0.1),
    ("Maybe", "True", "Is 2 < 3?", 0.05),
    ("", "42", "What is 6 * 7?", 0.0),
    ("The answer depends on context", "Paris",
     "What is the capital of France?", 0.0),
]

assert len(TASKS) == 30, f"Expected 30 tasks, got {len(TASKS)}"


def pearson_correlation(x: list[float], y: list[float]) -> float:
    """Compute Pearson correlation coefficient."""
    n = len(x)
    if n < 2:
        return 0.0
    mx = sum(x) / n
    my = sum(y) / n
    sx = math.sqrt(sum((xi - mx) ** 2 for xi in x) / n)
    sy = math.sqrt(sum((yi - my) ** 2 for yi in y) / n)
    if sx == 0 or sy == 0:
        return 0.0
    cov = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y)) / n
    return cov / (sx * sy)


def run_benchmark() -> float:
    """Run the grading strategies benchmark and return fitness score."""
    grader = GradingStrategy()

    strategies = [
        "exact_match", "fuzzy_match", "keyword_presence",
        "structural_check", "semantic_similarity",
    ]

    human_scores = [t[3] for t in TASKS]
    strategy_scores: dict[str, list[float]] = {s: [] for s in strategies}
    strategy_times: dict[str, float] = {s: 0.0 for s in strategies}

    # Grade each task with each strategy
    for output, expected, task_desc, _ in TASKS:
        for strategy in strategies:
            start = time.perf_counter()
            result = grader.grade(output, expected, task_desc, strategy=strategy)
            elapsed = time.perf_counter() - start
            strategy_scores[strategy].append(result["score"])
            strategy_times[strategy] += elapsed

    # --- Fitness component 1: Correlation with human judgment (0.5 weight) ---
    # Use the BEST strategy's correlation
    correlations = {}
    for s in strategies:
        correlations[s] = pearson_correlation(strategy_scores[s], human_scores)

    best_correlation = max(correlations.values())
    # Normalize: correlation ranges from -1 to 1, map to 0-1
    correlation_score = (best_correlation + 1.0) / 2.0

    # --- Fitness component 2: Consistency (0.3 weight) ---
    # Measure: do different strategies agree on ordering?
    # Compare each pair of strategies: do they rank tasks the same way?
    agreement_count = 0
    pair_count = 0
    for i in range(len(strategies)):
        for j in range(i + 1, len(strategies)):
            s_i = strategy_scores[strategies[i]]
            s_j = strategy_scores[strategies[j]]
            corr = pearson_correlation(s_i, s_j)
            agreement_count += (corr + 1.0) / 2.0
            pair_count += 1
    consistency_score = agreement_count / max(pair_count, 1)

    # --- Fitness component 3: Speed (0.2 weight) ---
    # All strategies should complete in under 1 second total
    total_time = sum(strategy_times.values())
    # Target: < 0.1s total -> 1.0, > 5s -> 0.0
    speed_score = max(0.0, min(1.0, 1.0 - (total_time - 0.1) / 4.9))

    # --- Final fitness ---
    fitness = (
        0.5 * correlation_score
        + 0.3 * consistency_score
        + 0.2 * speed_score
    )

    # Print diagnostics
    print("=== Grading Strategies Benchmark ===")
    print(f"Tasks: {len(TASKS)}")
    print()
    for s in strategies:
        corr = correlations[s]
        t = strategy_times[s]
        print(f"  {s:25s}  correlation={corr:+.4f}  time={t:.4f}s")
    print()
    print(f"Best correlation:  {best_correlation:+.4f}  -> score={correlation_score:.4f}")
    print(f"Consistency:       {consistency_score:.4f}")
    print(f"Speed:             {speed_score:.4f}  (total={total_time:.4f}s)")
    print()
    print(f"score: {fitness:.6f}")
    return fitness


if __name__ == "__main__":
    run_benchmark()
