"""
Benchmark: Reliability Metrics

Evaluates how well reliability metrics discriminate between consistent
and flaky tasks using simulated agent results.
20 tasks with known ground-truth reliability profiles.

Fitness signal: 0.4 * discrimination_power + 0.3 * k_recommendation_quality + 0.3 * metric_stability

Usage:
    python benchmark.py
"""

from __future__ import annotations

import math
import os
import random
import sys

# Allow importing from contracts/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from harness import ReliabilityMeasure

# ── Simulated agent trial results ─────────────────────────────────────
# Each entry: (list of trial results, ground_truth_category, expected_optimal_k)
# Categories: "reliable" (pass ~100%), "moderate" (pass ~50%), "flaky" (pass ~10-30%), "failing" (pass ~0%)
# Using fixed seed for reproducibility
_RNG = random.Random(42)


def _gen_trials(pass_rate: float, n: int = 20) -> list[bool]:
    """Generate n trial results with given pass rate (deterministic)."""
    return [_RNG.random() < pass_rate for _ in range(n)]


# 20 tasks with different reliability profiles
TASKS: list[tuple[list[bool], str, int]] = [
    # Reliable tasks (should pass consistently) -- optimal k=1
    (_gen_trials(1.0, 20), "reliable", 1),
    (_gen_trials(0.95, 20), "reliable", 1),
    (_gen_trials(0.98, 20), "reliable", 1),
    (_gen_trials(1.0, 20), "reliable", 1),
    (_gen_trials(0.90, 20), "reliable", 1),

    # Moderate tasks (pass about half the time) -- optimal k=3
    (_gen_trials(0.50, 20), "moderate", 3),
    (_gen_trials(0.55, 20), "moderate", 3),
    (_gen_trials(0.45, 20), "moderate", 3),
    (_gen_trials(0.60, 20), "moderate", 3),
    (_gen_trials(0.40, 20), "moderate", 3),

    # Flaky tasks (rarely pass) -- optimal k=5-10
    (_gen_trials(0.15, 20), "flaky", 5),
    (_gen_trials(0.20, 20), "flaky", 5),
    (_gen_trials(0.10, 20), "flaky", 10),
    (_gen_trials(0.25, 20), "flaky", 5),
    (_gen_trials(0.12, 20), "flaky", 10),

    # Failing tasks (almost never pass) -- optimal k=10
    (_gen_trials(0.05, 20), "failing", 10),
    (_gen_trials(0.0, 20), "failing", 10),
    (_gen_trials(0.02, 20), "failing", 10),
    (_gen_trials(0.0, 20), "failing", 10),
    (_gen_trials(0.03, 20), "failing", 10),
]

assert len(TASKS) == 20, f"Expected 20 tasks, got {len(TASKS)}"


def run_benchmark() -> float:
    """Run the reliability metrics benchmark and return fitness score."""
    measurer = ReliabilityMeasure()

    all_results = [t[0] for t in TASKS]
    categories = [t[1] for t in TASKS]
    expected_k = [t[2] for t in TASKS]

    # Get metrics
    metrics = measurer.measure(all_results)
    per_task = metrics["per_task"]

    # --- Component 1: Discrimination power (0.4 weight) ---
    # Can the metric distinguish between reliable, moderate, flaky, and failing?
    # Group pass rates by category and check separation
    category_rates: dict[str, list[float]] = {
        "reliable": [], "moderate": [], "flaky": [], "failing": []
    }
    for i, cat in enumerate(categories):
        category_rates[cat].append(per_task[i]["pass_rate"])

    # Check that category means are properly ordered
    cat_means: dict[str, float] = {}
    for cat, rates in category_rates.items():
        cat_means[cat] = sum(rates) / max(len(rates), 1)

    # Ideal ordering: reliable > moderate > flaky > failing
    ordered_cats = ["reliable", "moderate", "flaky", "failing"]
    ordered_means = [cat_means[c] for c in ordered_cats]

    # Count correct orderings (pairwise)
    correct_orderings = 0
    total_pairs = 0
    for i in range(len(ordered_means)):
        for j in range(i + 1, len(ordered_means)):
            total_pairs += 1
            if ordered_means[i] > ordered_means[j]:
                correct_orderings += 1

    discrimination = correct_orderings / max(total_pairs, 1)

    # Also check within-category variance is low (tight clusters)
    within_var = 0.0
    for cat in ordered_cats:
        rates = category_rates[cat]
        if len(rates) > 1:
            mean = cat_means[cat]
            var = sum((r - mean) ** 2 for r in rates) / len(rates)
            within_var += var
    within_var /= len(ordered_cats)
    # Lower within-category variance is better; map to 0-1
    cluster_tightness = max(0.0, 1.0 - within_var * 10)

    discrimination_score = 0.7 * discrimination + 0.3 * cluster_tightness

    # --- Component 2: K recommendation quality (0.3 weight) ---
    # Does recommend_k match expected optimal k?
    recommended = measurer.recommend_k(all_results)

    # Since baseline returns a single k for all tasks, compare to
    # the most common expected k
    from collections import Counter
    k_counts = Counter(expected_k)
    # Score: how many tasks would be well-served by the recommended k?
    k_match_score = 0.0
    for i, exp_k in enumerate(expected_k):
        if recommended == exp_k:
            k_match_score += 1.0
        elif abs(recommended - exp_k) <= 2:
            k_match_score += 0.5
        else:
            k_match_score += 0.0
    k_recommendation_quality = k_match_score / len(expected_k)

    # --- Component 3: Metric stability (0.3 weight) ---
    # Run measurement on subsets and check consistency
    # Split results into halves
    half = len(all_results) // 2
    m1 = measurer.measure(all_results[:half])
    m2 = measurer.measure(all_results[half:])

    # Stability of pass_at_1 between halves
    full_pass1 = metrics["pass_at_1"]
    h1_pass1 = m1["pass_at_1"]
    h2_pass1 = m2["pass_at_1"]

    # How close are the halves to each other?
    half_diff = abs(h1_pass1 - h2_pass1)
    # Expect some difference since halves have different task mixes,
    # but the metric should be stable within reason
    stability_raw = max(0.0, 1.0 - half_diff * 2)

    # Also check that aggregate_reliability is well-bounded
    agg = metrics["aggregate_reliability"]
    bounded = 1.0 if 0.0 <= agg <= 1.0 else 0.0

    metric_stability = 0.6 * stability_raw + 0.4 * bounded

    # --- Final fitness ---
    fitness = (
        0.4 * discrimination_score
        + 0.3 * k_recommendation_quality
        + 0.3 * metric_stability
    )

    # Print diagnostics
    print("=== Reliability Metrics Benchmark ===")
    print(f"Tasks: {len(TASKS)}")
    print()
    print("Category means:")
    for cat in ordered_cats:
        print(f"  {cat:12s}  mean_pass_rate={cat_means[cat]:.4f}")
    print()
    print(f"Discrimination:    {discrimination_score:.4f}")
    print(f"  ordering correct: {correct_orderings}/{total_pairs}")
    print(f"  cluster tightness: {cluster_tightness:.4f}")
    print()
    print(f"K recommendation:  {k_recommendation_quality:.4f}")
    print(f"  recommended k={recommended}")
    print()
    print(f"Metric stability:  {metric_stability:.4f}")
    print(f"  half diff: {half_diff:.4f}")
    print(f"  bounded: {bounded:.1f}")
    print()
    print(f"Metrics: pass@1={metrics['pass_at_1']:.4f}  "
          f"consistency={metrics['consistency']:.4f}  "
          f"flaky_rate={metrics['flaky_rate']:.4f}")
    print()
    print(f"score: {fitness:.6f}")
    return fitness


if __name__ == "__main__":
    run_benchmark()
