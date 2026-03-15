#!/usr/bin/env python3
"""
Benchmark: Difficulty Calibration
===================================
20 tasks with known solve rates testing calibration accuracy.

Fitness = 0.5 * ranking_correlation + 0.3 * prediction_accuracy + 0.2 * distribution_uniformity

Run:  python benchmark.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from harness import DifficultyCalibrater  # noqa: E402

# ── Inline task data with known solve rates ─────────────────────────
# Ground truth difficulty is derived from solve rates:
#   90%+ solved = easy (1-2), 40-60% = medium (3), <20% = hard (4-5)

TASKS: list[dict] = [
    # Easy tasks (high solve rate)
    {"id": "t01", "prompt": "What is 2+2?",                                           "difficulty": 1, "true_solve_rate": 0.98},
    {"id": "t02", "prompt": "Is the sky blue?",                                        "difficulty": 1, "true_solve_rate": 0.95},
    {"id": "t03", "prompt": "Name a color of the rainbow",                             "difficulty": 1, "true_solve_rate": 0.93},
    {"id": "t04", "prompt": "What is the capital of France?",                          "difficulty": 1, "true_solve_rate": 0.90},
    {"id": "t05", "prompt": "How many days in a week?",                                "difficulty": 1, "true_solve_rate": 0.97},
    {"id": "t06", "prompt": "What comes after Monday?",                                "difficulty": 1, "true_solve_rate": 0.92},
    {"id": "t07", "prompt": "Name one planet in our solar system",                     "difficulty": 2, "true_solve_rate": 0.88},
    # Medium tasks
    {"id": "t08", "prompt": "What is the square root of 144?",                         "difficulty": 3, "true_solve_rate": 0.60},
    {"id": "t09", "prompt": "Name the chemical formula for water",                     "difficulty": 3, "true_solve_rate": 0.55},
    {"id": "t10", "prompt": "What year did World War 2 end?",                          "difficulty": 3, "true_solve_rate": 0.50},
    {"id": "t11", "prompt": "Convert 100 Celsius to Fahrenheit",                       "difficulty": 3, "true_solve_rate": 0.45},
    {"id": "t12", "prompt": "What is the derivative of x squared?",                    "difficulty": 3, "true_solve_rate": 0.48},
    {"id": "t13", "prompt": "Name three programming paradigms",                        "difficulty": 3, "true_solve_rate": 0.42},
    # Hard tasks (low solve rate)
    {"id": "t14", "prompt": "What is the time complexity of merge sort and why?",       "difficulty": 4, "true_solve_rate": 0.18},
    {"id": "t15", "prompt": "Explain the CAP theorem and its implications for distributed databases", "difficulty": 5, "true_solve_rate": 0.12},
    {"id": "t16", "prompt": "Derive the quadratic formula from ax^2+bx+c=0",          "difficulty": 5, "true_solve_rate": 0.08},
    {"id": "t17", "prompt": "What is the significance of Godel's incompleteness theorems for formal systems?", "difficulty": 5, "true_solve_rate": 0.05},
    {"id": "t18", "prompt": "Explain why P vs NP is important and describe one approach to resolving it", "difficulty": 5, "true_solve_rate": 0.07},
    {"id": "t19", "prompt": "Describe the difference between strong and weak AI and the Chinese Room argument", "difficulty": 4, "true_solve_rate": 0.15},
    {"id": "t20", "prompt": "Implement a balanced binary search tree insert operation with rotations", "difficulty": 5, "true_solve_rate": 0.10},
]

# ── Simulated agent performance data ────────────────────────────────
# We simulate 5 agents with varying skill levels

def _generate_results(tasks: list[dict]) -> list[dict]:
    """Generate simulated solve attempts based on true solve rates."""
    import hashlib
    results = []
    agents = ["agent_expert", "agent_good", "agent_mid", "agent_weak", "agent_novice"]
    # Agent skill modifiers (applied to solve rate)
    skill = {"agent_expert": 1.2, "agent_good": 1.0, "agent_mid": 0.8,
             "agent_weak": 0.5, "agent_novice": 0.3}

    for task in tasks:
        tid = task["id"]
        base_rate = task["true_solve_rate"]
        for agent_id in agents:
            modifier = skill[agent_id]
            effective_rate = min(1.0, base_rate * modifier)
            # Deterministic "random" based on hash
            h = int(hashlib.md5(f"{tid}:{agent_id}".encode()).hexdigest()[:8], 16)
            threshold = int(effective_rate * 0xFFFFFFFF)
            solved = h < threshold
            results.append({
                "task_id": tid,
                "solved": solved,
                "agent_id": agent_id,
            })
    return results


def _spearman_rank_correlation(x: list[float], y: list[float]) -> float:
    """Compute Spearman rank correlation between two lists."""
    n = len(x)
    if n < 2:
        return 0.0

    def _rank(vals: list[float]) -> list[float]:
        sorted_indices = sorted(range(n), key=lambda i: vals[i])
        ranks = [0.0] * n
        for rank_val, idx in enumerate(sorted_indices):
            ranks[idx] = float(rank_val)
        return ranks

    rx = _rank(x)
    ry = _rank(y)

    d_sq_sum = sum((rx[i] - ry[i]) ** 2 for i in range(n))
    return 1.0 - (6.0 * d_sq_sum) / (n * (n * n - 1))


def run_benchmark() -> float:
    calibrater = DifficultyCalibrater()

    # Strip true_solve_rate from tasks before passing to calibrater
    input_tasks = [{k: v for k, v in t.items() if k != "true_solve_rate"} for t in TASKS]
    results = _generate_results(TASKS)

    # ── Test 1: Calibration accuracy ─────────────────────────────────
    try:
        calibrated = calibrater.calibrate(input_tasks, results)
    except Exception:
        calibrated = input_tasks

    # Build ground-truth difficulty from solve rates
    true_difficulties = []
    calibrated_difficulties = []
    for orig, cal in zip(TASKS, calibrated):
        # Ground truth: map solve rate to difficulty
        sr = orig["true_solve_rate"]
        true_diff = max(1.0, min(5.0, 1.0 + 4.0 * (1.0 - sr)))
        true_difficulties.append(true_diff)
        calibrated_difficulties.append(float(cal.get("difficulty", 3)))

    # Ranking correlation: does calibration order match true order?
    ranking_corr = _spearman_rank_correlation(true_difficulties, calibrated_difficulties)
    # Normalize to 0-1 (correlation ranges from -1 to 1)
    ranking_score = max(0.0, (ranking_corr + 1.0) / 2.0)

    # ── Test 2: Prediction accuracy ──────────────────────────────────
    prediction_errors: list[float] = []
    for task_data in TASKS:
        pred_input = {"prompt": task_data["prompt"]}
        try:
            predicted = calibrater.predict_difficulty(pred_input)
        except Exception:
            predicted = 3.0

        true_diff = 1.0 + 4.0 * (1.0 - task_data["true_solve_rate"])
        error = abs(predicted - true_diff) / 4.0  # normalize error to 0-1
        prediction_errors.append(error)

    avg_error = sum(prediction_errors) / len(prediction_errors)
    prediction_accuracy = max(0.0, 1.0 - avg_error)

    # ── Test 3: Distribution uniformity ──────────────────────────────
    # After calibration, check if difficulties are spread across 1-5
    if calibrated:
        diff_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        for t in calibrated:
            d = t.get("difficulty", 3)
            d_clamped = max(1, min(5, int(round(d))))
            diff_counts[d_clamped] = diff_counts.get(d_clamped, 0) + 1

        # Ideal: roughly uniform distribution
        n_tasks = len(calibrated)
        ideal_per_bin = n_tasks / 5.0
        # Use chi-squared-like measure
        deviations = sum(
            (count - ideal_per_bin) ** 2 / ideal_per_bin
            for count in diff_counts.values()
        )
        # Normalize: 0 deviation = 1.0, high deviation = 0.0
        # Max possible chi-sq is roughly n_tasks * 4/5 * 5 = 4*n_tasks
        max_dev = 4.0 * n_tasks
        uniformity = max(0.0, 1.0 - deviations / max_dev)
    else:
        uniformity = 0.0

    fitness = (
        0.5 * ranking_score
        + 0.3 * prediction_accuracy
        + 0.2 * uniformity
    )

    print(f"ranking_correlation:     {ranking_score:.4f} (raw: {ranking_corr:.4f})")
    print(f"prediction_accuracy:     {prediction_accuracy:.4f}")
    print(f"distribution_uniformity: {uniformity:.4f}")
    print(f"score: {fitness:.6f}")
    return fitness


if __name__ == "__main__":
    run_benchmark()
