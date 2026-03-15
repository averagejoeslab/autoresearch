#!/usr/bin/env python3
"""
Benchmark: Self-Check

30 (task, output) pairs with ground-truth labels.  Measures:
  - judgment_accuracy     – does is_correct match the label?
  - calibration_score     – is confidence correlated with correctness?
  - issue_detection_f1    – does it catch known issues?

Fitness = 0.4 * judgment_accuracy + 0.3 * calibration_score + 0.3 * issue_detection_f1
Prints  : score: X.XXXXXX
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from harness import SelfChecker  # noqa: E402

# ─────────────────────────────────────────────────────────────────────
# 30 test cases
# Each has: task, output, is_correct (bool), issues (list of issue keywords)
# ─────────────────────────────────────────────────────────────────────
CASES: list[dict] = [
    # ── Correct outputs (15) ─────────────────────────────────────────
    {"task": "What is 2 + 2?",
     "output": "4",
     "is_correct": True, "issues": []},

    {"task": "List the primary colors.",
     "output": "Red, blue, and yellow are the primary colors.",
     "is_correct": True, "issues": []},

    {"task": "Convert 100 Celsius to Fahrenheit.",
     "output": "100 degrees Celsius is 212 degrees Fahrenheit.",
     "is_correct": True, "issues": []},

    {"task": "What is the capital of France?",
     "output": "The capital of France is Paris.",
     "is_correct": True, "issues": []},

    {"task": "Write a Python function to add two numbers.",
     "output": "def add(a, b):\n    return a + b",
     "is_correct": True, "issues": []},

    {"task": "What is the square root of 144?",
     "output": "The square root of 144 is 12.",
     "is_correct": True, "issues": []},

    {"task": "Name three planets in our solar system.",
     "output": "Mars, Jupiter, and Saturn.",
     "is_correct": True, "issues": []},

    {"task": "What does HTML stand for?",
     "output": "HTML stands for HyperText Markup Language.",
     "is_correct": True, "issues": []},

    {"task": "Is water H2O?",
     "output": "Yes, water is H2O — two hydrogen atoms bonded to one oxygen atom.",
     "is_correct": True, "issues": []},

    {"task": "What year did World War II end?",
     "output": "World War II ended in 1945.",
     "is_correct": True, "issues": []},

    {"task": "Define photosynthesis in one sentence.",
     "output": "Photosynthesis is the process by which green plants convert sunlight, water, and CO2 into glucose and oxygen.",
     "is_correct": True, "issues": []},

    {"task": "Sort these numbers ascending: 5, 2, 8, 1.",
     "output": "1, 2, 5, 8",
     "is_correct": True, "issues": []},

    {"task": "What is the boiling point of water at sea level?",
     "output": "100 degrees Celsius or 212 degrees Fahrenheit.",
     "is_correct": True, "issues": []},

    {"task": "Name the author of Romeo and Juliet.",
     "output": "William Shakespeare wrote Romeo and Juliet.",
     "is_correct": True, "issues": []},

    {"task": "What is 15% of 200?",
     "output": "15% of 200 is 30.",
     "is_correct": True, "issues": []},

    # ── Incorrect outputs (15) ───────────────────────────────────────
    {"task": "What is 2 + 2?",
     "output": "5",
     "is_correct": False, "issues": ["wrong_answer"]},

    {"task": "What is the capital of Australia?",
     "output": "The capital of Australia is Sydney.",
     "is_correct": False, "issues": ["factual_error"]},

    {"task": "Convert 0 Celsius to Fahrenheit.",
     "output": "0 degrees Celsius is 0 degrees Fahrenheit.",
     "is_correct": False, "issues": ["wrong_calculation"]},

    {"task": "List the continents.",
     "output": "Europe, Asia, Africa, North America, South America.",
     "is_correct": False, "issues": ["incomplete"]},

    {"task": "Write a function to compute factorial.",
     "output": "def factorial(n):\n    return n * factorial(n)",
     "is_correct": False, "issues": ["logic_error"]},

    {"task": "What is the speed of light?",
     "output": "The speed of light is approximately 300,000 km/h.",
     "is_correct": False, "issues": ["wrong_unit"]},

    {"task": "Name the first president of the United States.",
     "output": "The first president of the United States was Abraham Lincoln.",
     "is_correct": False, "issues": ["factual_error"]},

    {"task": "What is the chemical formula for table salt?",
     "output": "The chemical formula for table salt is NaO.",
     "is_correct": False, "issues": ["factual_error"]},

    {"task": "Sort these numbers descending: 3, 1, 4, 1, 5.",
     "output": "1, 1, 3, 4, 5",
     "is_correct": False, "issues": ["wrong_order"]},

    {"task": "What is the largest ocean on Earth?",
     "output": "The largest ocean on Earth is the Atlantic Ocean.",
     "is_correct": False, "issues": ["factual_error"]},

    {"task": "Explain what a for-loop does.",
     "output": "A for-loop is a type of variable that stores numbers.",
     "is_correct": False, "issues": ["conceptual_error"]},

    {"task": "What is 7 * 8?",
     "output": "7 times 8 equals 54.",
     "is_correct": False, "issues": ["wrong_calculation"]},

    {"task": "Name three states in the USA.",
     "output": "London, Paris, and Berlin.",
     "is_correct": False, "issues": ["completely_wrong"]},

    {"task": "What does CPU stand for?",
     "output": "CPU stands for Central Program Utility.",
     "is_correct": False, "issues": ["factual_error"]},

    {"task": "How many days are in a leap year?",
     "output": "A leap year has 364 days.",
     "is_correct": False, "issues": ["wrong_answer"]},
]


# ─────────────────────────────────────────────────────────────────────
# Scoring helpers
# ─────────────────────────────────────────────────────────────────────

def _calibration(predictions: list[dict], labels: list[bool]) -> float:
    """Measure confidence-correctness correlation.

    Bucket by confidence quintiles, compare predicted confidence to
    actual accuracy in each bucket.  Return 1 - mean_absolute_deviation.
    """
    if not predictions:
        return 0.0

    n_bins = 5
    bins: list[list[tuple[float, bool]]] = [[] for _ in range(n_bins)]
    for pred, label in zip(predictions, labels):
        conf = max(0.0, min(1.0, pred.get("confidence", 0.5)))
        b = min(int(conf * n_bins), n_bins - 1)
        bins[b].append((conf, label))

    total_dev = 0.0
    counted = 0
    for bucket in bins:
        if not bucket:
            continue
        avg_conf = sum(c for c, _ in bucket) / len(bucket)
        avg_acc = sum(1.0 for _, l in bucket if l) / len(bucket)
        total_dev += abs(avg_conf - avg_acc)
        counted += 1

    if counted == 0:
        return 0.0
    mae = total_dev / counted
    return max(0.0, 1.0 - mae)


def _issue_detection_f1(predictions: list[dict], cases: list[dict]) -> float:
    """F1 for issue detection.

    A prediction "detects" an issue if it has any item in its issues list.
    Ground truth has issues when is_correct is False.
    """
    tp = fp = fn = 0
    for pred, case in zip(predictions, cases):
        pred_has_issues = len(pred.get("issues", [])) > 0
        actual_has_issues = len(case.get("issues", [])) > 0
        if pred_has_issues and actual_has_issues:
            tp += 1
        elif pred_has_issues and not actual_has_issues:
            fp += 1
        elif not pred_has_issues and actual_has_issues:
            fn += 1
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def run_benchmark() -> float:
    checker = SelfChecker()

    predictions: list[dict] = []
    labels: list[bool] = []
    correct_count = 0

    for case in CASES:
        result = checker.check(case["output"], case["task"])
        predictions.append(result)
        labels.append(case["is_correct"])
        pred_correct = result.get("is_correct", True)
        if pred_correct == case["is_correct"]:
            correct_count += 1

    judgment_accuracy = correct_count / len(CASES)
    calibration_score = _calibration(predictions, labels)
    issue_f1 = _issue_detection_f1(predictions, CASES)

    score = 0.4 * judgment_accuracy + 0.3 * calibration_score + 0.3 * issue_f1

    print(f"  judgment_accuracy : {judgment_accuracy:.6f}")
    print(f"  calibration_score : {calibration_score:.6f}")
    print(f"  issue_detection_f1: {issue_f1:.6f}")
    return score


if __name__ == "__main__":
    final = run_benchmark()
    print(f"score: {final:.6f}")
