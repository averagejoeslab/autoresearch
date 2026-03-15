#!/usr/bin/env python3
"""
Benchmark: Error Recovery

20 outputs with planted errors (and some clean outputs).  Measures:
  - detection F1            – precision/recall of error detection
  - fix accuracy            – does suggest_fix actually fix the error?
  - false alarm rate inverse – 1 - false-positive rate on clean outputs

Fitness = 0.4 * detection_f1 + 0.3 * fix_accuracy + 0.3 * false_alarm_rate_inverse
Prints  : score: X.XXXXXX
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from harness import ErrorRecovery  # noqa: E402

# ─────────────────────────────────────────────────────────────────────
# 20 test cases
# Each has:
#   task            – the originating task
#   output          – text with (or without) planted errors
#   has_errors      – bool: does this output contain errors?
#   error_types     – set of error categories present
#   fixed_output    – the corrected output (or None for clean)
# ─────────────────────────────────────────────────────────────────────
CASES: list[dict] = [
    # ── Outputs WITH errors (14) ─────────────────────────────────────
    {
        "task": "Write a JSON config for the server.",
        "output": '{"host": "localhost", "port": 8080, "debug": true',
        "has_errors": True,
        "error_types": {"unmatched_delimiter"},
        "fixed_output": '{"host": "localhost", "port": 8080, "debug": true}',
    },
    {
        "task": "Implement user registration.",
        "output": "def register(user):\n    # TODO: implement password hashing\n    db.save(user)\n    return user",
        "has_errors": True,
        "error_types": {"incomplete_marker"},
        "fixed_output": "def register(user):\n    user.password = hash_password(user.password)\n    db.save(user)\n    return user",
    },
    {
        "task": "Generate a summary of the meeting.",
        "output": "Meeting summary:\n- Discussed Q4 targets\n- [INSERT ACTION ITEMS HERE]\n- Next meeting on Friday",
        "has_errors": True,
        "error_types": {"placeholder"},
        "fixed_output": "Meeting summary:\n- Discussed Q4 targets\n- Action items assigned to team leads\n- Next meeting on Friday",
    },
    {
        "task": "Write a function to validate email.",
        "output": "def validate(email):\n    # FIXME: regex is incomplete\n    return '@' in email",
        "has_errors": True,
        "error_types": {"incomplete_marker"},
        "fixed_output": "def validate(email):\n    import re\n    return bool(re.match(r'^[\\w.+-]+@[\\w-]+\\.[\\w.]+$', email))",
    },
    {
        "task": "Create an HTML page.",
        "output": "<html><head><title>Test</title></head><body><div><p>Hello</p></body></html>",
        "has_errors": True,
        "error_types": {"unmatched_delimiter"},
        "fixed_output": "<html><head><title>Test</title></head><body><div><p>Hello</p></div></body></html>",
    },
    {
        "task": "Write a math expression evaluator.",
        "output": "def evaluate(expr):\n    # XXX: eval is dangerous\n    return eval(expr)",
        "has_errors": True,
        "error_types": {"incomplete_marker"},
        "fixed_output": "def evaluate(expr):\n    import ast\n    return ast.literal_eval(expr)",
    },
    {
        "task": "Draft a project proposal.",
        "output": "Project: Data Platform\nBudget: [YOUR BUDGET HERE]\nTimeline: 6 months\nTeam: [PLACEHOLDER - add team members]",
        "has_errors": True,
        "error_types": {"placeholder"},
        "fixed_output": "Project: Data Platform\nBudget: $150,000\nTimeline: 6 months\nTeam: 3 engineers, 1 PM, 1 designer",
    },
    {
        "task": "Write a list comprehension.",
        "output": "result = [x**2 for x in range(10]",
        "has_errors": True,
        "error_types": {"unmatched_delimiter"},
        "fixed_output": "result = [x**2 for x in range(10)]",
    },
    {
        "task": "Create a deployment script.",
        "output": "#!/bin/bash\nset -e\n# HACK: hardcoded path\ncd /tmp/deploy\n./run.sh",
        "has_errors": True,
        "error_types": {"incomplete_marker"},
        "fixed_output": '#!/bin/bash\nset -e\nDEPLOY_DIR="${DEPLOY_DIR:-/opt/deploy}"\ncd "$DEPLOY_DIR"\n./run.sh',
    },
    {
        "task": "Write a SQL query to find top customers.",
        "output": "SELECT name, SUM(amount FROM orders GROUP BY name ORDER BY SUM(amount) DESC LIMIT 10;",
        "has_errors": True,
        "error_types": {"unmatched_delimiter"},
        "fixed_output": "SELECT name, SUM(amount) FROM orders GROUP BY name ORDER BY SUM(amount) DESC LIMIT 10;",
    },
    {
        "task": "Write an API error response.",
        "output": '{"error": "Not found", "code": 404, "details": [FILL IN DETAILS]]}',
        "has_errors": True,
        "error_types": {"placeholder", "unmatched_delimiter"},
        "fixed_output": '{"error": "Not found", "code": 404, "details": []}',
    },
    {
        "task": "Create a configuration template.",
        "output": "database:\n  host: [INSERT HOST]\n  port: 5432\n  name: [INSERT DB NAME]",
        "has_errors": True,
        "error_types": {"placeholder"},
        "fixed_output": "database:\n  host: db.example.com\n  port: 5432\n  name: myapp_production",
    },
    {
        "task": "Write a test case.",
        "output": "def test_addition():\n    # TODO: add edge cases\n    assert add(1, 2) == 3",
        "has_errors": True,
        "error_types": {"incomplete_marker"},
        "fixed_output": "def test_addition():\n    assert add(1, 2) == 3\n    assert add(0, 0) == 0\n    assert add(-1, 1) == 0",
    },
    {
        "task": "Write a nested function call.",
        "output": "result = process(transform(data, config)",
        "has_errors": True,
        "error_types": {"unmatched_delimiter"},
        "fixed_output": "result = process(transform(data, config))",
    },

    # ── Clean outputs (6) ────────────────────────────────────────────
    {
        "task": "Write a hello world program.",
        "output": 'print("Hello, World!")',
        "has_errors": False,
        "error_types": set(),
        "fixed_output": None,
    },
    {
        "task": "Add two numbers.",
        "output": "def add(a, b):\n    return a + b",
        "has_errors": False,
        "error_types": set(),
        "fixed_output": None,
    },
    {
        "task": "Write a greeting message.",
        "output": "Welcome to our platform! We are glad to have you.",
        "has_errors": False,
        "error_types": set(),
        "fixed_output": None,
    },
    {
        "task": "Create a valid JSON object.",
        "output": '{"name": "Alice", "age": 30, "active": true}',
        "has_errors": False,
        "error_types": set(),
        "fixed_output": None,
    },
    {
        "task": "Write a list of fruits.",
        "output": "Apples, bananas, cherries, dates, and elderberries.",
        "has_errors": False,
        "error_types": set(),
        "fixed_output": None,
    },
    {
        "task": "Compute the area of a circle.",
        "output": "import math\n\ndef area(r):\n    return math.pi * r ** 2",
        "has_errors": False,
        "error_types": set(),
        "fixed_output": None,
    },
]


# ─────────────────────────────────────────────────────────────────────
# Scoring helpers
# ─────────────────────────────────────────────────────────────────────

def _detection_f1(predicted_has: list[bool], actual_has: list[bool]) -> float:
    tp = sum(1 for p, a in zip(predicted_has, actual_has) if p and a)
    fp = sum(1 for p, a in zip(predicted_has, actual_has) if p and not a)
    fn = sum(1 for p, a in zip(predicted_has, actual_has) if not p and a)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def _fix_accuracy(predictions: list[str | None], ground_truth: list[str | None]) -> float:
    """Fraction of erroneous outputs where the fix matches ground truth."""
    total = 0
    correct = 0
    for pred, gt in zip(predictions, ground_truth):
        if gt is None:
            continue  # clean output, skip
        total += 1
        if pred is not None and pred.strip() == gt.strip():
            correct += 1
    return correct / total if total else 1.0


def _false_alarm_rate_inverse(predicted_has: list[bool], actual_has: list[bool]) -> float:
    """1 - (false positives among clean outputs)."""
    clean_total = sum(1 for a in actual_has if not a)
    if clean_total == 0:
        return 1.0
    false_alarms = sum(1 for p, a in zip(predicted_has, actual_has) if p and not a)
    return 1.0 - false_alarms / clean_total


def run_benchmark() -> float:
    er = ErrorRecovery()

    predicted_has: list[bool] = []
    actual_has: list[bool] = []
    fixes: list[str | None] = []
    ground_truth_fixes: list[str | None] = []

    for case in CASES:
        errors = er.detect_errors(case["output"], case["task"])
        has = len(errors) > 0
        predicted_has.append(has)
        actual_has.append(case["has_errors"])

        # Try to fix if errors were detected
        if errors:
            fixed = case["output"]
            for err in errors:
                fixed = er.suggest_fix(fixed, err)
            fixes.append(fixed)
        else:
            fixes.append(None)
        ground_truth_fixes.append(case.get("fixed_output"))

    det_f1 = _detection_f1(predicted_has, actual_has)
    fix_acc = _fix_accuracy(fixes, ground_truth_fixes)
    far_inv = _false_alarm_rate_inverse(predicted_has, actual_has)

    score = 0.4 * det_f1 + 0.3 * fix_acc + 0.3 * far_inv

    print(f"  detection_f1         : {det_f1:.6f}")
    print(f"  fix_accuracy         : {fix_acc:.6f}")
    print(f"  false_alarm_inv      : {far_inv:.6f}")
    return score


if __name__ == "__main__":
    final = run_benchmark()
    print(f"score: {final:.6f}")
