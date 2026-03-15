#!/usr/bin/env python3
"""
Benchmark: Automated Grading

30 outputs pre-graded by humans (gold-standard scores 0-1).  Measures:
  - Pearson correlation    between auto-grader and human scores
  - Mean absolute error    (lower is better, reported as 1-MAE)
  - Ranking agreement      (Kendall's tau)

Fitness = 0.4 * correlation + 0.3 * (1 - mae) + 0.3 * ranking_agreement
Prints  : score: X.XXXXXX
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from harness import AutoGrader  # noqa: E402

# ─────────────────────────────────────────────────────────────────────
# Standard rubric used across all tasks (unless overridden)
# ─────────────────────────────────────────────────────────────────────
STANDARD_RUBRIC: dict = {
    "criteria": {
        "correctness": {"weight": 0.4, "description": "Factual accuracy and logical soundness"},
        "completeness": {"weight": 0.3, "description": "All parts of the task are addressed"},
        "clarity": {"weight": 0.3, "description": "Clear, well-organized, easy to understand"},
    }
}

# ─────────────────────────────────────────────────────────────────────
# 30 graded tasks
# Each has: task, output, human_score (float 0-1), rubric (optional)
# ─────────────────────────────────────────────────────────────────────
CASES: list[dict] = [
    # ── High-quality outputs (10) — human_score >= 0.8 ───────────────
    {"task": "Explain what a REST API is.",
     "output": "A REST API (Representational State Transfer Application Programming Interface) is an architectural style for designing networked applications. It uses HTTP methods (GET, POST, PUT, DELETE) to perform CRUD operations on resources identified by URLs. REST APIs are stateless, meaning each request contains all information needed for processing. They typically exchange data in JSON or XML format and follow principles like uniform interface, client-server separation, and layered system architecture.",
     "human_score": 0.95},

    {"task": "Write a Python function to check if a number is prime.",
     "output": "def is_prime(n):\n    if n < 2:\n        return False\n    if n == 2:\n        return True\n    if n % 2 == 0:\n        return False\n    for i in range(3, int(n**0.5) + 1, 2):\n        if n % i == 0:\n            return False\n    return True",
     "human_score": 0.92},

    {"task": "Summarize the benefits of version control.",
     "output": "Version control systems like Git provide several key benefits: (1) History tracking — every change is recorded with who made it and why. (2) Collaboration — multiple developers can work on the same codebase without conflicts. (3) Branching — experimental features can be developed in isolation. (4) Rollback — mistakes can be undone by reverting to previous versions. (5) Code review — pull requests enable peer review before merging changes.",
     "human_score": 0.90},

    {"task": "Explain the difference between a stack and a queue.",
     "output": "A stack follows Last-In-First-Out (LIFO) order: the most recently added element is removed first, like a stack of plates. Operations are push (add to top) and pop (remove from top). A queue follows First-In-First-Out (FIFO) order: the earliest added element is removed first, like a line at a store. Operations are enqueue (add to back) and dequeue (remove from front). Stacks are used for function calls and undo operations; queues are used for task scheduling and breadth-first search.",
     "human_score": 0.93},

    {"task": "What is the time complexity of binary search?",
     "output": "Binary search has O(log n) time complexity. It works by repeatedly dividing the sorted search space in half. At each step, it compares the target with the middle element and eliminates half of the remaining elements. For an array of n elements, it takes at most log2(n) comparisons. The space complexity is O(1) for iterative implementation and O(log n) for recursive due to the call stack.",
     "human_score": 0.94},

    {"task": "Write a SQL query to find duplicate emails in a users table.",
     "output": "SELECT email, COUNT(*) as count\nFROM users\nGROUP BY email\nHAVING COUNT(*) > 1\nORDER BY count DESC;",
     "human_score": 0.88},

    {"task": "Explain what Docker containers are.",
     "output": "Docker containers are lightweight, standalone, executable packages that include everything needed to run software: code, runtime, libraries, and system tools. Unlike virtual machines, containers share the host OS kernel, making them more efficient. They ensure consistency across development, testing, and production environments. Docker uses images (blueprints) to create containers, and Dockerfiles to define how images are built.",
     "human_score": 0.89},

    {"task": "Describe the MVC design pattern.",
     "output": "MVC (Model-View-Controller) separates an application into three components: the Model manages data and business logic, the View handles the user interface and display, and the Controller processes user input and coordinates between Model and View. This separation makes code more maintainable, testable, and allows multiple developers to work on different components simultaneously. Common in web frameworks like Django, Rails, and Spring.",
     "human_score": 0.91},

    {"task": "Write a regular expression to match email addresses.",
     "output": "^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$\n\nThis matches: one or more alphanumeric characters (plus dots, underscores, percent, plus, hyphen) before the @, followed by a domain name with at least one dot and a TLD of 2 or more letters.",
     "human_score": 0.85},

    {"task": "What is the CAP theorem?",
     "output": "The CAP theorem states that a distributed data store cannot simultaneously provide all three guarantees: Consistency (every read receives the most recent write), Availability (every request receives a response), and Partition tolerance (the system continues despite network partitions). Since network partitions are inevitable in distributed systems, designers must choose between consistency (CP systems like HBase) and availability (AP systems like Cassandra).",
     "human_score": 0.92},

    # ── Medium-quality outputs (10) — human_score 0.4-0.7 ───────────
    {"task": "Explain what a REST API is.",
     "output": "A REST API is a way for programs to talk to each other over the internet using HTTP.",
     "human_score": 0.40},

    {"task": "Write a Python function to check if a number is prime.",
     "output": "def is_prime(n):\n    for i in range(2, n):\n        if n % i == 0:\n            return False\n    return True",
     "human_score": 0.55},

    {"task": "Summarize the benefits of version control.",
     "output": "Version control lets you track changes and collaborate with others. It is useful for teams.",
     "human_score": 0.35},

    {"task": "Explain the difference between a stack and a queue.",
     "output": "A stack is LIFO and a queue is FIFO. They are both data structures used in programming.",
     "human_score": 0.45},

    {"task": "What is the time complexity of binary search?",
     "output": "Binary search is O(log n) because it splits the array in half each time.",
     "human_score": 0.55},

    {"task": "Write a SQL query to find duplicate emails.",
     "output": "SELECT * FROM users WHERE email IN (SELECT email FROM users GROUP BY email);",
     "human_score": 0.50},

    {"task": "Explain what Docker containers are.",
     "output": "Docker is a tool for running applications in containers. Containers are like lightweight VMs.",
     "human_score": 0.45},

    {"task": "Describe the MVC design pattern.",
     "output": "MVC stands for Model View Controller. It separates the app into three parts.",
     "human_score": 0.38},

    {"task": "Write a regular expression to match email addresses.",
     "output": ".*@.*\\..*",
     "human_score": 0.30},

    {"task": "What is the CAP theorem?",
     "output": "CAP theorem says you can only have two of three things: consistency, availability, and partition tolerance.",
     "human_score": 0.50},

    # ── Low-quality outputs (10) — human_score < 0.3 ────────────────
    {"task": "Explain what a REST API is.",
     "output": "It's an API.",
     "human_score": 0.10},

    {"task": "Write a Python function to check if a number is prime.",
     "output": "def is_prime(n):\n    return True",
     "human_score": 0.08},

    {"task": "Summarize the benefits of version control.",
     "output": "Version control is good.",
     "human_score": 0.05},

    {"task": "Explain the difference between a stack and a queue.",
     "output": "They are both collections.",
     "human_score": 0.10},

    {"task": "What is the time complexity of binary search?",
     "output": "It's fast.",
     "human_score": 0.05},

    {"task": "Write a SQL query to find duplicate emails.",
     "output": "SELECT * FROM users;",
     "human_score": 0.12},

    {"task": "Explain what Docker containers are.",
     "output": "Docker.",
     "human_score": 0.02},

    {"task": "Describe the MVC design pattern.",
     "output": "MVC is a pattern.",
     "human_score": 0.08},

    {"task": "Write a regular expression to match email addresses.",
     "output": "email",
     "human_score": 0.03},

    {"task": "What is the CAP theorem?",
     "output": "Something about databases.",
     "human_score": 0.10},
]


# ─────────────────────────────────────────────────────────────────────
# Scoring helpers
# ─────────────────────────────────────────────────────────────────────

def _pearson(xs: list[float], ys: list[float]) -> float:
    """Pearson correlation coefficient, mapped to [0, 1]."""
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return 0.0
    r = cov / (sx * sy)
    # Map from [-1, 1] to [0, 1]  (negative correlation is bad)
    return max(0.0, (r + 1.0) / 2.0)


def _mae(xs: list[float], ys: list[float]) -> float:
    """Mean absolute error."""
    return sum(abs(x - y) for x, y in zip(xs, ys)) / len(xs) if xs else 0.0


def _kendall_tau(xs: list[float], ys: list[float]) -> float:
    """Kendall rank correlation, mapped to [0, 1]."""
    n = len(xs)
    if n < 2:
        return 0.5
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            if dx * dy > 0:
                concordant += 1
            elif dx * dy < 0:
                discordant += 1
            # ties are ignored
    total = concordant + discordant
    if total == 0:
        return 0.5
    tau = (concordant - discordant) / total
    # Map from [-1, 1] to [0, 1]
    return max(0.0, (tau + 1.0) / 2.0)


def run_benchmark() -> float:
    grader = AutoGrader()

    auto_scores: list[float] = []
    human_scores: list[float] = []

    for case in CASES:
        rubric = case.get("rubric", STANDARD_RUBRIC)
        result = grader.grade(case["output"], case["task"], rubric)
        auto_scores.append(max(0.0, min(1.0, result.get("score", 0.0))))
        human_scores.append(case["human_score"])

    corr = _pearson(auto_scores, human_scores)
    mae = _mae(auto_scores, human_scores)
    tau = _kendall_tau(auto_scores, human_scores)

    score = 0.4 * corr + 0.3 * max(0.0, 1.0 - mae) + 0.3 * tau

    print(f"  pearson_corr      : {corr:.6f}")
    print(f"  mae               : {mae:.6f}")
    print(f"  kendall_tau       : {tau:.6f}")
    return score


if __name__ == "__main__":
    final = run_benchmark()
    print(f"score: {final:.6f}")
