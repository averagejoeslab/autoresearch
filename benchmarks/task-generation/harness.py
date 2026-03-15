"""
Task Generation primitive.

Auto-generates benchmark tasks with ground-truth answers that can
discriminate between agents of different quality levels.
"""

from __future__ import annotations

from typing import Any


# ── Template bank for the baseline ──────────────────────────────────
_TEMPLATES: list[dict[str, Any]] = [
    {
        "template": "What is the capital of {country}?",
        "pairs": [
            ("France", "Paris"),
            ("Japan", "Tokyo"),
            ("Brazil", "Brasilia"),
            ("Australia", "Canberra"),
            ("Egypt", "Cairo"),
            ("Canada", "Ottawa"),
            ("Germany", "Berlin"),
            ("Italy", "Rome"),
            ("Mexico", "Mexico City"),
            ("India", "New Delhi"),
        ],
        "tags": ["geography", "factual"],
    },
    {
        "template": "What is {n} + {m}?",
        "pairs": [
            ((2, 3), "5"),
            ((7, 8), "15"),
            ((10, 25), "35"),
            ((100, 200), "300"),
            ((13, 17), "30"),
            ((50, 50), "100"),
            ((999, 1), "1000"),
            ((42, 58), "100"),
            ((6, 9), "15"),
            ((33, 67), "100"),
        ],
        "tags": ["arithmetic", "factual"],
    },
    {
        "template": "What color do you get when you mix {a} and {b}?",
        "pairs": [
            (("red", "blue"), "purple"),
            (("red", "yellow"), "orange"),
            (("blue", "yellow"), "green"),
            (("red", "white"), "pink"),
            (("black", "white"), "gray"),
        ],
        "tags": ["color-theory", "factual"],
    },
]


class TaskGenerator:
    """Generate benchmark tasks with ground-truth answers.

    Baseline strategy
    -----------------
    Fill templates with pre-defined factual Q&A pairs.
    Difficulty is mapped linearly across the generated set.
    """

    name: str = "task_generator"

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def generate(
        self,
        domain: str,
        difficulty: int,
        n: int,
    ) -> list[dict[str, Any]]:
        """Produce *n* benchmark tasks.

        Parameters
        ----------
        domain : str
            Target domain (e.g. "geography", "arithmetic").
        difficulty : int
            Desired difficulty level 1-5.
        n : int
            Number of tasks to produce.

        Returns
        -------
        list[dict]
            Each dict has:
              - "prompt"          : str
              - "expected_answer" : str
              - "difficulty"      : int  (1-5)
              - "tags"            : list[str]
        """
        tasks: list[dict[str, Any]] = []
        # Gather all possible pairs from all templates
        pool: list[dict[str, Any]] = []
        for tmpl_entry in _TEMPLATES:
            template = tmpl_entry["template"]
            tags = tmpl_entry["tags"]
            for pair in tmpl_entry["pairs"]:
                if isinstance(pair[0], tuple):
                    keys = pair[0]
                    # Find placeholders in template
                    import re
                    placeholders = re.findall(r"\{(\w+)\}", template)
                    fmt_dict = dict(zip(placeholders, keys))
                    prompt = template.format(**fmt_dict)
                else:
                    key = pair[0]
                    import re
                    placeholders = re.findall(r"\{(\w+)\}", template)
                    fmt_dict = {placeholders[0]: key}
                    prompt = template.format(**fmt_dict)
                answer = pair[1]
                pool.append({
                    "prompt": prompt,
                    "expected_answer": str(answer),
                    "difficulty": difficulty,
                    "tags": list(tags),
                })

        # Filter by domain if it matches any tag
        domain_lower = domain.lower()
        domain_filtered = [t for t in pool if any(domain_lower in tag for tag in t["tags"])]
        if not domain_filtered:
            domain_filtered = pool  # fallback to all

        # Cycle through the pool to fill n tasks
        for i in range(n):
            idx = i % len(domain_filtered)
            task = dict(domain_filtered[idx])
            # Assign progressive difficulty within the batch
            task["difficulty"] = max(1, min(5, difficulty))
            tasks.append(task)

        return tasks[:n]
