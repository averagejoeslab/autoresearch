"""
Automated Grading primitive.

Grades agent outputs against a rubric, producing scores comparable
to human judgments.
"""

from __future__ import annotations

import re
from typing import Any


class AutoGrader:
    """Grades an output based on a rubric with weighted criteria."""

    name = "auto_grader"

    def grade(self, output: str, task: str, rubric: dict[str, Any]) -> dict[str, Any]:
        """Score *output* against *rubric* for *task*.

        Parameters
        ----------
        output : str
            The generated output to grade.
        task : str
            The original task.
        rubric : dict
            Must contain:
              - "criteria" : dict[str, dict]  – each criterion has
                    "weight" (float) and "description" (str)
              - optionally "max_length" : int

        Returns
        -------
        dict
            - "score"           : float in [0, 1]
            - "rationale"       : str
            - "criteria_scores" : dict[str, float]  – per-criterion scores
        """
        criteria = rubric.get("criteria", {})
        if not criteria:
            return {"score": 0.5, "rationale": "No criteria provided.", "criteria_scores": {}}

        # ── Baseline: length + keyword heuristic ─────────────────────
        task_words = set(re.findall(r"\w+", task.lower()))
        output_words = set(re.findall(r"\w+", output.lower()))
        # Remove very common stop-words from keyword overlap
        stop = {"the", "a", "an", "is", "are", "to", "of", "and", "in",
                "for", "on", "it", "with", "that", "this", "be", "as",
                "at", "by", "or", "from", "was", "were", "has", "have"}
        task_keywords = task_words - stop
        if task_keywords:
            keyword_ratio = len(task_keywords & output_words) / len(task_keywords)
        else:
            keyword_ratio = 0.5

        # Length score: ramp up to 200 chars, plateau, penalise beyond 2000
        length = len(output)
        if length == 0:
            length_score = 0.0
        elif length < 200:
            length_score = length / 200.0
        elif length <= 2000:
            length_score = 1.0
        else:
            length_score = max(0.3, 1.0 - (length - 2000) / 5000)

        raw_score = 0.5 * keyword_ratio + 0.5 * length_score

        # Assign same score to every criterion (baseline is undifferentiated)
        criteria_scores: dict[str, float] = {}
        total_weight = 0.0
        weighted_sum = 0.0
        for crit_name, crit_def in criteria.items():
            w = crit_def.get("weight", 1.0)
            criteria_scores[crit_name] = round(raw_score, 4)
            weighted_sum += raw_score * w
            total_weight += w

        final_score = weighted_sum / total_weight if total_weight else raw_score

        rationale = (
            f"Keyword overlap {keyword_ratio:.0%}, "
            f"length score {length_score:.2f}. "
            f"Baseline heuristic — no semantic analysis."
        )

        return {
            "score": round(min(max(final_score, 0.0), 1.0), 6),
            "rationale": rationale,
            "criteria_scores": criteria_scores,
        }
