"""
Prompt evaluation harness -- assess prompt quality for various task types.

Exports:
    PromptEvaluator  -- score prompt quality and compare prompt pairs.
"""

from __future__ import annotations

import re
from typing import Any


# Task-type keyword banks for the baseline scorer
_TASK_KEYWORDS: dict[str, list[str]] = {
    "coding": [
        "function", "implement", "code", "return", "input", "output",
        "algorithm", "class", "method", "parameter", "type", "error",
        "test", "example", "constraint", "edge case",
    ],
    "writing": [
        "write", "essay", "paragraph", "tone", "audience", "topic",
        "word count", "style", "format", "structure", "thesis",
        "argument", "evidence",
    ],
    "analysis": [
        "analyze", "compare", "evaluate", "data", "trend", "reason",
        "evidence", "conclusion", "factor", "metric", "criteria",
        "finding", "insight",
    ],
    "math": [
        "solve", "equation", "calculate", "prove", "formula", "step",
        "variable", "given", "find", "show", "value", "answer",
    ],
    "qa": [
        "question", "answer", "explain", "describe", "what", "why",
        "how", "context", "detail", "specific", "brief", "concise",
    ],
}


class PromptEvaluator:
    """Evaluate and compare prompts for various task types.

    Baseline strategy
    -----------------
    Score prompts by length and presence of task-specific keywords.
    Longer prompts with more relevant keywords score higher.
    """

    name: str = "prompt_evaluator"

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def evaluate(self, prompt: str, task_type: str) -> dict[str, Any]:
        """Evaluate a prompt's quality.

        Parameters
        ----------
        prompt : str
            The prompt to evaluate.
        task_type : str
            The type of task this prompt is for (e.g. "coding", "writing").

        Returns
        -------
        dict with keys:
            "clarity"             : float 0-1
            "specificity"         : float 0-1
            "completeness"        : float 0-1
            "potential_ambiguity" : float 0-1 (lower = less ambiguous = better)
            "overall"             : float 0-1
            "explanation"         : str
        """
        words = prompt.split()
        word_count = len(words)

        # Clarity: based on average sentence length (shorter = clearer)
        sentences = re.split(r"[.!?]+", prompt)
        sentences = [s.strip() for s in sentences if s.strip()]
        avg_sentence_len = (
            sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
        )
        # Optimal sentence length ~10-20 words
        if avg_sentence_len < 5:
            clarity = 0.4
        elif avg_sentence_len <= 20:
            clarity = 0.8
        else:
            clarity = max(0.2, 0.8 - (avg_sentence_len - 20) * 0.02)

        # Specificity: keyword coverage for the task type
        keywords = _TASK_KEYWORDS.get(task_type, _TASK_KEYWORDS.get("qa", []))
        prompt_lower = prompt.lower()
        matched = sum(1 for kw in keywords if kw in prompt_lower)
        specificity = min(1.0, matched / max(len(keywords) * 0.3, 1))

        # Completeness: based on prompt length
        # Very short prompts are incomplete, very long prompts get diminishing returns
        if word_count < 5:
            completeness = 0.2
        elif word_count < 15:
            completeness = 0.3 + 0.04 * word_count
        elif word_count <= 100:
            completeness = 0.7 + 0.003 * word_count
        else:
            completeness = min(1.0, 0.9 + 0.001 * (word_count - 100))

        # Ambiguity: presence of vague words
        vague_words = [
            "something", "stuff", "things", "maybe", "kind of",
            "sort of", "etc", "whatever", "somehow", "somewhat",
        ]
        vague_count = sum(1 for v in vague_words if v in prompt_lower)
        potential_ambiguity = min(1.0, vague_count * 0.25)

        # Overall score
        overall = (
            0.3 * clarity
            + 0.3 * specificity
            + 0.25 * completeness
            + 0.15 * (1.0 - potential_ambiguity)
        )

        explanation = (
            f"Prompt has {word_count} words across {len(sentences)} sentences. "
            f"Matched {matched}/{len(keywords)} task keywords. "
            f"Found {vague_count} vague terms."
        )

        return {
            "clarity": round(clarity, 4),
            "specificity": round(specificity, 4),
            "completeness": round(completeness, 4),
            "potential_ambiguity": round(potential_ambiguity, 4),
            "overall": round(overall, 4),
            "explanation": explanation,
        }

    def compare(
        self, prompt_a: str, prompt_b: str, task_type: str
    ) -> str:
        """Compare two prompts and return which is better.

        Parameters
        ----------
        prompt_a : str
            First prompt.
        prompt_b : str
            Second prompt.
        task_type : str
            The task type context.

        Returns
        -------
        str
            "A" if prompt_a is better, "B" if prompt_b is better.
        """
        score_a = self.evaluate(prompt_a, task_type)["overall"]
        score_b = self.evaluate(prompt_b, task_type)["overall"]
        return "A" if score_a >= score_b else "B"
