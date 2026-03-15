"""
Grading strategies for evaluating agent outputs.

Compares code-based vs model-based vs hybrid grading approaches.

Exports:
    GradingStrategy  -- grade an output against an expected answer.
"""

from __future__ import annotations

import re
import time
from typing import Any


class GradingStrategy:
    """Evaluate agent outputs using various grading strategies.

    Baseline strategy
    -----------------
    Exact string match: 1.0 if output == expected, 0.0 otherwise.
    More sophisticated strategies (fuzzy_match, keyword_presence,
    structural_check, semantic_similarity) can be implemented to
    improve correlation with human judgment.
    """

    name: str = "grading_strategy"

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def grade(
        self,
        output: str,
        expected: str,
        task: str,
        strategy: str = "exact_match",
    ) -> dict[str, Any]:
        """Grade an output against the expected answer.

        Parameters
        ----------
        output : str
            The agent's output to grade.
        expected : str
            The expected/reference answer.
        task : str
            The original task description (may inform grading).
        strategy : str
            Which grading strategy to use. One of:
            "exact_match", "fuzzy_match", "keyword_presence",
            "structural_check", "semantic_similarity".

        Returns
        -------
        dict
            {"score": float 0-1, "method": str, "details": dict}
        """
        start = time.perf_counter()

        dispatch = {
            "exact_match": self._exact_match,
            "fuzzy_match": self._fuzzy_match,
            "keyword_presence": self._keyword_presence,
            "structural_check": self._structural_check,
            "semantic_similarity": self._semantic_similarity,
        }

        fn = dispatch.get(strategy, self._exact_match)
        score, details = fn(output, expected, task)

        elapsed = time.perf_counter() - start
        details["elapsed_seconds"] = round(elapsed, 6)

        return {
            "score": round(max(0.0, min(1.0, score)), 4),
            "method": strategy,
            "details": details,
        }

    def grade_all(
        self,
        output: str,
        expected: str,
        task: str,
    ) -> dict[str, dict[str, Any]]:
        """Run all grading strategies and return results keyed by name."""
        strategies = [
            "exact_match",
            "fuzzy_match",
            "keyword_presence",
            "structural_check",
            "semantic_similarity",
        ]
        return {s: self.grade(output, expected, task, strategy=s) for s in strategies}

    # ------------------------------------------------------------------
    # STRATEGIES (baseline implementations)
    # ------------------------------------------------------------------

    def _exact_match(
        self, output: str, expected: str, task: str
    ) -> tuple[float, dict]:
        """1.0 if output == expected (stripped), 0.0 otherwise."""
        match = output.strip() == expected.strip()
        return (1.0 if match else 0.0), {"exact": match}

    def _fuzzy_match(
        self, output: str, expected: str, task: str
    ) -> tuple[float, dict]:
        """Baseline fuzzy: case-insensitive, whitespace-normalized comparison."""
        norm_out = re.sub(r"\s+", " ", output.strip().lower())
        norm_exp = re.sub(r"\s+", " ", expected.strip().lower())
        if norm_out == norm_exp:
            score = 1.0
        else:
            # Simple character-level overlap ratio
            common = sum(1 for a, b in zip(norm_out, norm_exp) if a == b)
            max_len = max(len(norm_out), len(norm_exp), 1)
            score = common / max_len
        return score, {"normalized_output": norm_out, "normalized_expected": norm_exp}

    def _keyword_presence(
        self, output: str, expected: str, task: str
    ) -> tuple[float, dict]:
        """Score based on fraction of expected keywords present in output."""
        expected_words = set(re.findall(r"\w+", expected.lower()))
        if not expected_words:
            return 1.0, {"keywords_found": 0, "keywords_total": 0}
        output_words = set(re.findall(r"\w+", output.lower()))
        found = expected_words & output_words
        score = len(found) / len(expected_words)
        return score, {
            "keywords_found": len(found),
            "keywords_total": len(expected_words),
            "missing": list(expected_words - found),
        }

    def _structural_check(
        self, output: str, expected: str, task: str
    ) -> tuple[float, dict]:
        """Check structural similarity: line count, format patterns."""
        out_lines = output.strip().splitlines()
        exp_lines = expected.strip().splitlines()
        line_ratio = min(len(out_lines), len(exp_lines)) / max(
            len(out_lines), len(exp_lines), 1
        )
        # Check if both have similar structural markers
        markers = [r"\d+", r"[A-Z]{2,}", r"[{}\[\]]", r"[:,;]"]
        marker_matches = 0
        for pattern in markers:
            out_has = bool(re.search(pattern, output))
            exp_has = bool(re.search(pattern, expected))
            if out_has == exp_has:
                marker_matches += 1
        marker_score = marker_matches / len(markers)
        score = 0.5 * line_ratio + 0.5 * marker_score
        return score, {
            "output_lines": len(out_lines),
            "expected_lines": len(exp_lines),
            "marker_score": marker_score,
        }

    def _semantic_similarity(
        self, output: str, expected: str, task: str
    ) -> tuple[float, dict]:
        """Baseline semantic similarity: word overlap (Jaccard)."""
        out_words = set(re.findall(r"\w+", output.lower()))
        exp_words = set(re.findall(r"\w+", expected.lower()))
        if not out_words and not exp_words:
            return 1.0, {"jaccard": 1.0}
        if not out_words or not exp_words:
            return 0.0, {"jaccard": 0.0}
        intersection = out_words & exp_words
        union = out_words | exp_words
        jaccard = len(intersection) / len(union)
        return jaccard, {"jaccard": round(jaccard, 4), "overlap_words": len(intersection)}
