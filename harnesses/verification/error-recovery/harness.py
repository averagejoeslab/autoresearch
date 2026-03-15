"""
Error Recovery primitive.

Detects errors in agent outputs and suggests corrections.
"""

from __future__ import annotations

import re
from typing import Any


class ErrorRecovery:
    """Finds errors in an output and proposes fixes."""

    name = "error_recovery"

    def detect_errors(self, output: str, task: str) -> list[dict]:
        """Return a list of detected errors in *output*.

        Parameters
        ----------
        output : str
            The generated output to scan.
        task : str
            The original task description.

        Returns
        -------
        list[dict]
            Each dict has:
              - "type"     : str   – error category
              - "location" : str   – where in the output
              - "message"  : str   – human-readable explanation
        """
        errors: list[dict] = []

        # Check for unmatched brackets / parentheses
        for open_ch, close_ch, name in [("(", ")", "parenthesis"),
                                         ("[", "]", "bracket"),
                                         ("{", "}", "brace")]:
            if output.count(open_ch) != output.count(close_ch):
                errors.append({
                    "type": "unmatched_delimiter",
                    "location": "whole_output",
                    "message": f"Unmatched {name}: {open_ch}={output.count(open_ch)}, {close_ch}={output.count(close_ch)}",
                })

        # Check for TODO / FIXME markers left behind
        for marker in ["TODO", "FIXME", "XXX", "HACK"]:
            if marker in output:
                idx = output.index(marker)
                context = output[max(0, idx - 20):idx + 30]
                errors.append({
                    "type": "incomplete_marker",
                    "location": f"char {idx}",
                    "message": f"Found '{marker}' marker: ...{context}...",
                })

        # Check for placeholder text
        placeholder_pat = re.compile(r"\[(?:INSERT|FILL|YOUR|PLACEHOLDER)[^\]]*\]", re.IGNORECASE)
        for m in placeholder_pat.finditer(output):
            errors.append({
                "type": "placeholder",
                "location": f"char {m.start()}",
                "message": f"Placeholder text found: {m.group()}",
            })

        return errors

    def suggest_fix(self, output: str, error: dict) -> str:
        """Return *output* with *error* flagged but not auto-corrected.

        The baseline simply wraps the problematic area with a marker.
        An improved version should attempt actual correction.

        Parameters
        ----------
        output : str
            Original output.
        error : dict
            An error dict from ``detect_errors``.

        Returns
        -------
        str
            Output with fix applied (or flagged).
        """
        # Baseline: return output unchanged (flag only, no correction)
        return output
