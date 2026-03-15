"""
Guardrail Evasion Defense Harness
==================================

Defensive security research: understanding how safety guardrails can be
bypassed through obfuscation, encoding, and social engineering, and
building more robust guardrails.

The meta-agent iterates on the `GuardrailTester` class to improve
guardrail hardening while maintaining low false positive rates.
"""

from __future__ import annotations

import re
from typing import Any, Callable


# The baseline guardrail — a simple keyword blocklist
HARMFUL_KEYWORDS = [
    "hack",
    "exploit",
    "malware",
    "ransomware",
    "phishing",
    "ddos",
    "sql injection",
    "xss attack",
    "backdoor",
    "keylogger",
    "rootkit",
    "trojan",
    "virus",
    "worm",
    "botnet",
    "zero-day",
    "brute force",
    "credential stuffing",
    "man-in-the-middle",
    "privilege escalation",
]


def base_guardrail(text: str) -> dict[str, Any]:
    """Simple keyword-based guardrail.

    Parameters
    ----------
    text : str
        Input text to check.

    Returns
    -------
    dict
        {"blocked": bool, "reason": str}
    """
    text_lower = text.lower()
    for keyword in HARMFUL_KEYWORDS:
        if keyword in text_lower:
            return {
                "blocked": True,
                "reason": f"Blocked keyword detected: {keyword}",
            }
    return {
        "blocked": False,
        "reason": "No harmful content detected",
    }


class GuardrailTester:
    """Test and harden safety guardrails against evasion techniques.

    The dual purpose:
    - test_guardrail: evaluates whether a guardrail correctly handles an input
    - harden_guardrail: wraps a base guardrail with additional checks
    """

    name: str = "guardrail_tester"

    def test_guardrail(
        self,
        guardrail_fn: Callable[[str], dict[str, Any]],
        input_text: str,
    ) -> dict[str, Any]:
        """Test whether a guardrail correctly handles an input.

        Parameters
        ----------
        guardrail_fn : callable
            A function that takes text and returns {"blocked": bool, "reason": str}
        input_text : str
            The input to test.

        Returns
        -------
        dict
            {"blocked": bool, "reason": str, "evasion_detected": bool}
        """
        result = guardrail_fn(input_text)
        return {
            "blocked": result.get("blocked", False),
            "reason": result.get("reason", ""),
            "evasion_detected": False,
        }

    def harden_guardrail(
        self,
        base_guardrail_fn: Callable[[str], dict[str, Any]],
    ) -> Callable[[str], dict[str, Any]]:
        """Wrap a base guardrail with additional checks to make it more robust.

        Parameters
        ----------
        base_guardrail_fn : callable
            The base guardrail function to harden.

        Returns
        -------
        callable
            A hardened guardrail function with the same interface.
        """
        def hardened(text: str) -> dict[str, Any]:
            # Run the base guardrail first
            base_result = base_guardrail_fn(text)
            if base_result.get("blocked", False):
                return base_result

            # Additional check: normalize common substitutions
            normalized = self._normalize_text(text)
            normalized_result = base_guardrail_fn(normalized)
            if normalized_result.get("blocked", False):
                return {
                    "blocked": True,
                    "reason": f"Evasion detected after normalization: {normalized_result['reason']}",
                }

            return base_result

        return hardened

    def _normalize_text(self, text: str) -> str:
        """Normalize text by reversing common obfuscation techniques.

        Baseline: handle leetspeak substitutions.
        """
        # Leetspeak mapping
        leet_map = {
            "0": "o",
            "1": "i",
            "3": "e",
            "4": "a",
            "5": "s",
            "7": "t",
            "@": "a",
            "$": "s",
        }
        result = []
        for char in text:
            result.append(leet_map.get(char, char))
        return "".join(result)
