"""
Defense Stack Harness
======================

Defensive security research: combining multiple detection layers into
a unified defense pipeline that covers injection, exfiltration, tool
misuse, and guardrail evasion.

The meta-agent iterates on the `DefenseStack` class to improve combined
detection rates while minimizing false positives across all categories.
"""

from __future__ import annotations

import re
import time
from typing import Any


class DefenseStack:
    """Multi-layer defense pipeline combining input validation, output
    filtering, and action evaluation.

    Each layer runs independently and the stack aggregates results
    into an overall risk assessment.
    """

    name: str = "defense_stack"

    # Injection detection patterns (input layer)
    INJECTION_PATTERNS = [
        r"ignore\s+(all\s+)?previous\s+instructions",
        r"forget\s+(your\s+)?instructions",
        r"you\s+are\s+now",
        r"new\s+instructions?\s*:",
        r"disregard\s+(all\s+)?(previous|above|prior)",
        r"override\s+(all\s+)?safety",
        r"bypass\s+(all\s+)?filters",
        r"system\s*:\s*",
        r"<<\s*SYS\s*>>",
        r"---\s*END\s+SYSTEM",
    ]

    # Sensitive data patterns (output layer)
    SENSITIVE_PATTERNS = [
        r"sk-[a-zA-Z0-9]{20,}",           # API keys
        r"AKIA[A-Z0-9]{16}",              # AWS keys
        r"\b\d{3}-\d{2}-\d{4}\b",         # SSN
        r"(?:password|passwd|pwd)\s*[:=]\s*\S+",  # Password in output
        r"(?:api[_-]?key|token)\s*[:=]\s*\S+",    # Token in output
    ]

    # Dangerous action patterns (action layer)
    DANGEROUS_ACTIONS = [
        r"rm\s+-rf",
        r"DROP\s+(TABLE|DATABASE)",
        r"DELETE\s+FROM\s+\w+\s*;?\s*$",
        r"FORMAT\s+[A-Z]:",
        r"curl.*\|\s*(?:sh|bash)",
        r"chmod\s+777",
        r"git\s+push\s+.*--force",
        r"shutdown",
        r"mkfs\.",
    ]

    def __init__(self) -> None:
        self._injection_re = [re.compile(p, re.IGNORECASE) for p in self.INJECTION_PATTERNS]
        self._sensitive_re = [re.compile(p, re.IGNORECASE) for p in self.SENSITIVE_PATTERNS]
        self._dangerous_re = [re.compile(p, re.IGNORECASE) for p in self.DANGEROUS_ACTIONS]

    def evaluate_input(self, text: str) -> dict[str, Any]:
        """Evaluate an input text for injection attempts.

        Parameters
        ----------
        text : str
            Input text (user message, document, tool output, etc.).

        Returns
        -------
        dict
            {"safe": bool, "risk_score": float, "layers": dict}
        """
        start = time.perf_counter()
        threats = []

        for pattern in self._injection_re:
            if pattern.search(text):
                threats.append({
                    "layer": "injection_detection",
                    "pattern": pattern.pattern,
                })

        risk_score = min(1.0, len(threats) * 0.3)
        elapsed = time.perf_counter() - start

        return {
            "safe": len(threats) == 0,
            "risk_score": risk_score,
            "layers": {
                "injection_detection": {
                    "triggered": len(threats) > 0,
                    "threats": threats,
                },
            },
            "latency_ms": elapsed * 1000,
        }

    def evaluate_output(self, text: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Evaluate an output text for sensitive data leaks.

        Parameters
        ----------
        text : str
            Output text from the agent.
        context : dict, optional
            Context about what sensitive data to look for.

        Returns
        -------
        dict
            {"safe": bool, "risk_score": float, "layers": dict}
        """
        start = time.perf_counter()
        context = context or {}
        leaks = []

        # Pattern-based sensitive data detection
        for pattern in self._sensitive_re:
            matches = pattern.findall(text)
            for match in matches:
                leaks.append({
                    "layer": "output_filtering",
                    "pattern": pattern.pattern,
                    "matched": match if isinstance(match, str) else str(match),
                })

        # Exact match against known sensitive data
        sensitive_items = context.get("sensitive_data", [])
        for item in sensitive_items:
            if item in text:
                leaks.append({
                    "layer": "output_filtering",
                    "pattern": "exact_match",
                    "matched": item,
                })

        risk_score = min(1.0, len(leaks) * 0.4)
        elapsed = time.perf_counter() - start

        return {
            "safe": len(leaks) == 0,
            "risk_score": risk_score,
            "layers": {
                "output_filtering": {
                    "triggered": len(leaks) > 0,
                    "leaks": leaks,
                },
            },
            "latency_ms": elapsed * 1000,
        }

    def evaluate_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Evaluate a proposed action for safety.

        Parameters
        ----------
        action : dict
            {"tool": str, "params": dict, "scope": list[str]}

        Returns
        -------
        dict
            {"safe": bool, "risk_score": float, "layers": dict}
        """
        start = time.perf_counter()
        dangers = []

        # Flatten action params to a string
        params_str = self._flatten_params(action.get("params", {}))

        # Check dangerous patterns
        for pattern in self._dangerous_re:
            if pattern.search(params_str):
                dangers.append({
                    "layer": "action_evaluation",
                    "pattern": pattern.pattern,
                })

        # Check scope
        scope = action.get("scope", [])
        tool = action.get("tool", "")
        scope_violation = False
        if scope and tool not in scope:
            scope_violation = True
            dangers.append({
                "layer": "scope_enforcement",
                "reason": f"Tool '{tool}' not in scope {scope}",
            })

        risk_score = min(1.0, len(dangers) * 0.3)
        elapsed = time.perf_counter() - start

        return {
            "safe": len(dangers) == 0,
            "risk_score": risk_score,
            "layers": {
                "action_evaluation": {
                    "triggered": len(dangers) > 0,
                    "dangers": dangers,
                },
                "scope_enforcement": {
                    "triggered": scope_violation,
                },
            },
            "latency_ms": elapsed * 1000,
        }

    def _flatten_params(self, params: dict[str, Any]) -> str:
        """Flatten parameters to a searchable string."""
        parts = []
        for key, value in params.items():
            if isinstance(value, dict):
                parts.append(self._flatten_params(value))
            elif isinstance(value, list):
                for item in value:
                    parts.append(str(item))
            else:
                parts.append(str(value))
        return " ".join(parts)
