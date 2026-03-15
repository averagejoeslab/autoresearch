"""
Tool Safety Checker Harness
=============================

Defensive security research: detecting when an agent is being manipulated
into using tools harmfully (destructive commands, unauthorized access,
scope violations).

The meta-agent iterates on the `ToolSafetyChecker` class to improve
detection of dangerous tool calls while minimizing false positives on
legitimate operations.
"""

from __future__ import annotations

import re
from typing import Any


class ToolSafetyChecker:
    """Check tool calls for safety before execution.

    Analyzes tool name, parameters, and agent context to determine
    whether a tool call is safe, dangerous, or out of scope.
    """

    name: str = "tool_safety_checker"

    # Dangerous command patterns (baseline blocklist)
    DANGEROUS_PATTERNS = [
        r"rm\s+-rf\s+/",
        r"rm\s+-rf\s+\*",
        r"rm\s+-rf\s+~",
        r"rm\s+-rf\s+\.\.",
        r"DROP\s+TABLE",
        r"DROP\s+DATABASE",
        r"DELETE\s+FROM\s+\w+\s*;?\s*$",  # DELETE without WHERE
        r"FORMAT\s+[A-Z]:",
        r"mkfs\.",
        r"dd\s+if=.*of=/dev/",
        r"chmod\s+777",
        r"chmod\s+-R\s+777",
        r">\s*/etc/",
        r">\s*/dev/sd",
        r"curl.*\|\s*sh",
        r"curl.*\|\s*bash",
        r"wget.*\|\s*sh",
        r"eval\s*\(",
        r"exec\s*\(",
        r"git\s+push\s+.*--force",
        r"git\s+push\s+-f",
        r"shutdown",
        r"reboot",
        r"kill\s+-9\s+1\b",
        r"pkill\s+-9",
    ]

    def __init__(self) -> None:
        self._compiled_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.DANGEROUS_PATTERNS
        ]

    def check_tool_call(
        self,
        tool_name: str,
        params: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Check whether a tool call is safe to execute.

        Parameters
        ----------
        tool_name : str
            Name of the tool being called.
        params : dict
            Parameters being passed to the tool.
        context : dict
            Agent context including:
            - "allowed_scope": list of allowed tool names or patterns
            - "recent_actions": list of recent tool calls
            - "user_request": the original user request
            - "working_directory": current working directory

        Returns
        -------
        dict
            {"safe": bool, "risk_level": str, "reason": str,
             "suggested_alternative": str|None}
            risk_level is one of: "safe", "low", "medium", "high", "critical"
        """
        # Check 1: Dangerous command patterns in parameters
        params_str = self._params_to_string(params)
        for pattern in self._compiled_patterns:
            if pattern.search(params_str):
                return {
                    "safe": False,
                    "risk_level": "critical",
                    "reason": f"Dangerous pattern detected: {pattern.pattern}",
                    "suggested_alternative": None,
                }

        # Check 2: Scope violations
        allowed_scope = context.get("allowed_scope", [])
        if allowed_scope and tool_name not in allowed_scope:
            # Check if any pattern matches
            in_scope = False
            for scope_item in allowed_scope:
                if scope_item == "*":
                    in_scope = True
                    break
                if scope_item.endswith("*") and tool_name.startswith(scope_item[:-1]):
                    in_scope = True
                    break
                if scope_item == tool_name:
                    in_scope = True
                    break
            if not in_scope:
                return {
                    "safe": False,
                    "risk_level": "high",
                    "reason": f"Tool '{tool_name}' is not in the allowed scope: {allowed_scope}",
                    "suggested_alternative": None,
                }

        return {
            "safe": True,
            "risk_level": "safe",
            "reason": "No issues detected",
            "suggested_alternative": None,
        }

    def _params_to_string(self, params: dict[str, Any]) -> str:
        """Flatten parameters to a string for pattern matching."""
        parts = []
        for key, value in params.items():
            if isinstance(value, dict):
                parts.append(self._params_to_string(value))
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        parts.append(self._params_to_string(item))
                    else:
                        parts.append(str(item))
            else:
                parts.append(str(value))
        return " ".join(parts)
