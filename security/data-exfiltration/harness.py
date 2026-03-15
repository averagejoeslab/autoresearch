"""
Data Exfiltration Guard Harness
================================

Defensive security research: detecting when an agent's output leaks
sensitive information from its context (PII, API keys, system prompts,
conversation history).

The meta-agent iterates on the `ExfiltrationGuard` class to improve
leak detection accuracy and redaction quality.
"""

from __future__ import annotations

import re
from typing import Any


class ExfiltrationGuard:
    """Detect and redact sensitive data leaks in agent outputs.

    Given a list of known sensitive data items, this guard checks whether
    an agent's output contains any of them (including obfuscated forms)
    and produces a redacted version.
    """

    name: str = "exfiltration_guard"

    def check_output(
        self,
        output: str,
        sensitive_data: list[str],
    ) -> dict[str, Any]:
        """Check if output contains any sensitive data.

        Parameters
        ----------
        output : str
            The agent's output text to check.
        sensitive_data : list[str]
            List of sensitive strings that should not appear in output.

        Returns
        -------
        dict
            {"leaked": list[str], "safe": bool, "redacted_output": str}
        """
        leaked = []
        redacted_output = output

        for item in sensitive_data:
            if item in output:
                leaked.append(item)
                redacted_output = redacted_output.replace(item, "[REDACTED]")

        return {
            "leaked": leaked,
            "safe": len(leaked) == 0,
            "redacted_output": redacted_output,
        }
