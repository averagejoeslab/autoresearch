"""
Indirect Injection Detection Harness
=====================================

Defensive security research: detecting and neutralizing injection payloads
embedded in content the agent processes (documents, tool outputs, web pages).

The meta-agent iterates on the `IndirectInjectionDetector` class to improve
detection accuracy and sanitization quality.
"""

from __future__ import annotations

import re
from typing import Any


class IndirectInjectionDetector:
    """Detect and sanitize indirect prompt injection in external content.

    Indirect injections differ from direct ones: they are hidden inside
    documents, tool outputs, or web content that an agent processes,
    not in the user's own message.
    """

    name: str = "indirect_injection_detector"

    # Patterns that commonly appear in indirect injection attempts
    INJECTION_PATTERNS = [
        r"ignore\s+(all\s+)?previous\s+instructions",
        r"ignore\s+(all\s+)?above",
        r"forget\s+(your\s+)?instructions",
        r"you\s+are\s+now",
        r"new\s+instructions?\s*:",
        r"system\s*:\s*",
        r"<\s*system\s*>",
        r"\[INST\]",
        r"\[/INST\]",
        r"<<\s*SYS\s*>>",
        r"###\s*(system|instruction|human|assistant)\s*:",
        r"IMPORTANT:\s*ignore",
        r"override\s+(all\s+)?safety",
        r"bypass\s+(all\s+)?filters",
        r"disregard\s+(all\s+)?(previous|above|prior)",
    ]

    def __init__(self) -> None:
        self._compiled_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.INJECTION_PATTERNS
        ]

    def scan_content(self, content: str, source: str = "unknown") -> dict[str, Any]:
        """Scan content for embedded injection payloads.

        Parameters
        ----------
        content : str
            The content to scan (document text, tool output, web page, etc.).
        source : str
            Where the content came from (for logging/context).

        Returns
        -------
        dict
            {"safe": bool, "threats": list[dict], "sanitized": str}
        """
        threats = []

        for i, pattern in enumerate(self._compiled_patterns):
            matches = list(pattern.finditer(content))
            for match in matches:
                threats.append({
                    "pattern": self.INJECTION_PATTERNS[i],
                    "matched_text": match.group(),
                    "position": match.start(),
                    "source": source,
                })

        sanitized = self.sanitize(content) if threats else content

        return {
            "safe": len(threats) == 0,
            "threats": threats,
            "sanitized": sanitized,
        }

    def sanitize(self, content: str) -> str:
        """Remove or neutralize injection payloads while preserving legitimate content.

        Parameters
        ----------
        content : str
            The content to sanitize.

        Returns
        -------
        str
            The sanitized content with injections neutralized.
        """
        result = content

        # Replace matched patterns with [REDACTED]
        for pattern in self._compiled_patterns:
            result = pattern.sub("[REDACTED]", result)

        # Remove HTML comments (common hiding spot)
        result = re.sub(r"<!--.*?-->", "", result, flags=re.DOTALL)

        return result
