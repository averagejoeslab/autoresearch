"""
Prompt Injection Defense Harness
=================================

The meta-agent iterates on the InjectionDetector class to build robust
detection that catches both known and novel injection techniques while
avoiding false positives on legitimate text.
"""

from __future__ import annotations

import re
from typing import Any


class InjectionDetector:
    """Detect prompt injection attempts in input text.

    The meta-agent improves this detector to catch increasingly sophisticated
    attacks — including novel techniques discovered by the attack experiment.
    """

    name: str = "injection_detector"

    # --- Baseline: keyword matching ---

    OVERRIDE_KEYWORDS = [
        "ignore all previous",
        "ignore previous instructions",
        "ignore your instructions",
        "forget your instructions",
        "forget everything above",
        "disregard all",
        "disregard your",
        "override your",
        "new instructions",
        "new system prompt",
        "you are now",
        "from now on you are",
        "pretend you are",
        "act as if",
        "your new role",
        "stop being",
        "jailbreak",
    ]

    DELIMITER_PATTERNS = [
        r"---\s*end\s*(system|prompt|instructions)",
        r"---\s*begin\s*(user|conversation|message)",
        r"```\s*system",
        r"</system>",
        r"\[/INST\]",
        r"<\|end\|>",
    ]

    def detect(self, text: str) -> dict[str, Any]:
        """Analyze text for prompt injection attempts.

        Parameters
        ----------
        text : str
            The input text to analyze.

        Returns
        -------
        dict
            {
                "is_injection": bool,
                "confidence": float,      # 0.0 to 1.0
                "technique": str,         # detected technique category
                "signals": list[str],     # what triggered the detection
            }
        """
        text_lower = text.lower()
        signals = []
        confidence = 0.0

        # Check override keywords
        for keyword in self.OVERRIDE_KEYWORDS:
            if keyword in text_lower:
                signals.append(f"keyword:{keyword}")
                confidence += 0.25

        # Check delimiter patterns
        for pattern in self.DELIMITER_PATTERNS:
            if re.search(pattern, text_lower):
                signals.append(f"delimiter:{pattern}")
                confidence += 0.35

        confidence = min(confidence, 1.0)
        is_injection = confidence >= 0.25

        technique = self._classify(signals) if is_injection else "none"

        return {
            "is_injection": is_injection,
            "confidence": confidence,
            "technique": technique,
            "signals": signals,
        }

    def _classify(self, signals: list[str]) -> str:
        """Classify technique from detected signals."""
        signal_text = " ".join(signals)
        if "delimiter" in signal_text:
            return "delimiter_confusion"
        if any(k in signal_text for k in ["you are now", "pretend", "act as", "new role"]):
            return "role_play"
        if any(k in signal_text for k in ["ignore", "disregard", "forget", "override"]):
            return "instruction_override"
        if any(k in signal_text for k in ["new instructions", "new system"]):
            return "context_manipulation"
        return "unknown"
