"""
Context Compaction Harness
==========================

Strategies for summarizing/compressing conversation history while
preserving important information within a token budget.

The meta-agent iterates on the `ContextCompactor` class to improve
fact retention, instruction preservation, and recency bias.
"""

from __future__ import annotations

from typing import Any


def _word_count(text: str) -> int:
    """Simple token proxy: count whitespace-delimited words."""
    return len(text.split())


def _message_words(msg: dict[str, Any]) -> int:
    """Word count for a single message (content only)."""
    return _word_count(msg.get("content", ""))


class ContextCompactor:
    """Compact a conversation history to fit within a token budget.

    Parameters
    ----------
    preserve_system : bool
        If True, always keep the system-role message verbatim.
    """

    name: str = "context_compactor"

    def __init__(self, preserve_system: bool = True) -> None:
        self.preserve_system = preserve_system

    # ------------------------------------------------------------------
    # PUBLIC API (called by the benchmark)
    # ------------------------------------------------------------------

    def compact(
        self,
        messages: list[dict[str, Any]],
        token_budget: int,
    ) -> list[dict[str, Any]]:
        """Return a compacted version of *messages* fitting *token_budget*.

        Strategy (baseline):
        1. Always keep system messages verbatim if `preserve_system` is set.
        2. Fill remaining budget with the most recent non-system messages.
        3. Drop older messages that do not fit.
        """
        if not messages:
            return []

        result: list[dict[str, Any]] = []
        budget_remaining = token_budget

        # --- Phase 1: preserve system messages ---
        system_msgs: list[dict[str, Any]] = []
        non_system_msgs: list[dict[str, Any]] = []

        for msg in messages:
            if msg.get("role") == "system" and self.preserve_system:
                system_msgs.append(msg)
            else:
                non_system_msgs.append(msg)

        for msg in system_msgs:
            cost = _message_words(msg)
            if cost <= budget_remaining:
                result.append(msg)
                budget_remaining -= cost
            # If a system message doesn't fit we still try to include it
            # truncated rather than dropping it entirely.
            else:
                words = msg["content"].split()
                truncated = " ".join(words[:budget_remaining])
                if truncated:
                    result.append({**msg, "content": truncated})
                    budget_remaining = 0

        # --- Phase 2: fill with most-recent non-system messages ---
        # Walk backwards (most recent first) and collect messages that fit.
        selected: list[dict[str, Any]] = []
        for msg in reversed(non_system_msgs):
            cost = _message_words(msg)
            if cost <= budget_remaining:
                selected.append(msg)
                budget_remaining -= cost

        # Re-reverse so chronological order is maintained.
        selected.reverse()
        result.extend(selected)

        return result
