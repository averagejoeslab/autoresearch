"""
Window Packing Harness
======================

Allocates a fixed token budget across named sections (system prompt,
conversation history, tool results, current task) so that the most
important content is included in the context window.

The meta-agent iterates on the `WindowPacker` class to improve
content coverage, priority alignment, and minimum-section guarantees.
"""

from __future__ import annotations

from typing import Any


def _word_count(text: str) -> int:
    """Simple token proxy: count whitespace-delimited words."""
    return len(text.split())


def _section_words(entries: list[dict[str, Any]]) -> int:
    """Total word count across all entries in a section."""
    return sum(_word_count(e.get("content", "")) for e in entries)


# Default budget proportions for each section.
DEFAULT_PROPORTIONS: dict[str, float] = {
    "system": 0.10,
    "history": 0.50,
    "tools": 0.20,
    "current": 0.20,
}

# Priority weights (higher = more important when budget is tight).
SECTION_PRIORITIES: dict[str, float] = {
    "system": 1.0,
    "current": 0.9,
    "history": 0.7,
    "tools": 0.6,
}


class WindowPacker:
    """Pack sections of content into a fixed token budget.

    Parameters
    ----------
    proportions : dict[str, float] | None
        Fraction of total_budget allocated to each section.  Defaults
        to ``DEFAULT_PROPORTIONS``.
    """

    name: str = "window_packer"

    def __init__(
        self,
        proportions: dict[str, float] | None = None,
    ) -> None:
        self.proportions = dict(proportions or DEFAULT_PROPORTIONS)

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def pack(
        self,
        sections: dict[str, list[dict[str, Any]]],
        total_budget: int,
    ) -> dict[str, list[dict[str, Any]]]:
        """Allocate *total_budget* across *sections* and truncate.

        Strategy (baseline — fixed proportions):
        1. Compute each section's slice of the budget via proportions.
        2. For each section, keep as many entries as fit (from the end
           for history, from the start for everything else).
        3. Any unused budget is NOT redistributed (the meta-agent can
           improve this).
        """
        packed: dict[str, list[dict[str, Any]]] = {}

        for section_name, entries in sections.items():
            proportion = self.proportions.get(section_name, 0.0)
            section_budget = int(total_budget * proportion)

            if not entries or section_budget <= 0:
                packed[section_name] = []
                continue

            packed[section_name] = self._fill_section(
                entries, section_budget, prefer_recent=(section_name == "history")
            )

        return packed

    # ------------------------------------------------------------------
    # INTERNALS
    # ------------------------------------------------------------------

    @staticmethod
    def _fill_section(
        entries: list[dict[str, Any]],
        budget: int,
        prefer_recent: bool = False,
    ) -> list[dict[str, Any]]:
        """Select entries that fit within *budget* words.

        If *prefer_recent*, iterate from the end (most recent first).
        """
        ordered = list(reversed(entries)) if prefer_recent else list(entries)
        selected: list[dict[str, Any]] = []
        remaining = budget

        for entry in ordered:
            cost = _word_count(entry.get("content", ""))
            if cost <= remaining:
                selected.append(entry)
                remaining -= cost

        if prefer_recent:
            selected.reverse()

        return selected
