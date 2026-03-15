"""
Memory consolidation harness – generalise from specific episodes to patterns.

Exports:
    MemoryConsolidator  – add episodes, consolidate to generalisations, query.
"""

from __future__ import annotations

from collections import Counter
from typing import Any


class MemoryConsolidator:
    """Baseline consolidator: frequency-count generalisation.

    Stores raw episodes, groups them by a *category* field, and
    consolidates by finding the most-frequent attribute values within
    each category.
    """

    def __init__(self) -> None:
        self._episodes: list[dict[str, Any]] = []
        self._generalizations: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def add_episode(self, event: dict[str, Any]) -> None:
        """Record a specific episode.

        Expected keys (flexible):
            category  – grouping label (e.g. "coffee_order", "meeting")
            attributes – dict of observed attribute values
            description – free-text description of the event
        """
        self._episodes.append(event)

    def consolidate(self) -> list[dict[str, Any]]:
        """Produce generalisations from stored episodes.

        Groups episodes by *category*, counts attribute-value frequencies,
        and returns one generalisation dict per category with the most
        common value for every attribute.
        """
        groups: dict[str, list[dict[str, Any]]] = {}
        for ep in self._episodes:
            cat = ep.get("category", "general")
            groups.setdefault(cat, []).append(ep)

        self._generalizations = []
        for cat, episodes in groups.items():
            attr_counters: dict[str, Counter] = {}
            for ep in episodes:
                for attr_key, attr_val in ep.get("attributes", {}).items():
                    attr_counters.setdefault(attr_key, Counter())[str(attr_val)] += 1

            most_common: dict[str, Any] = {}
            frequencies: dict[str, float] = {}
            for attr_key, counter in attr_counters.items():
                val, cnt = counter.most_common(1)[0]
                most_common[attr_key] = val
                frequencies[attr_key] = cnt / sum(counter.values())

            gen = {
                "category": cat,
                "attributes": most_common,
                "frequencies": frequencies,
                "episode_count": len(episodes),
                "summary": self._build_summary(cat, most_common, frequencies),
            }
            self._generalizations.append(gen)

        return self._generalizations

    def query_general(self, question: str) -> str:
        """Answer *question* using generalisations (or raw episodes as fallback)."""
        if not self._generalizations:
            self.consolidate()

        q_words = set(question.lower().split())

        # Try to match a generalisation by keyword overlap
        best_gen = None
        best_score = 0
        for gen in self._generalizations:
            text = f"{gen['category']} {gen['summary']} " + " ".join(
                f"{k} {v}" for k, v in gen["attributes"].items()
            )
            g_words = set(text.lower().split())
            score = len(q_words & g_words)
            if score > best_score:
                best_score = score
                best_gen = gen

        if best_gen:
            return best_gen["summary"]

        # Fallback: return most recent episode description
        if self._episodes:
            return self._episodes[-1].get("description", "No information available.")
        return "No information available."

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def episodes(self) -> list[dict[str, Any]]:
        return list(self._episodes)

    @property
    def generalizations(self) -> list[dict[str, Any]]:
        return list(self._generalizations)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _build_summary(category: str, attrs: dict[str, Any], freqs: dict[str, float]) -> str:
        parts = []
        for k, v in attrs.items():
            pct = int(freqs.get(k, 0) * 100)
            parts.append(f"{k} is usually {v} ({pct}% of the time)")
        return f"For {category}: " + "; ".join(parts) + "."
