"""
Memory retrieval harness – storage backends and search strategies for agent memory.

Exports:
    MemoryStore  – store key-value pairs with metadata; retrieve by query.
"""

from __future__ import annotations

import time
from typing import Any


class MemoryStore:
    """Simple keyword-overlap memory store (baseline).

    Stores (key, value, metadata) tuples and retrieves the top-k most
    relevant entries for a given query using word-overlap scoring.
    """

    def __init__(self) -> None:
        self._entries: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def store(self, key: str, value: str, metadata: dict[str, Any] | None = None) -> None:
        """Insert a fact into the store."""
        self._entries.append({
            "key": key,
            "value": value,
            "metadata": metadata or {},
            "stored_at": time.time(),
        })

    def retrieve(self, query: str, k: int = 5, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Return the *k* entries most relevant to *query*.

        Relevance is measured by word-overlap between the query and the
        concatenation of key + value + metadata-values for each entry.

        Optional *filters* dict narrows results to entries whose metadata
        contains all specified key-value pairs.
        """
        query_words = self._tokenize(query)
        if not query_words:
            return []

        candidates = self._entries
        if filters:
            candidates = [
                e for e in candidates
                if all(e["metadata"].get(fk) == fv for fk, fv in filters.items())
            ]

        scored: list[tuple[float, int, dict[str, Any]]] = []
        for idx, entry in enumerate(candidates):
            text = f"{entry['key']} {entry['value']} " + " ".join(
                str(v) for v in entry["metadata"].values()
            )
            entry_words = self._tokenize(text)
            if not entry_words:
                continue
            overlap = len(query_words & entry_words)
            score = overlap / max(len(query_words), 1)
            scored.append((score, idx, entry))

        scored.sort(key=lambda t: (-t[0], t[1]))
        return [
            {"key": e["key"], "value": e["value"], "metadata": e["metadata"], "score": s}
            for s, _, e in scored[:k]
        ]

    def size(self) -> int:
        """Return the number of stored entries."""
        return len(self._entries)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """Lowercase split into word-set."""
        return set(text.lower().split())
