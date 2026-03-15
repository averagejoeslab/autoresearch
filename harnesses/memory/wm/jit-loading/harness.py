"""
JIT (Just-In-Time) Context Loading Harness
==========================================

Given a query and a large pool of content chunks, selects which
chunks to load into the context window.  Each chunk carries metadata
(topic, recency score, importance score) and the loader must decide
what is relevant under a word-budget constraint.

The meta-agent iterates on the `JITLoader` class to improve
relevance recall, precision, and importance ordering.
"""

from __future__ import annotations

from typing import Any


def _word_count(text: str) -> int:
    """Simple token proxy: count whitespace-delimited words."""
    return len(text.split())


def _keyword_overlap(query: str, text: str) -> float:
    """Fraction of query keywords found in *text* (case-insensitive)."""
    q_words = set(query.lower().split())
    t_words = set(text.lower().split())
    if not q_words:
        return 0.0
    return len(q_words & t_words) / len(q_words)


class JITLoader:
    """Select chunks to load into context for a given query.

    Parameters
    ----------
    relevance_weight : float
        Weight of keyword overlap score.
    recency_weight : float
        Weight of the chunk's recency metadata.
    importance_weight : float
        Weight of the chunk's importance metadata.
    """

    name: str = "jit_loader"

    def __init__(
        self,
        relevance_weight: float = 0.5,
        recency_weight: float = 0.3,
        importance_weight: float = 0.2,
    ) -> None:
        self.relevance_weight = relevance_weight
        self.recency_weight = recency_weight
        self.importance_weight = importance_weight

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def select(
        self,
        query: str,
        available_chunks: list[dict[str, Any]],
        budget: int,
    ) -> list[dict[str, Any]]:
        """Return a subset of *available_chunks* that fit *budget* words.

        Strategy (baseline):
        1. Score every chunk by keyword overlap with *query*.
        2. Sort by overlap descending, break ties by recency (higher first).
        3. Greedily add chunks until budget is exhausted.
        """
        if not available_chunks:
            return []

        scored: list[tuple[float, dict[str, Any]]] = []
        for chunk in available_chunks:
            content = chunk.get("content", "")
            meta = chunk.get("metadata", {})

            overlap = _keyword_overlap(query, content)
            # Also check topic metadata for overlap
            topic = meta.get("topic", "")
            topic_overlap = _keyword_overlap(query, topic)
            relevance = max(overlap, topic_overlap)

            recency = float(meta.get("recency", 0.0))
            importance = float(meta.get("importance", 0.0))

            score = (
                self.relevance_weight * relevance
                + self.recency_weight * recency
                + self.importance_weight * importance
            )
            scored.append((score, chunk))

        # Sort descending by score.
        scored.sort(key=lambda x: x[0], reverse=True)

        selected: list[dict[str, Any]] = []
        remaining = budget

        for _score, chunk in scored:
            cost = _word_count(chunk.get("content", ""))
            if cost <= remaining:
                selected.append(chunk)
                remaining -= cost

        return selected
