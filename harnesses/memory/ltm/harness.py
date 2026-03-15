"""
Long-Term Memory system for an AI agent.

The meta-agent iterates on the ENTIRE architecture — storage format,
indexing strategy, retrieval algorithm, consolidation logic, pruning policy.
Everything is fair game.

Exports:
    LongTermMemory  – complete LTM with store / retrieve / consolidate / prune.
"""

from __future__ import annotations

import time
import uuid
from typing import Any


class LongTermMemory:
    """Complete long-term memory system for an AI agent.

    The meta-agent iterates on the entire architecture — storage format,
    indexing strategy, retrieval algorithm, consolidation strategy, pruning
    policy, memory linking, deduplication.

    Architecture dimensions (ALL mutable):
      1. STORAGE_FORMAT       – flat list (baseline)
      2. INDEX_STRATEGY       – linear scan (baseline)
      3. RETRIEVAL_SCORING    – keyword overlap (baseline)
      4. CONSOLIDATION_METHOD – frequency counting (baseline)
      5. PRUNING_POLICY       – none (baseline)
      6. MEMORY_LINKING       – none (baseline)
      7. DEDUPLICATION        – none (baseline)

    Baseline is intentionally simple: flat list, linear scan, keyword overlap.
    A better implementation would use inverted indices, TF-IDF or BM25,
    graph-based linking, hierarchical consolidation, etc.
    """

    def __init__(self) -> None:
        # ── Flat list storage (baseline) ──────────────────────────────
        self._memories: list[dict[str, Any]] = []
        # ── Consolidated knowledge (baseline: frequency counts) ───────
        self._knowledge: dict[str, dict[str, Any]] = {}
        # ── Stats ─────────────────────────────────────────────────────
        self._consolidation_count: int = 0

    # ==================================================================
    # Store
    # ==================================================================

    def store(self, content: str, category: str = "", metadata: dict | None = None) -> str:
        """Store a memory. Returns a unique memory ID.

        Parameters
        ----------
        content : str
            The text content of the memory.
        category : str
            Optional category label (e.g. "geography", "coffee_orders").
        metadata : dict, optional
            Arbitrary key-value pairs attached to this memory.

        Returns
        -------
        str
            Unique identifier for the stored memory.
        """
        memory_id = uuid.uuid4().hex[:12]
        entry: dict[str, Any] = {
            "id": memory_id,
            "content": content,
            "category": category,
            "metadata": metadata or {},
            "timestamp": time.time(),
            "access_count": 0,
        }
        self._memories.append(entry)
        return memory_id

    # ==================================================================
    # Retrieve
    # ==================================================================

    def retrieve(self, query: str, k: int = 5, filters: dict | None = None) -> list[dict]:
        """Retrieve the k most relevant memories for a query.

        Parameters
        ----------
        query : str
            Natural-language query.
        k : int
            Maximum number of results to return.
        filters : dict, optional
            Filter criteria, e.g. {"category": "geography"}.

        Returns
        -------
        list[dict]
            Up to *k* memory dicts, each with at least "id", "content",
            "category", "score".  Sorted by descending relevance.
        """
        filters = filters or {}

        # ── Filter ────────────────────────────────────────────────────
        candidates = self._memories
        if "category" in filters:
            candidates = [m for m in candidates if m["category"] == filters["category"]]

        # ── Score by keyword overlap (baseline) ───────────────────────
        query_words = set(query.lower().split())
        scored: list[tuple[float, dict]] = []
        for mem in candidates:
            content_words = set(mem["content"].lower().split())
            if not query_words:
                overlap = 0.0
            else:
                overlap = len(query_words & content_words) / len(query_words)
            scored.append((overlap, mem))

        # ── Sort descending by score ──────────────────────────────────
        scored.sort(key=lambda x: x[0], reverse=True)

        results: list[dict] = []
        for score, mem in scored[:k]:
            mem["access_count"] = mem.get("access_count", 0) + 1
            results.append({
                "id": mem["id"],
                "content": mem["content"],
                "category": mem["category"],
                "metadata": mem.get("metadata", {}),
                "score": score,
            })
        return results

    # ==================================================================
    # Consolidate
    # ==================================================================

    def consolidate(self) -> dict:
        """Analyze stored memories and extract generalizations.

        Baseline: group memories by category and find the most frequently
        mentioned words/phrases to form simple generalizations.

        For example, 10 episodes of "user orders latte" produces a
        knowledge entry: "user most commonly mentions: latte, orders".

        Returns
        -------
        dict
            Stats: {"categories_processed": int, "rules_generated": int,
                     "memories_analyzed": int}.
        """
        # ── Group by category ─────────────────────────────────────────
        by_category: dict[str, list[dict]] = {}
        for mem in self._memories:
            cat = mem.get("category", "uncategorized") or "uncategorized"
            by_category.setdefault(cat, []).append(mem)

        rules_generated = 0

        for cat, mems in by_category.items():
            if len(mems) < 2:
                continue

            # ── Frequency counting (baseline) ─────────────────────────
            word_freq: dict[str, int] = {}
            for mem in mems:
                words = mem["content"].lower().split()
                for w in words:
                    # Skip very short / common words
                    if len(w) <= 2:
                        continue
                    word_freq[w] = word_freq.get(w, 0) + 1

            # ── Pick top recurring terms ──────────────────────────────
            if not word_freq:
                continue

            sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            top_terms = [w for w, c in sorted_words[:5] if c >= 2]

            if top_terms:
                self._knowledge[cat] = {
                    "category": cat,
                    "pattern": f"Most common terms: {', '.join(top_terms)}",
                    "top_terms": top_terms,
                    "episode_count": len(mems),
                    "source": "frequency_counting",
                }
                rules_generated += 1

        self._consolidation_count += 1

        return {
            "categories_processed": len(by_category),
            "rules_generated": rules_generated,
            "memories_analyzed": len(self._memories),
        }

    # ==================================================================
    # Query Knowledge
    # ==================================================================

    def query_knowledge(self, question: str) -> str:
        """Answer a question using consolidated knowledge (semantic memory)
        or specific episodes (episodic memory) as appropriate.

        Parameters
        ----------
        question : str
            Natural-language question.

        Returns
        -------
        str
            Best-effort answer drawn from stored knowledge and memories.
        """
        question_lower = question.lower()
        question_words = set(question_lower.split())

        # ── First: check consolidated knowledge ──────────────────────
        best_knowledge_score = 0.0
        best_knowledge: dict[str, Any] | None = None
        for cat, knowledge in self._knowledge.items():
            cat_words = set(cat.lower().replace("_", " ").split())
            terms = set(knowledge.get("top_terms", []))
            combined = cat_words | terms
            if combined:
                score = len(question_words & combined) / max(len(question_words), 1)
                if score > best_knowledge_score:
                    best_knowledge_score = score
                    best_knowledge = knowledge

        # ── Second: retrieve relevant episodes ───────────────────────
        episodes = self.retrieve(question, k=5)

        # ── Compose answer ───────────────────────────────────────────
        parts: list[str] = []

        if best_knowledge and best_knowledge_score > 0:
            parts.append(f"[Knowledge] {best_knowledge['pattern']}")

        for ep in episodes:
            if ep["score"] > 0:
                parts.append(ep["content"])

        if not parts:
            return "No relevant information found."

        return " | ".join(parts)

    # ==================================================================
    # Prune
    # ==================================================================

    def prune(self, max_memories: int | None = None, min_relevance: float | None = None) -> int:
        """Remove low-value memories. Returns count removed.

        Baseline: no pruning (returns 0).  A better implementation would
        use age, access frequency, importance scores, or a hybrid policy.

        Parameters
        ----------
        max_memories : int, optional
            If set, keep only the N most recent memories.
        min_relevance : float, optional
            If set, remove memories with access_count below this threshold.

        Returns
        -------
        int
            Number of memories removed.
        """
        # ── Baseline: no-op ───────────────────────────────────────────
        before = len(self._memories)

        if max_memories is not None and len(self._memories) > max_memories:
            # Keep most recent
            self._memories = self._memories[-max_memories:]

        if min_relevance is not None:
            self._memories = [
                m for m in self._memories
                if m.get("access_count", 0) >= min_relevance
            ]

        return before - len(self._memories)

    # ==================================================================
    # Stats
    # ==================================================================

    def get_stats(self) -> dict:
        """Return summary statistics about the memory system.

        Returns
        -------
        dict
            Keys: total_memories, categories, consolidation_count,
            knowledge_entries, avg_access_count.
        """
        categories = set(m.get("category", "") for m in self._memories)
        total_access = sum(m.get("access_count", 0) for m in self._memories)
        n = max(len(self._memories), 1)

        return {
            "total_memories": len(self._memories),
            "categories": sorted(categories),
            "category_count": len(categories),
            "consolidation_count": self._consolidation_count,
            "knowledge_entries": len(self._knowledge),
            "avg_access_count": total_access / n,
        }
