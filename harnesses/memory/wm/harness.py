"""
Working Memory — Complete context window management for AI agents.

This is the ENTIRE working memory system as a single, unified primitive.
It handles eviction, budget allocation, compression, and JIT loading
as one coherent architecture — not fragmented sub-components.

The meta-agent iterates on the whole design: storage format, eviction
strategy, budget allocation, compression method, JIT relevance scoring.
Everything is mutable.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any


# =====================================================================
# Configurable dimensions — ALL mutable by the meta-agent
# =====================================================================

# 1. Eviction strategy when the window exceeds budget
#    Options: "fifo", "lru", "importance_weighted", "hybrid"
EVICTION_STRATEGY: str = "fifo"

# 2. Budget allocation across sections (fractions, must sum to 1.0)
#    Keys: "system", "history", "tool_results", "jit_loaded", "current"
BUDGET_ALLOCATION: dict[str, float] = {
    "system": 0.15,
    "history": 0.40,
    "tool_results": 0.20,
    "jit_loaded": 0.10,
    "current": 0.15,
}

# 3. Compression method applied during compact()
#    Options: "truncate", "extract_key_facts", "summarize", "hierarchical"
COMPRESSION_METHOD: str = "truncate"

# 4. JIT relevance scoring method
#    Options: "keyword_overlap", "tfidf", "configurable_weights"
JIT_RELEVANCE_SCORING: str = "keyword_overlap"

# 5. Section priorities — higher number = keep longer during eviction
SECTION_PRIORITIES: dict[str, int] = {
    "system": 100,     # never evict
    "current": 90,     # almost never evict
    "tool_results": 50,
    "jit_loaded": 40,
    "history": 30,
}

# 6. Roles/types that are NEVER evicted (anchored in window)
ANCHOR_ROLES: set[str] = {"system"}

# 7. Recency weight: how much recency matters relative to importance
#    0.0 = only importance, 1.0 = only recency
RECENCY_WEIGHT: float = 1.0


# =====================================================================
# Working Memory System
# =====================================================================

class WorkingMemory:
    """Complete working memory system for an AI agent.

    The meta-agent iterates on the entire architecture — storage format,
    eviction strategy, budget allocation, compression, JIT loading.
    Everything is fair game.
    """

    def __init__(self, total_budget: int = 4096):
        """Initialize with a total token budget (in words as proxy).

        Parameters
        ----------
        total_budget : int
            Maximum number of words the context window can hold.
            Words are used as a proxy for tokens (roughly 1 word ~ 1.3 tokens).
        """
        self.total_budget = total_budget

        # ── internal storage ────────────────────────────────────────
        self._messages: list[dict[str, Any]] = []
        self._access_order: list[int] = []  # indices, most-recently-accessed last
        self._next_id: int = 0

        # ── stats ───────────────────────────────────────────────────
        self._eviction_count: int = 0
        self._compaction_count: int = 0
        self._total_ingested: int = 0
        self._words_compressed: int = 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _word_count(text: str) -> int:
        """Count words in a string (proxy for token count)."""
        return len(text.split())

    def _message_size(self, msg: dict[str, Any]) -> int:
        """Total word count of a message (content + role label)."""
        return self._word_count(msg.get("content", "")) + 1  # +1 for role tag

    def _current_usage(self) -> int:
        """Total words currently in the window."""
        return sum(self._message_size(m) for m in self._messages)

    def _section_usage(self) -> dict[str, int]:
        """Words used by each section."""
        usage: dict[str, int] = {
            "system": 0, "history": 0, "tool_results": 0,
            "jit_loaded": 0, "current": 0,
        }
        for m in self._messages:
            section = m.get("section", "history")
            if section not in usage:
                section = "history"
            usage[section] += self._message_size(m)
        return usage

    def _touch(self, msg_id: int) -> None:
        """Mark a message as recently accessed (for LRU)."""
        if msg_id in self._access_order:
            self._access_order.remove(msg_id)
        self._access_order.append(msg_id)

    def _classify_section(self, message: dict[str, Any]) -> str:
        """Assign a message to a budget section based on its role."""
        role = message.get("role", "user")
        if role == "system":
            return "system"
        if role == "tool_result":
            return "tool_results"
        if message.get("jit_loaded", False):
            return "jit_loaded"
        if message.get("is_current", False):
            return "current"
        return "history"

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def ingest(self, message: dict[str, Any]) -> None:
        """Add a new message to working memory. May trigger eviction.

        Parameters
        ----------
        message : dict
            Must have at least "role" (str) and "content" (str).
            Optional: "metadata" (dict), "importance" (float 0-1),
                      "is_current" (bool), "jit_loaded" (bool).
        """
        msg = dict(message)  # shallow copy
        msg["_id"] = self._next_id
        self._next_id += 1
        self._total_ingested += 1

        # Assign section
        if "section" not in msg:
            msg["section"] = self._classify_section(msg)

        self._messages.append(msg)
        self._touch(msg["_id"])

        # Evict if over budget
        self._evict_if_needed()

    def get_window(self) -> list[dict[str, Any]]:
        """Return the current context window contents, formatted for the LLM.

        Returns
        -------
        list[dict]
            Each dict has "role" and "content" at minimum.
            Messages are in chronological order.
        """
        result = []
        for m in self._messages:
            self._touch(m["_id"])
            # Return a clean copy without internal fields
            clean = {k: v for k, v in m.items() if not k.startswith("_")}
            result.append(clean)
        return result

    def query_and_load(
        self, query: str, available_chunks: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """JIT loading: select relevant chunks and load them into the window.

        Parameters
        ----------
        query : str
            The current task/question that needs context.
        available_chunks : list[dict]
            Each dict has at least "content" (str) and optionally
            "id", "source", "metadata".

        Returns
        -------
        list[dict]
            The chunks that were loaded into the window.
        """
        if not available_chunks:
            return []

        # Determine JIT budget (in words)
        jit_budget_words = int(self.total_budget * BUDGET_ALLOCATION.get("jit_loaded", 0.10))

        # Score each chunk for relevance
        scored = []
        for chunk in available_chunks:
            relevance = self._score_relevance(query, chunk)
            scored.append((relevance, chunk))

        # Sort by relevance descending
        scored.sort(key=lambda x: x[0], reverse=True)

        # Load chunks until JIT budget is exhausted
        loaded: list[dict[str, Any]] = []
        words_used = 0
        for relevance, chunk in scored:
            if relevance <= 0:
                break
            chunk_words = self._word_count(chunk.get("content", ""))
            if words_used + chunk_words > jit_budget_words:
                continue  # skip this chunk, try smaller ones
            # Ingest as JIT-loaded message
            jit_msg = {
                "role": "jit_context",
                "content": chunk.get("content", ""),
                "jit_loaded": True,
                "section": "jit_loaded",
                "source_id": chunk.get("id"),
                "metadata": chunk.get("metadata", {}),
            }
            self.ingest(jit_msg)
            loaded.append(chunk)
            words_used += chunk_words

        return loaded

    def compact(self) -> dict[str, Any]:
        """Compress the current window to free up space.

        Returns
        -------
        dict
            Metadata about the compaction: words_before, words_after,
            messages_before, messages_after, method.
        """
        words_before = self._current_usage()
        messages_before = len(self._messages)

        if COMPRESSION_METHOD == "truncate":
            self._compact_truncate()
        elif COMPRESSION_METHOD == "extract_key_facts":
            self._compact_extract_key_facts()
        elif COMPRESSION_METHOD == "summarize":
            self._compact_summarize()
        elif COMPRESSION_METHOD == "hierarchical":
            self._compact_hierarchical()
        else:
            self._compact_truncate()

        words_after = self._current_usage()
        messages_after = len(self._messages)
        self._compaction_count += 1
        self._words_compressed += max(0, words_before - words_after)

        return {
            "words_before": words_before,
            "words_after": words_after,
            "messages_before": messages_before,
            "messages_after": messages_after,
            "method": COMPRESSION_METHOD,
            "compression_ratio": words_after / words_before if words_before > 0 else 1.0,
        }

    def get_budget_allocation(self) -> dict[str, Any]:
        """Return current budget allocation across sections.

        Returns
        -------
        dict
            Keys: section names. Values: dict with "allocated_words",
            "used_words", "utilization".
        """
        usage = self._section_usage()
        result = {}
        for section, fraction in BUDGET_ALLOCATION.items():
            allocated = int(self.total_budget * fraction)
            used = usage.get(section, 0)
            result[section] = {
                "allocated_words": allocated,
                "used_words": used,
                "utilization": used / allocated if allocated > 0 else 0.0,
            }
        return result

    def get_stats(self) -> dict[str, Any]:
        """Return memory stats: usage, eviction count, compression ratio, etc.

        Returns
        -------
        dict
            Comprehensive statistics about the working memory state.
        """
        usage = self._current_usage()
        return {
            "total_budget": self.total_budget,
            "current_usage_words": usage,
            "utilization": usage / self.total_budget if self.total_budget > 0 else 0.0,
            "message_count": len(self._messages),
            "total_ingested": self._total_ingested,
            "eviction_count": self._eviction_count,
            "compaction_count": self._compaction_count,
            "words_compressed": self._words_compressed,
            "section_usage": self._section_usage(),
        }

    # ------------------------------------------------------------------
    # Eviction strategies
    # ------------------------------------------------------------------

    def _evict_if_needed(self) -> None:
        """Evict messages until usage fits within total budget."""
        while self._current_usage() > self.total_budget:
            victim = self._select_eviction_victim()
            if victim is None:
                break  # nothing left to evict
            self._messages.remove(victim)
            if victim["_id"] in self._access_order:
                self._access_order.remove(victim["_id"])
            self._eviction_count += 1

    def _select_eviction_victim(self) -> dict[str, Any] | None:
        """Select the next message to evict based on the current strategy."""
        # Never evict anchored messages
        candidates = [
            m for m in self._messages
            if m.get("role") not in ANCHOR_ROLES
        ]
        if not candidates:
            return None

        if EVICTION_STRATEGY == "fifo":
            return candidates[0]

        elif EVICTION_STRATEGY == "lru":
            # Evict the least-recently-accessed candidate
            id_to_msg = {m["_id"]: m for m in candidates}
            for msg_id in self._access_order:
                if msg_id in id_to_msg:
                    return id_to_msg[msg_id]
            return candidates[0]  # fallback

        elif EVICTION_STRATEGY == "importance_weighted":
            # Evict lowest importance; ties broken by age (oldest first)
            candidates.sort(
                key=lambda m: (
                    SECTION_PRIORITIES.get(m.get("section", "history"), 30),
                    m.get("importance", 0.5),
                    m["_id"],  # older messages have lower IDs
                )
            )
            return candidates[0]

        elif EVICTION_STRATEGY == "hybrid":
            # Combine recency, importance, and section priority
            max_id = max(m["_id"] for m in self._messages) if self._messages else 1
            scored = []
            for m in candidates:
                age_norm = m["_id"] / max_id if max_id > 0 else 0
                importance = m.get("importance", 0.5)
                section_prio = SECTION_PRIORITIES.get(
                    m.get("section", "history"), 30
                ) / 100.0
                # Higher keep_score = less likely to evict
                keep_score = (
                    RECENCY_WEIGHT * age_norm
                    + (1 - RECENCY_WEIGHT) * (importance * 0.5 + section_prio * 0.5)
                )
                scored.append((keep_score, m))
            scored.sort(key=lambda x: x[0])
            return scored[0][1]

        else:
            return candidates[0]

    # ------------------------------------------------------------------
    # Compression strategies
    # ------------------------------------------------------------------

    def _compact_truncate(self) -> None:
        """Truncate: remove oldest non-anchored messages until at 50% budget."""
        target = self.total_budget // 2
        while self._current_usage() > target:
            victim = self._select_eviction_victim()
            if victim is None:
                break
            self._messages.remove(victim)
            if victim["_id"] in self._access_order:
                self._access_order.remove(victim["_id"])
            self._eviction_count += 1

    def _compact_extract_key_facts(self) -> None:
        """Extract key facts from older messages, replace with condensed version."""
        target = self.total_budget // 2
        if self._current_usage() <= target:
            return

        # Find non-anchored messages, oldest first
        compactable = [
            m for m in self._messages
            if m.get("role") not in ANCHOR_ROLES
        ]
        if not compactable:
            return

        # Take the oldest half and extract key facts
        half = max(1, len(compactable) // 2)
        to_compress = compactable[:half]

        facts = []
        for m in to_compress:
            content = m.get("content", "")
            # Extract sentences that contain key patterns
            sentences = re.split(r'[.!?]+', content)
            for s in sentences:
                s = s.strip()
                if len(s.split()) >= 3:  # skip trivial fragments
                    facts.append(s)

        # Remove originals
        for m in to_compress:
            if m in self._messages:
                self._messages.remove(m)

        # Insert condensed summary if there are facts
        if facts:
            # Keep unique facts, limit to reasonable size
            unique_facts = list(dict.fromkeys(facts))[:20]
            summary_content = "Key facts from earlier conversation: " + ". ".join(unique_facts) + "."
            summary_msg = {
                "role": "assistant",
                "content": summary_content,
                "section": "history",
                "_id": self._next_id,
                "is_summary": True,
            }
            self._next_id += 1
            # Insert after system messages
            insert_pos = 0
            for i, m in enumerate(self._messages):
                if m.get("role") == "system":
                    insert_pos = i + 1
                else:
                    break
            self._messages.insert(insert_pos, summary_msg)

    def _compact_summarize(self) -> None:
        """Summarize: collapse older messages into a brief summary."""
        # For pure-Python, summarize = extract first sentence of each message
        target = self.total_budget // 2
        if self._current_usage() <= target:
            return

        compactable = [
            m for m in self._messages
            if m.get("role") not in ANCHOR_ROLES
        ]
        if not compactable:
            return

        half = max(1, len(compactable) // 2)
        to_compress = compactable[:half]

        summaries = []
        for m in to_compress:
            content = m.get("content", "")
            first_sentence = content.split(".")[0].strip()
            if first_sentence:
                role_tag = m.get("role", "unknown")
                summaries.append(f"[{role_tag}] {first_sentence}")

        for m in to_compress:
            if m in self._messages:
                self._messages.remove(m)

        if summaries:
            summary_content = "Summary of earlier messages: " + "; ".join(summaries) + "."
            summary_msg = {
                "role": "assistant",
                "content": summary_content,
                "section": "history",
                "_id": self._next_id,
                "is_summary": True,
            }
            self._next_id += 1
            insert_pos = 0
            for i, m in enumerate(self._messages):
                if m.get("role") == "system":
                    insert_pos = i + 1
                else:
                    break
            self._messages.insert(insert_pos, summary_msg)

    def _compact_hierarchical(self) -> None:
        """Hierarchical: recent=full, older=summary, oldest=keywords only."""
        compactable = [
            m for m in self._messages
            if m.get("role") not in ANCHOR_ROLES
        ]
        if not compactable:
            return

        n = len(compactable)
        # Split into thirds: oldest, older, recent
        third = max(1, n // 3)
        oldest = compactable[:third]
        older = compactable[third:2 * third]
        # recent (last third) stays untouched

        # Oldest -> keywords only
        if oldest:
            keywords_list = []
            for m in oldest:
                content = m.get("content", "")
                words = content.lower().split()
                # Extract words that appear meaningful (> 3 chars, not stopwords)
                stopwords = {"the", "and", "for", "are", "but", "not", "you",
                             "all", "can", "had", "her", "was", "one", "our",
                             "out", "has", "have", "been", "this", "that",
                             "with", "from", "they", "will", "would", "there",
                             "their", "what", "about", "which", "when", "make"}
                meaningful = [w for w in words if len(w) > 3 and w not in stopwords]
                keywords_list.extend(meaningful[:5])

            for m in oldest:
                if m in self._messages:
                    self._messages.remove(m)

            if keywords_list:
                kw_content = "Keywords from oldest context: " + ", ".join(keywords_list[:30])
                kw_msg = {
                    "role": "assistant",
                    "content": kw_content,
                    "section": "history",
                    "_id": self._next_id,
                    "is_summary": True,
                }
                self._next_id += 1
                insert_pos = 0
                for i, m in enumerate(self._messages):
                    if m.get("role") == "system":
                        insert_pos = i + 1
                    else:
                        break
                self._messages.insert(insert_pos, kw_msg)

        # Older -> first-sentence summaries
        if older:
            summaries = []
            for m in older:
                content = m.get("content", "")
                first = content.split(".")[0].strip()
                if first:
                    summaries.append(first)

            for m in older:
                if m in self._messages:
                    self._messages.remove(m)

            if summaries:
                sum_content = "Summary of older context: " + ". ".join(summaries) + "."
                sum_msg = {
                    "role": "assistant",
                    "content": sum_content,
                    "section": "history",
                    "_id": self._next_id,
                    "is_summary": True,
                }
                self._next_id += 1
                insert_pos = 0
                for i, m in enumerate(self._messages):
                    if m.get("role") == "system":
                        insert_pos = i + 1
                    else:
                        break
                self._messages.insert(insert_pos, sum_msg)

    # ------------------------------------------------------------------
    # JIT relevance scoring
    # ------------------------------------------------------------------

    def _score_relevance(self, query: str, chunk: dict[str, Any]) -> float:
        """Score a chunk's relevance to a query.

        Returns a float in [0, 1].
        """
        if JIT_RELEVANCE_SCORING == "keyword_overlap":
            return self._relevance_keyword_overlap(query, chunk)
        elif JIT_RELEVANCE_SCORING == "tfidf":
            return self._relevance_tfidf(query, chunk)
        elif JIT_RELEVANCE_SCORING == "configurable_weights":
            return self._relevance_configurable(query, chunk)
        else:
            return self._relevance_keyword_overlap(query, chunk)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple word tokenizer: lowercase, split on non-alphanumeric."""
        return re.findall(r'[a-z0-9]+', text.lower())

    def _relevance_keyword_overlap(self, query: str, chunk: dict[str, Any]) -> float:
        """Jaccard-like keyword overlap between query and chunk content."""
        query_tokens = set(self._tokenize(query))
        chunk_tokens = set(self._tokenize(chunk.get("content", "")))
        if not query_tokens or not chunk_tokens:
            return 0.0
        intersection = query_tokens & chunk_tokens
        union = query_tokens | chunk_tokens
        return len(intersection) / len(union) if union else 0.0

    def _relevance_tfidf(self, query: str, chunk: dict[str, Any]) -> float:
        """TF-IDF-inspired scoring using the current window as the corpus."""
        query_tokens = self._tokenize(query)
        chunk_tokens = self._tokenize(chunk.get("content", ""))

        if not query_tokens or not chunk_tokens:
            return 0.0

        # Build document frequency from current window
        doc_count = max(1, len(self._messages))
        df: Counter = Counter()
        for m in self._messages:
            unique_words = set(self._tokenize(m.get("content", "")))
            for w in unique_words:
                df[w] += 1

        # Also count the chunk itself
        chunk_unique = set(chunk_tokens)
        for w in chunk_unique:
            df[w] += 1

        # Score: sum of TF-IDF for query terms found in chunk
        chunk_tf: Counter = Counter(chunk_tokens)
        total_chunk_words = len(chunk_tokens)
        score = 0.0
        for qt in set(query_tokens):
            if qt in chunk_tf:
                tf = chunk_tf[qt] / total_chunk_words
                idf = math.log((doc_count + 1) / (df.get(qt, 0) + 1)) + 1
                score += tf * idf

        # Normalize to [0, 1] range approximately
        max_possible = len(set(query_tokens)) * (math.log(doc_count + 1) + 1)
        return min(1.0, score / max_possible) if max_possible > 0 else 0.0

    def _relevance_configurable(self, query: str, chunk: dict[str, Any]) -> float:
        """Weighted combination of keyword overlap and metadata signals."""
        keyword_score = self._relevance_keyword_overlap(query, chunk)

        # Bonus for matching metadata tags
        meta_bonus = 0.0
        chunk_meta = chunk.get("metadata", {})
        if isinstance(chunk_meta, dict):
            meta_text = " ".join(str(v) for v in chunk_meta.values())
            meta_tokens = set(self._tokenize(meta_text))
            query_tokens = set(self._tokenize(query))
            if meta_tokens and query_tokens:
                meta_bonus = len(meta_tokens & query_tokens) / len(query_tokens)

        return 0.7 * keyword_score + 0.3 * meta_bonus
