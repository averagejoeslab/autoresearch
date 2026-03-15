# Long-Term Memory — Autoresearch Program

## Objective

Improve the `LongTermMemory` class in `harness.py` so that it achieves the highest
possible composite score on the benchmark.  This is the **complete long-term memory
architecture** for an AI agent — storage, indexing, retrieval, consolidation, and
pruning as ONE unified system.

## Why This Matters

Long-term memory is the critical gap in production AI agents today.  Current systems
use flat files and git history as "memory" — no indexing, no consolidation, no
forgetting.  ICLR 2026 dedicated a full workshop (MemAgents) to this problem.
The key unsolved challenge: **episodic-to-semantic consolidation** — how to turn
thousands of raw episodes into compact, queryable knowledge.

### Relationship to Working Memory

LTM provides the pool that JIT loading draws from.  Better LTM indexing means
better JIT loading means better working memory quality.  If LTM retrieval is
poor, the agent's working memory will be filled with irrelevant context.

## Fitness Signal

```
score = 0.30 * retrieval_f1
      + 0.25 * consolidation_score
      + 0.20 * multi_hop_accuracy
      + 0.15 * scale_stability
      + 0.10 * (1 - avg_latency_normalized)
```

Higher is better.  Maximum possible score is 1.0.

- **retrieval_f1** (30%) — F1 of recall@5 and precision@5 across exact, semantic,
  and cross-category queries over 100 stored facts.
- **consolidation_score** (25%) — Ability to extract generalizations from episodic
  memories (60%) while preserving access to minority/exception events (40%).
- **multi_hop_accuracy** (20%) — Ability to answer questions requiring chaining
  2-3 stored facts together.
- **scale_stability** (15%) — Retrieval quality remains stable as memory count
  grows from 50 to 500.
- **latency** (10%, inverted) — Total benchmark runtime; faster is better.

## What You May Change

- **`harness.py`** — the `LongTermMemory` class and any helpers.  This is the
  ONLY file you may edit.

## What You Must NOT Change

- **`benchmark.py`** — locked evaluation.  Do not edit.
- The public API:
  - `store(content: str, category: str, metadata: dict) -> str`
  - `retrieve(query: str, k: int, filters: dict) -> list[dict]`
  - `consolidate() -> dict`
  - `query_knowledge(question: str) -> str`
  - `prune(max_memories: int, min_relevance: float) -> int`
  - `get_stats() -> dict`
- Each retrieved memory must have keys: `id`, `content`, `category`, `score`.
- `consolidate()` must return a dict with at least `categories_processed`,
  `rules_generated`, `memories_analyzed`.

## Baseline

The baseline uses:
- **Flat list** storage
- **Linear scan** with keyword overlap scoring for retrieval
- **Frequency counting** for consolidation (counts recurring words per category)
- **No pruning**, no linking, no deduplication

This scores reasonably on exact-match retrieval but poorly on semantic queries,
consolidation quality, and multi-hop reasoning.

## Architectural Patterns to Explore

The meta-agent can try entirely different architectural patterns:

### 1. Inverted Index + TF-IDF
Build a term-to-document inverted index at store time.  Score with TF-IDF cosine
similarity instead of raw keyword overlap.  Should dramatically improve both
retrieval precision and scale stability (sub-linear lookup).

### 2. BM25 Retrieval
Implement Okapi BM25 scoring with tunable k1 and b parameters.  BM25 handles
term frequency saturation and document length normalization better than raw TF-IDF.

### 3. Graph-Based Memory
Store memories as nodes in a graph.  Add typed edges:
- `same_category` — memories in the same category
- `keyword_overlap` — memories sharing significant terms
- `temporal_proximity` — memories stored close in time
- `causal_chain` — facts that reference each other

For multi-hop queries, traverse the graph to find connected fact chains.

### 4. Hierarchical Memory
Three tiers:
1. **Raw episodes** — individual stored memories
2. **Category summaries** — per-category consolidations
3. **Global knowledge** — cross-category patterns and rules

Retrieval checks all tiers; consolidation promotes from tier 1 to 2 to 3.

### 5. Two-Tier: Episodic + Semantic Store
Separate episodic memory (raw events) from semantic memory (derived knowledge).
Consolidation explicitly creates semantic entries from episodic patterns.
`query_knowledge` checks semantic store first, falls back to episodic.

### 6. A-MEM Style (Note-Based)
Each memory becomes a "note" with:
- Raw content
- Extracted keywords/tags
- Links to similar notes (by keyword co-occurrence or content similarity)
- Importance score (decays with time, increases with access)

### 7. N-gram Index
Build an index on character n-grams (e.g., trigrams) rather than whole words.
Provides fuzzy matching — "capital" matches "capitals", "capitalize", etc.

## Experimentation Loop

1. Read the current `harness.py` and understand the baseline.
2. Run `python benchmark.py` to get the baseline score.
3. Hypothesize an improvement and implement it in `harness.py`.
4. Run `python benchmark.py` again and compare scores.
5. If the score improved, keep the change.  If not, revert and try something else.
6. Repeat until diminishing returns.

Key insight: the four benchmark categories stress different capabilities.
Improving retrieval scoring (TF-IDF, BM25) should boost `retrieval_f1` and
`scale_stability`.  Building a graph or chain index should boost `multi_hop`.
Smarter consolidation (pattern extraction, majority detection) should boost
`consolidation_score`.  Inverted indices should improve `latency`.

## Output Format

After each experiment, log:

```
## Experiment N: <short title>
Hypothesis: <what you expect to happen>
Change: <what you modified>
Result: <new score vs previous score>
Delta: <+/- change>
Conclusion: <keep/revert and why>
```

## Constraints

- No external dependencies beyond Python standard library.
- No LLM calls — all operations must be purely algorithmic.
- No network calls, no disk I/O (beyond what Python needs internally).
- Must be runnable standalone: `python benchmark.py` from this directory.
