# Working Memory — Autoresearch Program

## Objective

Improve the `WorkingMemory` class in `harness.py` to achieve the highest possible composite score on the benchmark. This is the **complete working memory system** for an AI agent — not individual sub-components, but the entire context window management architecture as ONE unified system.

Working memory = the agent's active context window. It decides:
- What stays in the window, what gets evicted
- How the token budget is allocated across sections (system prompt, history, tool results, current task)
- When and how to pull information from long-term memory into the active window (JIT loading)
- How to compress/summarize information to fit more into limited space

The meta-agent iterates on the **entire architectural pattern** — it can try completely different approaches to working memory, not just tune individual knobs.

## Benchmark

`benchmark.py` runs 40 tasks across 4 categories:

### Category 1: Information Retention Under Pressure (35% weight)
- 15 scenarios across different domains (software, medical, financial, legal, etc.)
- 60-message conversations with 5 key facts embedded at positions 5, 15, 25, 35, 55
- Budget squeezed to 30% of original, then compacted
- Measures: what fraction of facts survive compaction (marker word detection)
- **This is the #1 open problem** — per Anthropic's "effective context engineering" blog, fact retention under compression is the biggest quality blocker for agents

### Category 2: Budget Allocation Adaptation (25% weight)
- 10 workload scenarios: history-heavy, tool-heavy, and mixed
- Measures: does the allocation adapt to the workload? Does the dominant section get proportional space?
- Compares actual allocation fractions to ideal fractions based on input volume

### Category 3: JIT Loading Accuracy (25% weight)
- 10 scenarios with a query + 30 available chunks (5 relevant, 25 irrelevant)
- Covers: database, auth, Kubernetes, validation, notifications, testing, search, rate limiting, migration, observability
- Measures: F1 score (precision + recall) of chunk selection

### Category 4: System Prompt Preservation (15% weight)
- 5 different system prompts (engineer, medical, financial, legal, data science)
- 50 messages of conversation, then compaction
- Measures: word-level preservation rate of the original system prompt

Fitness signal:
```
score = 0.35 * fact_retention
      + 0.25 * budget_adaptation
      + 0.25 * jit_loading_f1
      + 0.15 * system_prompt_preservation
```

## Setup

```bash
cd harnesses/memory/wm
python benchmark.py
```

The benchmark prints `score: X.XXXXXX` at the end. Your goal is to maximize this score.

## What you can modify

- **`harness.py`** — the `WorkingMemory` class and ALL module-level configuration in this file. This is the ONLY file you may edit.
- You may change **anything** in harness.py: the class, the configuration constants, helper functions, the entire architecture.
- You must preserve the public API methods:
  - `__init__(self, total_budget: int = 4096)`
  - `ingest(self, message: dict) -> None`
  - `get_window(self) -> list[dict]`
  - `query_and_load(self, query: str, available_chunks: list[dict]) -> list[dict]`
  - `compact(self) -> dict`
  - `get_budget_allocation(self) -> dict`
  - `get_stats(self) -> dict`

## What you CANNOT modify

- **`benchmark.py`** — this is the locked evaluation. Do not edit it.
- The `contracts/` package.

## Configurable Dimensions (all in harness.py)

1. **EVICTION_STRATEGY**: "fifo", "lru", "importance_weighted", "hybrid"
2. **BUDGET_ALLOCATION**: fixed proportions vs adaptive vs priority-based
3. **COMPRESSION_METHOD**: "truncate", "extract_key_facts", "summarize", "hierarchical"
4. **JIT_RELEVANCE_SCORING**: "keyword_overlap", "tfidf", "configurable_weights"
5. **SECTION_PRIORITIES**: which sections get more budget protection
6. **ANCHOR_ROLES**: which roles are never evicted (system prompt, key decisions)
7. **RECENCY_WEIGHT**: how much recency matters vs importance (0.0 to 1.0)

## Baseline

The baseline uses the simplest possible strategies:
- **Eviction**: FIFO (oldest non-anchored message goes first)
- **Budget allocation**: Fixed proportions (system 15%, history 40%, tool results 20%, JIT 10%, current 15%)
- **Compression**: Truncate (just remove oldest messages until at 50% budget)
- **JIT loading**: Keyword overlap (Jaccard similarity between query and chunk)
- **Anchoring**: Only system messages are anchored

This baseline is intentionally simple. It provides a functional but suboptimal starting point.

## Architectural Patterns to Explore

These are fundamentally different approaches to the working memory problem — not just knob-turning:

### 1. Sliding Window with Importance Anchors
Keep a sliding window of recent messages, but "pin" important messages so they survive eviction. Importance could be detected by:
- Explicit markers (e.g., messages containing numbers, dates, decisions)
- Keyword density (technical terms, proper nouns)
- Message role (system prompts, first user message)

### 2. Hierarchical Compression
Three tiers of information density:
- **Hot** (recent): Full message content, no compression
- **Warm** (older): Summarized to first sentence or key facts
- **Cold** (oldest): Keywords only

As messages age, they move from hot -> warm -> cold. This maximizes information density per word.

### 3. Priority Queue with Section-Aware Eviction
Instead of FIFO, maintain a priority queue where eviction score combines:
- Section priority (system >> current >> tools >> JIT >> history)
- Recency (newer = higher priority)
- Importance (detected from content)
- Size (prefer evicting large, low-value messages)

### 4. Ring Buffer with Overflow Summarization
Fixed-size ring buffer for each section. When a section overflows:
- Summarize the oldest N messages into a single condensed message
- Push the summary to the front of the buffer
- Continue filling

### 5. Two-Tier: Hot + Warm
- **Hot tier**: Recent messages at full fidelity (last N messages)
- **Warm tier**: Compressed older messages (key facts, decisions, numbers)
- JIT loading pulls from external chunks into the hot tier
- Compaction moves hot -> warm

### 6. Fact-Extraction Pipeline
When compacting, don't just truncate — extract structured facts:
- Numbers and measurements (e.g., "$847.2 million", "99.9th percentile")
- Dates and deadlines (e.g., "September 15th", "Q3")
- Named entities (e.g., "PostgreSQL 15", "Istio 1.19")
- Decisions (e.g., "agreed to use gRPC", "approved $200 million buyback")

Store extracted facts separately and always include them in the window.

### 7. Adaptive Budget Rebalancing
Instead of fixed proportions, dynamically adjust budget allocation based on:
- Actual content volume per section
- Which sections have been queried/accessed most recently
- The current task's likely needs

## Why This Matters

Context management is the **#1 quality blocker for AI agents**. A 2024 survey found 32% of practitioners cite context window limitations as their primary challenge. The problems are:

1. **Lost facts**: Important information gets pushed out of the window by newer, less important content
2. **Wasted budget**: Fixed allocation wastes space on sections that don't need it
3. **Poor retrieval**: JIT loading picks irrelevant chunks, wasting precious budget
4. **Degraded instructions**: System prompt gets corrupted or lost after many turns

Solving working memory well is a prerequisite for every other agent capability (planning, tool use, verification).

## Relationship to Long-Term Memory (LTM)

Working memory decides:
- **What to load from LTM**: The `query_and_load` method selects which stored knowledge to bring into the active window
- **What to persist back to LTM**: When facts are evicted from working memory, they could be persisted to LTM for later retrieval (not yet implemented in baseline)

This experiment focuses on the working memory side. The `harnesses/memory/ltm/` experiment (when built) will focus on the storage and retrieval side.

## Experimentation Loop

1. Read the current `harness.py` and understand the baseline.
2. Run `python benchmark.py` to get the baseline score.
3. Hypothesize an improvement and implement it in `harness.py`.
4. Run `python benchmark.py` again and compare scores.
5. If the score improved, keep the change. If not, revert and try something else.
6. Repeat until diminishing returns.

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

## Logging

Append all experiment logs to `experiments.log` in this directory. Include timestamps and full benchmark output for each run.

## Constraints

- No external dependencies beyond Python stdlib.
- No LLM calls — all logic must be purely algorithmic.
- All messages must retain their "role" and "content" keys.
- System messages with role "system" must never be evicted (they are anchored).
- The `total_budget` attribute on the instance may be changed externally (the benchmark does this to simulate pressure).
