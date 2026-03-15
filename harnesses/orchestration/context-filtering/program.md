# Context Filter

## Goal

Improve the `ContextFilter` class in `harness.py` to maximize the benchmark score. The benchmark evaluates selecting relevant context items for a subtask within a budget constraint.

## Benchmark

`benchmark.py` runs 20 test cases. Each provides:
- A pool of 50 context items (facts, instructions, observations about a web app project)
- A subtask description (e.g., "Set up JWT authentication for the Flask API")
- A budget of 10 items
- Ground-truth relevant item IDs

The fitness signal is:
```
score = 0.4 * recall + 0.3 * precision + 0.3 * budget_efficiency
```

## Baseline

The current baseline takes the most recent items by timestamp that fit the budget. It ignores the subtask description entirely.

Baseline score: **~0.40**

## What to improve

The baseline has very low recall (~0.18) and precision (~0.10) because it ignores topical relevance completely. The only thing it does well is budget efficiency (always uses full budget).

### Improvement directions

1. **Keyword matching**: Tokenize the subtask description and each context item's content. Score items by word overlap (TF-IDF, Jaccard, or simple intersection). Select the highest-scoring items within budget.

2. **Semantic grouping**: Context items about related topics (e.g., "PostgreSQL" and "database migrations") should be co-selected when the subtask is about databases.

3. **Type-aware selection**: Instructions and facts are generally more relevant than observations. Weight item types differently.

4. **Recency as a tiebreaker**: When relevance scores are tied, prefer more recent items. The current baseline uses recency as the only signal -- make it a secondary signal instead.

5. **Synonym and related-term expansion**: "JWT" relates to "authentication", "token", "security". Build a small synonym map or use word co-occurrence from the context pool itself.

## Constraints

- Only modify `harness.py`
- Do not modify `benchmark.py`
- No external dependencies (stdlib only)
- No LLM calls -- the logic must be purely algorithmic
- The class must keep the same method signatures

## Running

```bash
python benchmark.py
```

Score is printed as `score: X.XXXXXX` on the last line.
