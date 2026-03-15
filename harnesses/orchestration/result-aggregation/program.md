# Result Aggregation

## Goal

Improve the `ResultAggregator` class in `harness.py` to maximize the benchmark score. The benchmark evaluates merging outputs from multiple sub-agents into a coherent, complete answer.

## Benchmark

`benchmark.py` runs 15 test cases covering:
- Complementary results (non-overlapping information)
- Overlapping results (agreement between agents)
- Contradictory results (agents disagree)
- Single results, empty results
- Mixed confidence levels
- Duplicate information across agents

The fitness signal is:
```
score = 0.4 * completeness + 0.3 * contradiction_handling + 0.3 * coherence
```

## Baseline

The current baseline concatenates all results, each prefixed with `[subtask]: output`.

Baseline score: **~0.95**

## What to improve

The baseline already scores well because concatenation preserves all key facts (high completeness) and the labeling provides structure (high coherence). The main weakness is **contradiction handling** (~0.83). The baseline includes contradictory statements without flagging them.

### Improvement directions

1. **Contradiction detection**: When agents disagree, add hedging language ("however", "on the other hand", "note: conflicting information"). Look for antonyms, negation patterns, or opposite sentiment.

2. **Confidence-weighted merging**: Give more weight to high-confidence results. When there's a contradiction, prefer the higher-confidence agent's position while still mentioning the alternative.

3. **Deduplication**: When multiple agents say the same thing, merge into a single statement rather than repeating.

4. **Structured output**: Organize by topic rather than by agent/subtask when it makes the answer more coherent.

5. **Summary generation**: Add a brief synthesis at the top or bottom that captures the overall answer to the original task.

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
