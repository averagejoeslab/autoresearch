# Memory Consolidation – Autoresearch Program

## Objective

Improve the `MemoryConsolidator` class in `harness.py` so that it achieves
the highest possible score on `benchmark.py`.  The benchmark adds 20
specific episodes across four categories (coffee orders, meeting
preferences, lunch choices, exercise habits) and tests whether the
consolidator can:

1. **Generalise** – identify dominant patterns (e.g., "user usually orders
   a latte").
2. **Preserve specifics** – still recall minority events (e.g., "user once
   ordered a cappuccino").

## Fitness Signal

```
score = 0.6 * generalization_accuracy + 0.4 * specificity_preservation
```

Higher is better.  Maximum possible score is 1.0.

## What You May Change

- **`harness.py`** – everything.  Rewrite the consolidator, add helper
  classes, change the consolidation algorithm, etc.

## What You Must NOT Change

- **`benchmark.py`** – this file is locked.
- The public API must remain compatible:
  - `add_episode(event: dict) -> None`
  - `consolidate() -> list[dict]`
  - `query_general(question: str) -> str`
  - The `episodes` and `generalizations` properties must exist.

## Approach Suggestions

1. **Run the benchmark** first and note the baseline score.
2. **Analyse specificity failures** – the baseline only surfaces the most
   common value, so `query_general("Did the user ever order a
   cappuccino?")` will fail.  You need a way to mention minority values.
3. **Improve consolidation**:
   - Track frequency distributions, not just the mode.
   - Produce summaries that mention both the dominant pattern and notable
     exceptions.
   - Consider temporal patterns (day-of-week, time-of-day).
4. **Improve query_general**:
   - Better keyword matching between the question and stored knowledge.
   - If the question asks about a specific value ("cappuccino"), search
     the raw episodes as well as generalisations.
   - Return answers that directly address the question.
5. **Iterate** – one change at a time, re-run benchmark.

## Constraints

- No network calls during benchmark execution.
- No external data files.
- Python standard library + numpy are available.
