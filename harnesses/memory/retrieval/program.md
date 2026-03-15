# Memory Retrieval – Autoresearch Program

## Objective

Improve the `MemoryStore` class in `harness.py` so that it achieves the
highest possible score on `benchmark.py`.  The benchmark stores 100 facts
(key-value pairs with metadata) and runs 30 retrieval queries across three
categories: exact match, keyword overlap, and metadata-filtered.

## Fitness Signal

```
score = 0.5 * recall@5 + 0.3 * precision@5 + 0.2 * (1 - latency_normalized)
```

Higher is better.  Maximum possible score is 1.0.

## What You May Change

- **`harness.py`** – everything is fair game.  You may rewrite the entire
  `MemoryStore` class, add helper classes, change the retrieval algorithm,
  add indexing structures, etc.

## What You Must NOT Change

- **`benchmark.py`** – this file is locked.  Do not modify it.
- The public API must remain compatible:
  - `store(key: str, value: str, metadata: dict) -> None`
  - `retrieve(query: str, k: int, filters: dict | None) -> list[dict]`
  - `size() -> int`
  - Each returned dict must contain at least `key`, `value`, `metadata`,
    and `score` fields.

## Approach Suggestions

1. **Start by running the benchmark** to see the baseline score.
2. **Analyze which queries fail** – print the query, expected keys, and
   retrieved keys to understand what the keyword-overlap baseline misses.
3. **Improve retrieval** – consider:
   - TF-IDF weighting instead of raw word overlap.
   - Stemming / lemmatization for better keyword matching.
   - Bigram or trigram matching.
   - Inverted index for faster lookups.
   - Metadata-aware scoring (boost entries whose metadata matches query
     terms).
   - Embedding-based similarity if a model is available.
4. **Optimise latency** – the benchmark penalises slow retrieval.  Index
   structures (inverted index, approximate nearest neighbor) help.
5. **Iterate** – make one change, re-run the benchmark, keep what helps.

## Constraints

- No network calls during benchmark execution.
- No reading external files (all data is inline in `benchmark.py`).
- Python standard library + numpy are available.  If you want other
  packages, check availability before importing.
