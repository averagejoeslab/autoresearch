# JIT Context Loading — Autoresearch Program

## Objective

Improve the `JITLoader` class in `harness.py` so that it achieves the highest possible composite score on the benchmark. The benchmark measures three things:

1. **Recall of relevant chunks** (50% weight): What fraction of the truly relevant chunks were selected?
2. **Precision** (30% weight): What fraction of selected chunks are actually relevant?
3. **Importance ordering** (20% weight): Among selected relevant chunks, are they the most important ones?

## Setup

```bash
cd harnesses/context/jit-loading
python benchmark.py
```

The benchmark prints `score: X.XXXXXX` at the end. Your goal is to maximize this score.

## What you can modify

- **`harness.py`** — the `JITLoader` class and any helpers in this file. This is the ONLY file you may edit.
- You may add new methods, change the scoring/selection algorithm, add new heuristics, etc.
- You must preserve the public API: `select(query: str, available_chunks: list[dict], budget: int) -> list[dict]`
- Token counting uses `len(content.split())` as a word-count proxy.
- Each chunk has `"content"` (text) and `"metadata"` with `"topic"`, `"recency"` (0-1), and `"importance"` (0-1).

## What you CANNOT modify

- **`benchmark.py`** — this is the locked evaluation. Do not edit it.
- The `contracts/` package.

## Baseline

The baseline strategy:
1. Score each chunk by: `0.5 * keyword_overlap + 0.3 * recency + 0.2 * importance`
2. Sort chunks by score descending.
3. Greedily select chunks that fit the budget.

Keyword overlap is computed as the fraction of query words found in the chunk content (or topic metadata), case-insensitive.

## How the Benchmark Builds Tasks

Understanding the benchmark's task construction helps guide improvements:
- Relevant chunks contain topically related technical content with specific keywords.
- Irrelevant chunks are either off-topic technical content or completely unrelated filler (weather, cooking, etc.).
- Topic metadata mirrors the content's domain (e.g., "python debugging").
- Importance scores for relevant chunks are generally higher (0.6-1.0) than irrelevant ones (0.2-0.5).

## Experimentation Ideas

- **TF-IDF-like scoring**: Weight rare/distinctive query words more heavily than common ones.
- **Bigram/trigram matching**: Match multi-word phrases, not just individual words.
- **Topic clustering**: Group chunks by topic and prefer chunks from the query's topic cluster.
- **Importance-first selection**: Among chunks with any keyword overlap, sort by importance first.
- **Threshold-based filtering**: Only consider chunks above a minimum relevance score, then sort by importance.
- **Content diversity**: Avoid selecting chunks with highly overlapping content.
- **Metadata exploitation**: Use the topic field more aggressively — exact topic match should be a strong signal.
- **Two-stage selection**: Stage 1 filters by relevance threshold, Stage 2 ranks by importance within the filtered set.
- **Dynamic weight tuning**: Adjust the balance between relevance, recency, and importance based on query characteristics.

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
- No LLM calls — the selection must be purely algorithmic.
- The selected chunks must fit within the word budget.
- The `_relevant` field in chunk metadata is ground-truth for the benchmark only — your harness must NOT read it. The harness should treat chunks as opaque (only using `content` and `metadata`).
- Return chunks as-is from the input list (do not modify chunk contents).
