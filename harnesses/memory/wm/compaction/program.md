# Context Compaction — Autoresearch Program

## Objective

Improve the `ContextCompactor` class in `harness.py` so that it achieves the highest possible composite score on the benchmark. The benchmark measures three things:

1. **Fact retention** (50% weight): Key facts embedded in long conversations must survive compaction.
2. **Instruction preservation** (30% weight): System prompts must be kept verbatim.
3. **Recency bias** (20% weight): Recent messages should be preserved more than old ones.

## Setup

```bash
cd harnesses/context/compaction
python benchmark.py
```

The benchmark prints `score: X.XXXXXX` at the end. Your goal is to maximize this score.

## What you can modify

- **`harness.py`** — the `ContextCompactor` class and any helpers in this file. This is the ONLY file you may edit.
- You may add new methods, change the algorithm, add heuristics, etc.
- You must preserve the public API: `compact(messages: list[dict], token_budget: int) -> list[dict]`
- Token counting uses `len(msg["content"].split())` as a word-count proxy.

## What you CANNOT modify

- **`benchmark.py`** — this is the locked evaluation. Do not edit it.
- The `contracts/` package.

## Baseline

The baseline strategy is:
1. Keep system messages verbatim (truncate if they exceed the entire budget).
2. Fill remaining budget with the most recent non-system messages.
3. Drop older messages that do not fit.

This scores reasonably on recency and instruction preservation but poorly on fact retention (important facts embedded early in the conversation get dropped).

## Experimentation Ideas

Here are directions to explore (not exhaustive):

- **Extractive summarization**: Scan dropped messages for key facts (numbers, proper nouns, config values) and append a summary message.
- **Importance scoring**: Score each message by information density (unique words, numbers, named entities) and keep high-scoring messages regardless of position.
- **Sliding-window merge**: Merge adjacent short messages into combined messages to save per-message overhead.
- **Two-pass compaction**: First pass identifies "anchor" messages containing key info; second pass fills remaining budget chronologically.
- **Hybrid recency + importance**: Keep the last N messages plus the top-K most important older messages.

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
- No LLM calls — the compaction must be purely algorithmic.
- The compacted output must respect the token budget (word count).
- Messages in the output must preserve their original `role` field.
- The output should maintain chronological order where possible.
