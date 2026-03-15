# Grading Strategies -- Autoresearch Program

## Goal
Improve the `GradingStrategy` class in `harness.py` so its grading methods correlate as highly as possible with human judgment scores.

## Fitness Signal
```
fitness = 0.5 * correlation_with_human + 0.3 * consistency + 0.2 * speed
```
Run `python benchmark.py` to get the current score.

## What You Can Change
- **Only** edit `harness.py` in this directory.
- Do **not** edit `benchmark.py` -- it is the locked evaluation.
- You may add new grading strategies or improve existing ones.
- The class interface must remain: `grade(output, expected, task, strategy) -> {"score": float, "method": str, "details": dict}`.
- No LLM/API calls allowed in the harness (must be pure code).

## Baseline
The baseline uses exact string match for `exact_match` and simple heuristics for the other strategies. It achieves a moderate score.

## Improvement Ideas (ranked by likely impact)
1. **Fuzzy match**: Use edit distance (Levenshtein), normalized by string length. Consider case-insensitive matching and whitespace normalization.
2. **Keyword presence**: Weight keywords by importance (TF-IDF style). Penalize extra/irrelevant content.
3. **Semantic similarity**: Use word overlap with IDF weighting, n-gram overlap, or cosine similarity on word frequency vectors.
4. **Structural check**: Parse for expected formats (numbers, code, lists). Match structural patterns rather than exact text.
5. **Ensemble / hybrid**: Combine multiple strategies with learned weights. Use task type to select the best strategy automatically.
6. **Partial credit**: Give graduated scores for near-misses (e.g., "43" when expected "42" gets 0.3 not 0.0).

## Workflow
1. Run `python benchmark.py` to see the current score.
2. Analyze which tasks the baseline gets wrong (low correlation with human scores).
3. Edit `harness.py` to improve a strategy.
4. Run benchmark again. If score improves, keep the change. If not, revert.
5. Repeat until diminishing returns.

## Constraints
- Must remain a single `harness.py` file.
- No external dependencies beyond Python stdlib.
- All strategies must complete within reasonable time (speed is 20% of fitness).
- The `grade()` method signature must not change.
