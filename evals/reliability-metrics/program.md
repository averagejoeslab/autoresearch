# Reliability Metrics -- Autoresearch Program

## Goal
Improve the `ReliabilityMeasure` class in `harness.py` so it accurately discriminates between reliable, moderate, flaky, and failing tasks, and recommends appropriate k values for each.

## Fitness Signal
```
fitness = 0.4 * discrimination_power + 0.3 * k_recommendation_quality + 0.3 * metric_stability
```
Run `python benchmark.py` to get the current score.

## What You Can Change
- **Only** edit `harness.py` in this directory.
- Do **not** edit `benchmark.py` -- it is the locked evaluation.
- The class interface must remain:
  - `measure(results: list[list[bool]]) -> dict`
  - `recommend_k(task_results: list[list[bool]]) -> int`
- No LLM/API calls allowed.

## Baseline
The baseline uses pass@1 (single-trial pass rate) and always recommends k=1. It ignores variance, consistency, and flakiness information.

## Improvement Ideas (ranked by likely impact)
1. **Proper pass@k calculation**: Implement the unbiased pass@k estimator: `pass@k = 1 - C(n-c, k) / C(n, k)` where n=total trials, c=correct trials.
2. **Per-task k recommendation**: Analyze per-task variance. High-variance tasks need higher k. Use the pass rate to determine: if pass_rate > 0.9, k=1; if 0.3-0.9, k=3; if 0.1-0.3, k=5; if < 0.1, k=10.
3. **pass^k metric**: Probability that ALL k trials pass: `pass^k = pass_rate^k`. Useful for tasks requiring consistent success.
4. **Variance-based discrimination**: Use variance of trial results to distinguish flaky from consistent tasks.
5. **Confidence intervals**: Compute confidence bounds on pass rates to better distinguish categories.
6. **Aggregate reliability**: Weight by both pass rate AND consistency for a more nuanced aggregate score.

## Workflow
1. Run `python benchmark.py` to see the current score.
2. Note the discrimination power and k recommendation quality scores.
3. Improve `measure()` to better separate reliability categories.
4. Improve `recommend_k()` to suggest appropriate k values.
5. Run benchmark again and iterate.

## Constraints
- Single `harness.py` file, no external dependencies.
- `measure()` must return all keys the benchmark expects.
- `recommend_k()` must return one of: 1, 3, 5, or 10.
