# Difficulty Calibration

## Goal

Improve the `DifficultyCalibrater` class in `harness.py` to maximize the benchmark score. The benchmark tests two abilities:

1. **calibrate(tasks, results) -> tasks** -- re-assigning difficulty based on actual agent performance
2. **predict_difficulty(task) -> float** -- predicting difficulty for new tasks without performance data

## Benchmark

`benchmark.py` uses 20 tasks with known true solve rates (easy: 90%+, medium: 40-60%, hard: <20%). It simulates 5 agents of varying skill levels generating performance data, then evaluates:

- **Ranking correlation** (Spearman): do calibrated difficulties rank tasks the same way as true solve rates?
- **Prediction accuracy**: does predict_difficulty correlate with actual difficulty?
- **Distribution uniformity**: are calibrated difficulties spread across 1-5, not clustered?

The fitness signal is:
```
score = 0.5 * ranking_correlation + 0.3 * prediction_accuracy + 0.2 * distribution_uniformity
```

## Baseline

The current baseline:
- **calibrate**: maps solve_rate directly to difficulty via `round(1 + 4 * (1 - solve_rate))`
- **predict_difficulty**: uses prompt word count as a proxy (longer = harder)

Baseline score: **~0.81**

## What to improve

1. **Prediction accuracy** (~0.59): Word count is a weak proxy. Better features:
   - Presence of technical jargon ("theorem", "derive", "implement", "complexity")
   - Question complexity (multi-part questions are harder)
   - Mathematical notation or formulas
   - Abstract vs. concrete concepts

2. **Ranking correlation** (~0.94): Already good, but could improve by:
   - Smoothing solve rates across similar agents
   - Bayesian updating when sample sizes are small

3. **Distribution uniformity** (~0.83): The calibration clusters too many tasks at the extremes (1 or 5). Consider:
   - Quantile-based binning instead of linear mapping
   - Forcing a more uniform distribution across difficulty levels

### Improvement directions

1. **Feature-based difficulty prediction**: Extract features from the prompt (question words, technical terms, sentence count, formula presence) and combine them into a difficulty score.

2. **Quantile calibration**: Rank tasks by solve rate and assign difficulties to create roughly equal-sized bins.

3. **Agent-weighted solve rates**: Weight solve attempts by agent quality (expert solves count less for difficulty estimation than novice failures).

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
