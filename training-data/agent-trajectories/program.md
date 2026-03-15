# Agent Trajectory Curation -- Autoresearch Program

## Goal
Improve the `TrajectoryCurator` class in `harness.py` so it accurately scores trajectory quality (correlating with human judgments) and selects diverse, high-quality trajectories within a budget.

## Fitness Signal
```
fitness = 0.5 * scoring_correlation + 0.3 * selection_quality + 0.2 * diversity
```
Run `python benchmark.py` to get the current score.

## What You Can Change
- **Only** edit `harness.py` in this directory.
- Do **not** edit `benchmark.py` -- it is the locked evaluation.
- The class interface must remain:
  - `score_trajectory(trajectory: list[dict]) -> float` returning quality in [0, 1]
  - `select_trajectories(trajectories, budget) -> list[int]` returning indices
- No LLM/API calls allowed.

## Baseline
The baseline scores by inverse step count (1/n_steps). Selection is greedy top-k by score. This captures the "fewer steps = better" signal but ignores success rate, action quality, and diversity.

## Improvement Ideas (ranked by likely impact)
1. **Success rate**: Factor in what fraction of steps succeeded. A 3-step trajectory with all successes is better than a 3-step trajectory with 2 failures.
2. **Final outcome**: Weight the last step's success heavily -- did the trajectory reach its goal?
3. **Efficiency ratio**: Combine step count with success rate: `success_rate / log(n_steps + 1)`.
4. **Error recovery**: Trajectories that fail then recover may be more valuable training data than ones that never fail.
5. **Diverse selection**: Instead of pure greedy, use maximal marginal relevance (MMR) to select trajectories that are both high-quality AND diverse.
6. **Category-aware selection**: Ensure the selection covers different trajectory types (different action patterns, step counts, etc.).
7. **Diminishing returns**: Penalize selecting too many similar trajectories.

## Workflow
1. Run `python benchmark.py` to see the current score.
2. Analyze the scoring correlation -- which trajectories are mis-scored?
3. Improve `score_trajectory()` to account for more quality signals.
4. Improve `select_trajectories()` for better quality + diversity.
5. Run benchmark again and iterate.

## Constraints
- Single `harness.py` file, no external dependencies.
- Score must be in [0.0, 1.0] range.
- Selected indices must be valid (within bounds, no duplicates).
- Selection must respect the budget limit.
