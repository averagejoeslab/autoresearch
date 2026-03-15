# Strategy Selection — Autoresearch Program

## Objective

Improve the `StrategySelector` class in `harness.py` so that it achieves the highest possible composite score on the benchmark. The benchmark measures:

1. **Classification accuracy** (70% weight): What fraction of tasks get the correct strategy label?
2. **Macro F1** (30% weight): Balanced F1 across all 4 strategy classes (handles class imbalance).

## Setup

```bash
cd harnesses/planning/strategy-selection
python benchmark.py
```

The benchmark prints `score: X.XXXXXX` at the end. Your goal is to maximize this score.

## What you can modify

- **`harness.py`** — the `StrategySelector` class and any helpers in this file. This is the ONLY file you may edit.
- You may add new methods, lookup tables, feature-based rules, etc.
- You must preserve the public API: `select(task: str, task_features: dict) -> str`
- The return value must be one of: `"linear"`, `"tree"`, `"iterative"`, `"decompose"`.

## What you CANNOT modify

- **`benchmark.py`** — this is the locked evaluation. Do not edit it.
- The `contracts/` package.

## Baseline

The baseline always returns `"linear"`. This gets accuracy equal to the fraction of "linear" tasks in the dataset and an F1 of 0 for three of the four classes.

## Strategy Definitions

- **linear**: Simple sequential tasks (low complexity, no dependencies, no creativity, no search needed).
- **tree**: Tasks requiring exploration of alternatives (high complexity, requires search, debugging/investigation).
- **iterative**: Tasks requiring refinement through cycles (creative tasks, optimization, polishing).
- **decompose**: Large tasks that can be broken into parallel subtasks (high complexity, has dependencies).

## Available Features

The `task_features` dict contains:
- `complexity` (int 1-5): How complex the task is.
- `requires_search` (bool): Whether the task requires searching/exploring.
- `has_dependencies` (bool): Whether the task has multiple interdependent parts.
- `is_creative` (bool): Whether the task requires creative or subjective output.
- `time_pressure` (bool): Whether the task is time-sensitive.

## Experimentation Ideas

- **Decision tree**: Build a simple if/else tree based on the feature values.
- **Feature combinations**: `requires_search AND complexity >= 3` → "tree"; `is_creative` → "iterative"; etc.
- **Keyword matching**: Scan the task text for words like "debug", "investigate", "refine", "build", "migrate".
- **Weighted scoring**: Score each strategy based on how many features favor it; pick the highest.
- **Priority rules**: If `has_dependencies` and `complexity >= 4`, always "decompose" regardless of other features.

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
- Return value must be one of the 4 valid strategies.
