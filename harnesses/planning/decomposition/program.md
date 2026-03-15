# Task Decomposition — Autoresearch Program

## Objective

Improve the `TaskDecomposer` class in `harness.py` so that it achieves the highest possible composite score on the benchmark. The benchmark measures three things:

1. **Step coverage** (40% weight): All necessary steps from the reference decomposition must be present (measured by keyword overlap between produced and reference step actions/descriptions).
2. **Ordering** (30% weight): Dependency indices in each step must reference earlier steps only — no forward references or cycles.
3. **Granularity** (30% weight): The number of steps should be close to the reference count — not too coarse, not too fine.

## Setup

```bash
cd harnesses/planning/decomposition
python benchmark.py
```

The benchmark prints `score: X.XXXXXX` at the end. Your goal is to maximize this score.

## What you can modify

- **`harness.py`** — the `TaskDecomposer` class and any helpers in this file. This is the ONLY file you may edit.
- You may add new methods, change the algorithm, add heuristics, lookup tables, etc.
- You must preserve the public API: `decompose(task: str, context: dict) -> list[dict]`
- Each returned step must have keys: `"action"` (str), `"description"` (str), `"depends_on"` (list[int]).

## What you CANNOT modify

- **`benchmark.py`** — this is the locked evaluation. Do not edit it.
- The `contracts/` package.

## Baseline

The baseline strategy produces 3 generic steps for every task:
1. "prepare" — gather requirements
2. "execute" — carry out the work (depends on step 0)
3. "verify" — validate results (depends on step 1)

This scores reasonably on ordering (dependencies are valid) but poorly on coverage (generic labels miss domain-specific actions) and granularity (3 steps is too coarse for most 5-7 step reference plans).

## Experimentation Ideas

Here are directions to explore (not exhaustive):

- **Keyword-based templates**: Match task keywords (e.g., "deploy", "test", "database", "design") to domain-specific step templates.
- **Context-aware decomposition**: Use the `context["domain"]` field to select from pre-built domain plans.
- **Adaptive granularity**: Estimate task complexity from word count or keyword density, then produce an appropriate number of steps.
- **Dependency graph construction**: Analyze which steps logically precede others (e.g., "test" always follows "implement").
- **Common-prefix patterns**: Many tasks follow setup → implement → test → deploy patterns; build a library of these.
- **Hybrid approach**: Start with a template, then refine based on task-specific keywords.

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
- No LLM calls — the decomposition must be purely algorithmic.
- Each step must have all three required keys: `action`, `description`, `depends_on`.
- Dependency indices must be valid (non-negative, less than the step's own index).
