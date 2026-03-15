# Plan Recovery — Autoresearch Program

## Objective

Improve the `PlanRecovery` class in `harness.py` so that it achieves the highest possible composite score on the benchmark. The benchmark measures:

1. **Goal preservation** (40% weight): Does the recovery plan still contain the actions needed to achieve the goal?
2. **Failure avoidance** (30% weight): Is the failed step's action absent from the recovery plan?
3. **Plan minimality** (30% weight): Is the recovery plan close to the ideal length (not bloated)?

## Setup

```bash
cd harnesses/planning/recovery
python benchmark.py
```

The benchmark prints `score: X.XXXXXX` at the end. Your goal is to maximize this score.

## What you can modify

- **`harness.py`** — the `PlanRecovery` class and any helpers in this file. This is the ONLY file you may edit.
- You may add new methods, heuristics, etc.
- You must preserve the public API: `recover(original_plan: list[dict], failed_step: int, error: str) -> list[dict]`
- Each returned step must have keys: `"action"` (str), `"description"` (str), `"depends_on"` (list[int]).

## What you CANNOT modify

- **`benchmark.py`** — this is the locked evaluation. Do not edit it.
- The `contracts/` package.

## Baseline

The baseline removes the failed step and remaps dependency indices. Steps that depended on the failed step have that dependency dropped. This preserves the remaining goals and avoids the failure, but does not add alternative steps or adjust descriptions.

## Key Challenges

- When the failed step is a *prerequisite* for later steps, simply removing it may leave dangling references. The baseline handles index remapping.
- Some plans need a *replacement* step (e.g., if "deploy to AWS" fails, substitute "deploy to GCP"). The baseline does not attempt substitution.
- The error message can hint at what went wrong and what alternatives might work.

## Experimentation Ideas

- **Alternative step insertion**: When a step fails, insert a replacement that achieves the same sub-goal differently. Parse the error string for clues.
- **Dependency repair**: If step C depends on failed step B, and B depends on A, make C depend directly on A.
- **Error-aware recovery**: Use keywords in the error message (e.g., "timeout", "permission denied", "not installed") to choose a recovery strategy.
- **Cascade detection**: If removing a step makes downstream steps impossible, remove those too and flag the plan as partially achievable.
- **Plan compaction**: After removal, merge adjacent trivial steps to keep the plan lean.

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
- No LLM calls — the recovery must be purely algorithmic.
- Dependency indices in the output must be valid (non-negative, referencing earlier steps).
- The output plan must not contain the failed step's action.
