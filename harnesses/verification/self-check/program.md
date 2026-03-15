# Self-Check — Autoresearch Program

## Objective

Improve the `SelfChecker` class in `harness.py` so that it achieves the highest possible composite score on the benchmark. The benchmark measures:

1. **Judgment accuracy** (40% weight): Does `is_correct` match the ground-truth label for each (task, output) pair?
2. **Calibration score** (30% weight): Is `confidence` correlated with actual correctness? (High confidence on correct outputs, low confidence on incorrect ones.)
3. **Issue detection F1** (30% weight): Does the checker detect issues when the output is incorrect, and stay silent when it is correct?

## Setup

```bash
cd harnesses/verification/self-check
python benchmark.py
```

The benchmark prints `score: X.XXXXXX` at the end. Your goal is to maximize this score.

## What you can modify

- **`harness.py`** — the `SelfChecker` class and any helpers in this file. This is the ONLY file you may edit.
- You may add new methods, lookup tables, heuristics, etc.
- You must preserve the public API: `check(output: str, task: str) -> dict`
- The returned dict must have keys: `"is_correct"` (bool), `"confidence"` (float 0-1), `"issues"` (list[str]).

## What you CANNOT modify

- **`benchmark.py`** — this is the locked evaluation. Do not edit it.
- The `contracts/` package.

## Baseline

The baseline always returns `{"is_correct": True, "confidence": 0.5, "issues": []}`. This gets 50% judgment accuracy (correct on all correct outputs, wrong on all incorrect outputs), mediocre calibration (constant confidence), and 0.0 issue detection F1 (never detects issues).

## Error Types in the Dataset

The dataset contains outputs with these error types:
- **Factual errors**: Wrong facts (e.g., "capital of Australia is Sydney")
- **Wrong calculations**: Arithmetic mistakes (e.g., "7 * 8 = 54")
- **Logic errors**: Code bugs (infinite recursion, missing base cases)
- **Incomplete answers**: Missing information (e.g., only 5 of 7 continents)
- **Wrong units**: Correct number but wrong unit
- **Conceptual errors**: Fundamentally wrong definitions

## Experimentation Ideas

- **Length heuristic**: Very short outputs are more likely to be wrong or incomplete.
- **Task-output alignment**: Check whether key words from the task appear in the output (if the task asks about "Australia" and the output says "Sydney", that alone is suspicious).
- **Number validation**: For math tasks, extract numbers from the output and validate simple arithmetic.
- **Known-fact lookup**: Build a small lookup table of commonly tested facts (capitals, formulas, definitions).
- **Pattern detection**: Look for signs of incorrectness — outputs that are suspiciously short, outputs that contradict themselves, code with obvious bugs.
- **Confidence calibration**: Vary confidence based on how many quality signals pass or fail.

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
- No LLM calls — the checking must be purely algorithmic.
- `confidence` must be a float in [0, 1].
- `issues` must be a list of strings (can be empty).
