# Error Recovery — Autoresearch Program

## Objective

Improve the `ErrorRecovery` class in `harness.py` so that it achieves the highest possible composite score on the benchmark. The benchmark measures:

1. **Detection F1** (40% weight): Precision and recall of detecting errors in outputs.
2. **Fix accuracy** (30% weight): Does `suggest_fix` produce output matching the ground-truth corrected version?
3. **False alarm rate inverse** (30% weight): Avoiding false positives on clean outputs (1 - false positive rate).

## Setup

```bash
cd harnesses/verification/error-recovery
python benchmark.py
```

The benchmark prints `score: X.XXXXXX` at the end. Your goal is to maximize this score.

## What you can modify

- **`harness.py`** — the `ErrorRecovery` class and any helpers in this file. This is the ONLY file you may edit.
- You may add new methods, patterns, heuristics, etc.
- You must preserve the public API:
  - `detect_errors(output: str, task: str) -> list[dict]` — each dict has `"type"`, `"location"`, `"message"`.
  - `suggest_fix(output: str, error: dict) -> str` — returns the corrected output string.

## What you CANNOT modify

- **`benchmark.py`** — this is the locked evaluation. Do not edit it.
- The `contracts/` package.

## Baseline

The baseline detects three error categories:
1. **Unmatched delimiters**: Counts `()`, `[]`, `{}` pairs.
2. **Incomplete markers**: Finds `TODO`, `FIXME`, `XXX`, `HACK` in the output.
3. **Placeholders**: Regex for `[INSERT ...]`, `[FILL ...]`, `[YOUR ...]`, `[PLACEHOLDER ...]`.

The baseline `suggest_fix` returns the output unchanged (detection only, no correction). This means fix_accuracy will be low.

## Error Types in the Dataset

- **Unmatched delimiters**: Missing closing `}`, `)`, `]` in JSON, code, SQL.
- **Incomplete markers**: TODO/FIXME/XXX/HACK left in code.
- **Placeholder text**: `[INSERT ...]`, `[FILL ...]`, `[YOUR ...]` patterns.
- **Multiple error types**: Some outputs have more than one error category.
- **Clean outputs**: Some outputs have no errors at all (must not false-alarm).

## Experimentation Ideas

- **Delimiter auto-fix**: When unmatched delimiters are found, append the missing closing delimiter.
- **Marker removal**: Strip TODO/FIXME lines or replace them with meaningful content.
- **Placeholder replacement**: Replace placeholder patterns with sensible defaults.
- **Context-aware fixes**: Use the task description to generate better replacement text.
- **JSON/code parsing**: Try to parse structured outputs and fix syntax errors.
- **Confidence thresholds**: Only report errors above a confidence threshold to reduce false alarms.

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
- No LLM calls — the detection and fixing must be purely algorithmic.
- Error dicts must have `type`, `location`, and `message` keys.
- `suggest_fix` must return a string (the corrected output).
