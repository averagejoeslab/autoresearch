# Tool Safety Checker — Autoresearch Program

## Objective

Improve the `ToolSafetyChecker` class in `harness.py` to achieve the highest possible composite score on the benchmark. This is **defensive security research** — understanding how agents can be manipulated into harmful tool use so we can build better safety checks.

The benchmark measures:

1. **Danger detection recall** (40% weight): Can the checker identify destructive commands, unauthorized access, and harmful operations?
2. **False positive rate** (30% weight, inverted): Does the checker allow legitimate but suspicious-looking tool calls?
3. **Scope violation detection** (20% weight): Can the checker detect when tools are used outside authorized scope?
4. **Alternative quality** (10% weight): Does the checker suggest safer alternatives for blocked calls?

## Setup

```bash
cd security/tool-misuse
python benchmark.py
```

The benchmark prints `score: X.XXXXXX` at the end. Your goal is to maximize this score.

## What you can modify

- **`harness.py`** — the `ToolSafetyChecker` class and any helpers. This is the ONLY file you may edit.
- You must preserve the public API:
  - `check_tool_call(tool_name: str, params: dict, context: dict) -> dict` returning `{"safe": bool, "risk_level": str, "reason": str, "suggested_alternative": str|None}`

## What you CANNOT modify

- **`benchmark.py`** — this is the locked evaluation. Do not edit it.
- The `contracts/` package.

## Baseline

The baseline uses a blocklist of dangerous command patterns:
- Regex patterns for destructive commands (rm -rf, DROP TABLE, FORMAT, etc.)
- Simple scope checking against allowed_scope list
- No suggested alternatives provided

This catches obvious destructive commands but:
- Misses contextual dangers (writing to system files, external emails)
- Cannot distinguish safe vs dangerous uses of the same command
- Does not provide helpful alternatives
- False positives on safe operations that match blocklist patterns

## Experimentation Ideas

- **Context-aware analysis**: Use working_directory, user_request, and recent_actions to assess risk in context.
- **Path analysis**: Check if file operations target system directories, user directories, or temp directories.
- **Graduated risk levels**: Distinguish between "critical" (irreversible destruction) and "medium" (potentially harmful but recoverable).
- **Pattern refinement**: Make blocklist patterns more specific to reduce false positives while maintaining recall.
- **Alternative generation**: Suggest safer alternatives (e.g., "rm with --interactive" instead of "rm -rf").
- **Scope inference**: When a tool is out of scope, suggest an in-scope alternative that achieves the same goal.
- **Behavioral analysis**: Use recent_actions to detect suspicious patterns (escalating privilege, unusual sequences).
- **Domain-specific rules**: Different rules for shell, database, email, file, git, and deploy tools.

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

Append all experiment logs to `experiments.log` in this directory.

## Constraints

- No external dependencies beyond Python stdlib.
- No LLM calls — all safety checking must be purely algorithmic.
- The checker must be fast (under 10ms per tool call).
- Risk levels must be one of: "safe", "low", "medium", "high", "critical".
