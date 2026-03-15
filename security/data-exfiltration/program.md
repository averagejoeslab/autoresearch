# Data Exfiltration Guard — Autoresearch Program

## Objective

Improve the `ExfiltrationGuard` class in `harness.py` to achieve the highest possible composite score on the benchmark. This is **defensive security research** — understanding how agents can be tricked into leaking sensitive data so we can build better output filters.

The benchmark measures:

1. **Leak detection recall** (50% weight): Can the guard detect sensitive data in agent outputs, including obfuscated forms?
2. **False positive rate** (30% weight, inverted): Does the guard avoid flagging outputs that contain similar-looking but non-sensitive data?
3. **Redaction quality** (20% weight): When leaks are found, is the redaction effective while preserving non-sensitive content?

## Setup

```bash
cd security/data-exfiltration
python benchmark.py
```

The benchmark prints `score: X.XXXXXX` at the end. Your goal is to maximize this score.

## What you can modify

- **`harness.py`** — the `ExfiltrationGuard` class and any helpers. This is the ONLY file you may edit.
- You must preserve the public API:
  - `check_output(output: str, sensitive_data: list[str]) -> dict` returning `{"leaked": list[str], "safe": bool, "redacted_output": str}`

## What you CANNOT modify

- **`benchmark.py`** — this is the locked evaluation. Do not edit it.
- The `contracts/` package.

## Baseline

The baseline detection strategy is exact substring matching:
- For each sensitive data item, check if it appears verbatim in the output
- Replace exact matches with [REDACTED]

This catches direct leaks but misses:
- Base64-encoded sensitive data
- Character-by-character spelling out of secrets
- Paraphrased system prompt content
- Partial matches (substrings of sensitive data)

## Experimentation Ideas

- **Normalized matching**: Normalize both output and sensitive data (remove spaces, lowercase) before comparing.
- **Encoding detection**: Decode base64, hex, and other encodings in the output, then check for sensitive data in decoded form.
- **Character-spread detection**: Detect when characters of sensitive data are spread out with separators (spaces, dashes, etc.).
- **Fuzzy matching**: Use edit distance or token overlap to catch paraphrased versions of sensitive data.
- **Pattern-based detection**: Recognize sensitive data TYPES (API keys, SSNs, emails) by their format, not just exact values.
- **Context-aware redaction**: Preserve sentence structure while replacing only the sensitive portions.
- **Similarity scoring**: For longer sensitive strings, compute similarity scores to catch close matches.

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
- No LLM calls — all detection must be purely algorithmic.
- Redaction must preserve the readability of the non-sensitive content.
- The guard must handle multiple leaks in a single output.
