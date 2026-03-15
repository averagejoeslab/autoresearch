# Indirect Injection Detection — Autoresearch Program

## Objective

Improve the `IndirectInjectionDetector` class in `harness.py` to achieve the highest possible composite score on the benchmark. This is **defensive security research** — understanding how attackers hide injection payloads in documents and tool outputs so we can build better content scanning and sanitization.

The benchmark measures:

1. **Detection recall** (40% weight): Can the scanner find hidden injections across diverse embedding techniques?
2. **False positive rate** (30% weight, inverted): Does the scanner avoid flagging legitimate documents that discuss injection topics?
3. **Sanitization quality** (30% weight): Can the sanitizer neutralize injections while preserving legitimate content?

## Setup

```bash
cd security/indirect-injection
python benchmark.py
```

The benchmark prints `score: X.XXXXXX` at the end. Your goal is to maximize this score.

## What you can modify

- **`harness.py`** — the `IndirectInjectionDetector` class and any helpers. This is the ONLY file you may edit.
- You may add new methods, change detection algorithms, add pattern databases, etc.
- You must preserve the public API:
  - `scan_content(content: str, source: str) -> dict` returning `{"safe": bool, "threats": list[dict], "sanitized": str}`
  - `sanitize(content: str) -> str`

## What you CANNOT modify

- **`benchmark.py`** — this is the locked evaluation. Do not edit it.
- The `contracts/` package.

## Baseline

The baseline detection strategy uses regex pattern matching:
- Scan for common injection phrases (ignore previous instructions, you are now, etc.)
- Strip HTML comments during sanitization
- Replace matched patterns with [REDACTED]

This catches obvious injections but misses:
- Base64-encoded payloads
- Unicode-hidden instructions
- Injections in data fields (JSON, CSV)
- Subtle context manipulation without keyword matches

## Experimentation Ideas

- **Multi-layer scanning**: Check raw text, decoded text (base64, hex), and normalized text (unicode normalization) separately.
- **Source-aware detection**: Use the `source` parameter to apply different scanning strategies for different content types.
- **Structural analysis**: Detect suspicious formatting changes (hidden HTML, zero-width characters, code blocks with instructions).
- **Context-aware sanitization**: Preserve document structure while removing only the injected portions.
- **Entropy detection**: High-entropy strings in otherwise normal text may indicate encoded payloads.
- **Delimiter detection**: Look for fake system/user/assistant delimiters that do not belong in the content type.
- **Semantic boundary analysis**: Detect when text shifts from informational to imperative (commanding the AI).

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
- Sanitized output must preserve the structure and legitimate content of the original document.
- The scanner must handle all content types (markdown, HTML, JSON, CSV, plain text).
