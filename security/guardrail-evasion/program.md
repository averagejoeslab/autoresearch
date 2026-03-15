# Guardrail Evasion Defense — Autoresearch Program

## Objective

Improve the `GuardrailTester` class in `harness.py` to achieve the highest possible composite score on the benchmark. This is **defensive security research** — understanding how guardrails get bypassed so we can build more robust ones.

The benchmark measures:

1. **Evasion catch rate** (40% weight): Does the hardened guardrail catch obfuscated harmful inputs?
2. **False positive rate** (30% weight, inverted): Does the hardened guardrail allow legitimate security discussions?
3. **Robustness improvement** (30% weight): How much does hardening improve over the base guardrail?

## Setup

```bash
cd security/guardrail-evasion
python benchmark.py
```

The benchmark prints `score: X.XXXXXX` at the end. Your goal is to maximize this score.

## What you can modify

- **`harness.py`** — the `GuardrailTester` class, `base_guardrail`, `HARMFUL_KEYWORDS`, and any helpers. This is the ONLY file you may edit.
- You must preserve the public API:
  - `test_guardrail(guardrail_fn, input_text: str) -> dict`
  - `harden_guardrail(base_guardrail_fn) -> callable` returning a function with the same interface as `base_guardrail`
  - `base_guardrail(text: str) -> dict` returning `{"blocked": bool, "reason": str}`

## What you CANNOT modify

- **`benchmark.py`** — this is the locked evaluation. Do not edit it.
- The `contracts/` package.

## Baseline

The baseline system:
- `base_guardrail`: Exact keyword matching against a blocklist of harmful terms
- `harden_guardrail`: Adds leetspeak normalization (0->o, 3->e, @->a, etc.)

This catches some leetspeak evasions but misses:
- Unicode homoglyph substitutions (Cyrillic 'a' for Latin 'a')
- Word splitting with spaces, dots, dashes, or zero-width characters
- Case variations and mixed obfuscation
- The base guardrail blocks legitimate security discussions (high false positive rate)

## Experimentation Ideas

- **Unicode normalization**: Use Python's `unicodedata.normalize` to handle homoglyph attacks (NFKD decomposition).
- **Zero-width character removal**: Strip zero-width spaces, joiners, and other invisible Unicode characters.
- **Separator removal**: Collapse dots, dashes, spaces within words before checking keywords.
- **Context-aware filtering**: Analyze whether the text is asking "how to attack" vs "how to defend against". Legitimate security discussions typically include defensive framing.
- **Intent classification**: Score inputs on attack-intent signals (imperative mood + harmful topic) vs defensive-intent signals (question about protection, education context).
- **N-gram matching**: Generate character n-grams to catch obfuscated keywords even with inserted characters.
- **Whitelist patterns**: Recognize safe patterns like "prevent X", "protect against X", "defend from X" and allow them even when X is a harmful keyword.

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
- No LLM calls — all guardrail logic must be purely algorithmic.
- The hardened guardrail must maintain the same interface as the base guardrail.
- Focus on reducing false positives as much as increasing true positives.
