# Prompt Injection Detection — Autoresearch Program

## Objective

Improve the `InjectionCrafter` class in `harness.py` so that it achieves the highest possible composite score on the benchmark. This is **defensive security research** — understanding how prompt injection works so we can build better detection.

The benchmark measures:

1. **Detection F1** (50% weight): Can the detector correctly identify injection attempts while avoiding false positives?
2. **False positive rate** (30% weight, inverted): Does the detector avoid flagging legitimate text as injections?
3. **Technique diversity** (20% weight): Does the system understand multiple injection techniques?

## Setup

```bash
cd security/prompt-injection
python benchmark.py
```

The benchmark prints `score: X.XXXXXX` at the end. Your goal is to maximize this score.

## What you can modify

- **`harness.py`** — the `InjectionCrafter` class and any helpers. This is the ONLY file you may edit.
- You may add new methods, change detection algorithms, add pattern databases, etc.
- You must preserve the public API:
  - `craft_injection(goal: str, context: str) -> str`
  - `detect_injection(text: str) -> dict` returning `{"is_injection": bool, "confidence": float, "technique": str}`

## What you CANNOT modify

- **`benchmark.py`** — this is the locked evaluation. Do not edit it.
- The `contracts/` package.

## Baseline

The baseline detection strategy is keyword matching:
- Scan text for known injection phrases ("ignore previous", "you are now", "forget your instructions", etc.)
- Classify technique based on which keywords matched
- Confidence scales with number of keyword matches

This achieves reasonable recall on obvious injections but:
- Misses obfuscated or subtle injection attempts
- Produces false positives on legitimate text that mentions injection concepts
- Cannot detect encoding-based attacks

## Experimentation Ideas

- **Contextual analysis**: Look at sentence structure, not just keywords. Real injections have imperative mood directed at the AI.
- **Encoding detection**: Check for base64, hex, ROT13, and other encoded payloads that might contain hidden instructions.
- **Delimiter awareness**: Detect fake system/user delimiters that try to redefine conversation boundaries.
- **Semantic scoring**: Weight features like imperative verbs, second-person pronouns targeting the AI, urgency language.
- **Negative signals**: Identify markers of legitimate text (questions, academic discussion, code context) to reduce false positives.
- **Multi-signal fusion**: Combine weak signals that individually are not conclusive but together indicate injection.
- **Pattern diversity**: Build technique-specific detectors that each specialize in one attack vector.

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
- No LLM calls — all detection must be purely algorithmic.
- The detector must return results in under 100ms per input.
- Focus on improving DETECTION quality, not attack quality.
