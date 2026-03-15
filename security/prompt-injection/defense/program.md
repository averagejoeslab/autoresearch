# Prompt Injection Defense — Autoresearch Program

## Objective

Build a robust injection detector by iterating on the `InjectionDetector` class in `harness.py`. The detector must catch known AND novel injection techniques while avoiding false positives on legitimate text.

## The Score

```
score = 0.40 * detection_f1            — overall precision/recall balance
      + 0.25 * (1 - false_positive_rate) — avoiding blocking legitimate text
      + 0.20 * advanced_detection_rate   — catching sophisticated attacks
      + 0.15 * technique_coverage        — detecting diverse attack categories
```

The `advanced_detection_rate` specifically measures performance on Tier 3 attacks — these are sophisticated techniques like homoglyphs, zero-width characters, fake conversations, XML injection, code comment injection, and output format manipulation. **This is where keyword detectors fail and where real improvement happens.**

## Setup

```bash
cd security/prompt-injection/defense
python benchmark.py
```

## What you can modify

- **`harness.py`** — the `InjectionDetector` class. This is the ONLY file you may edit.
- You must preserve: `detect(text) -> {"is_injection": bool, "confidence": float, "technique": str, "signals": list}`

## What you CANNOT modify

- **`benchmark.py`** — locked evaluation corpus and scoring.

## Attack Corpus (3 Tiers)

The benchmark tests against 30 attacks across 3 difficulty tiers:

- **Tier 1 — Basic** (10 attacks): Direct keyword-based injections. The baseline catches most of these.
- **Tier 2 — Intermediate** (10 attacks): Social engineering, encoding, embedded instructions, fake memos. Keyword detectors miss ~half.
- **Tier 3 — Advanced** (10 attacks): Homoglyphs, zero-width characters, XML injection, code comment injection, fake conversations, output format manipulation. **Keyword detectors catch almost none.**

Plus 15 legitimate inputs that look suspicious but aren't (security research discussions, code with "ignore" in variable names, base64 questions, etc.).

## Baseline

The baseline uses keyword matching: scan for "ignore previous", "you are now", "disregard", "jailbreak", etc. This catches basic attacks but:
- Misses encoding, homoglyphs, and structural attacks entirely
- Produces false positives on security discussions
- Cannot detect Tier 3 attacks at all

## Experimentation Ideas

### Beyond keywords
- **Character-level analysis**: Detect homoglyphs (Cyrillic characters that look like Latin), zero-width characters, unusual Unicode
- **Structural analysis**: Look for fake delimiters (XML tags, markdown, conversation format), unusual document structure
- **Semantic signals**: Imperative verbs directed at the AI ("you must", "you will"), urgency language, authority claims
- **Negative signals**: Reduce confidence for academic discussion, code context, documentation references
- **Encoding detection**: Detect base64 strings, hex sequences, and other encoded content

### Reducing false positives
- **Context awareness**: "ignore_previous_results()" in code context is not an injection
- **Topic modeling**: Security research discussions mention injection terms but aren't injections
- **Confidence calibration**: Use multi-signal fusion instead of single-keyword triggers

### Catching Tier 3
- **Homoglyph detection**: Check for mixed scripts (Latin + Cyrillic)
- **Zero-width character detection**: Check for invisible Unicode characters
- **Structural injection**: Detect XML/HTML/JSON that tries to redefine context
- **Fake conversation detection**: Spot fabricated assistant turns
- **Code analysis**: Detect instructions hidden in code comments

## Experimentation Loop

LOOP FOREVER:

1. Run `python benchmark.py` to get current score.
2. Analyze which attacks are missed and which cause false positives.
3. Implement a detection improvement targeting the weakest area.
4. Run benchmark again.
5. If improved → commit and keep. If not → revert.
6. Never stop.

## Logging

```
commit	score	status	description
a1b2c3d	0.450000	keep	baseline keyword detector
b2c3d4e	0.520000	keep	added homoglyph detection for Tier 3
```
