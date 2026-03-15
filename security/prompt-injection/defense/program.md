# Prompt Injection Defense — Autoresearch Program

## Objective

Build a robust injection detector by iterating on the `InjectionDetector` class in `harness.py`. The detector uses a **BERT-based classifier** as its core engine. The meta-agent iterates on the full pipeline: model selection, preprocessing, threshold tuning, and ensemble strategies.

## The Score

```
score = 0.40 * detection_f1            — overall precision/recall balance
      + 0.25 * (1 - false_positive_rate) — avoiding blocking legitimate text
      + 0.20 * advanced_detection_rate   — catching sophisticated Tier 3 attacks
      + 0.15 * technique_coverage        — detecting diverse attack categories
```

The `advanced_detection_rate` (20% weight) specifically measures Tier 3 attacks: homoglyphs, zero-width characters, fake conversations, XML injection, code comment injection. **This is where real improvement happens.**

## Setup

```bash
# Install BERT dependencies
pip install transformers torch

# Run benchmark (downloads model on first run)
cd security/prompt-injection/defense
python benchmark.py
```

If `transformers`/`torch` are not installed, the harness falls back to a heuristic detector (lower baseline score).

## Architecture

The detector pipeline has 4 stages, all modifiable:

```
Input text
    │
    ▼
[1. Preprocess]      ← Unicode normalization, encoding detection, cleaning
    │
    ▼
[2. BERT Classify]   ← HuggingFace text-classification pipeline
    │
    ▼
[3. Post-process]    ← Heuristic boosts, ensemble signals, confidence adjustment
    │
    ▼
Detection result
```

## What you can modify

- **`harness.py`** — the `InjectionDetector` class. This is the ONLY file you may edit.
- You must preserve: `detect(text) -> {"is_injection": bool, "confidence": float, "technique": str, "signals": list}`

### Tunable knobs

| Parameter | Current | What it does |
|-----------|---------|-------------|
| `MODEL_NAME` | `protectai/deberta-v3-base-prompt-injection-v2` | Which HuggingFace model to use |
| `THRESHOLD` | `0.5` | Classification confidence threshold |
| `MAX_LENGTH` | `512` | Max tokens for BERT input |
| `_preprocess()` | Strip whitespace | Preprocessing before classification |
| `_postprocess()` | Pass-through | Post-processing / ensemble signals |

## What you CANNOT modify

- **`benchmark.py`** — locked evaluation corpus and scoring.

## Attack Corpus

30 attacks across 3 difficulty tiers + 15 legitimate inputs:

- **Tier 1 — Basic** (10): Direct keyword-based injections
- **Tier 2 — Intermediate** (10): Social engineering, encoding, embedded instructions
- **Tier 3 — Advanced** (10): Homoglyphs, zero-width chars, XML injection, fake conversations, code comment injection
- **Legitimate** (15): Security research discussions, code with "ignore" in names, base64 questions

## Experimentation Ideas

### Model selection
- Try different models: `deepset/deberta-v3-base-injection`, `laiyer/deberta-v3-base-prompt-injection`, `fmops/distilbert-prompt-injection`
- Compare model sizes vs accuracy tradeoffs
- Try fine-tuning on the specific attack corpus (if you generate additional training data)

### Preprocessing improvements
- **Unicode normalization**: NFKC normalize to catch homoglyphs (Cyrillic а → Latin a)
- **Zero-width character stripping**: Remove `\u200b`, `\u200c`, `\u200d`, `\ufeff`
- **Encoding detection**: Detect and decode base64/hex before classification
- **Structural normalization**: Strip HTML/XML/markdown formatting

### Post-processing / ensemble
- **Heuristic boosts**: If BERT is uncertain but heuristic signals are strong, boost confidence
- **Structural analysis**: Detect fake delimiters, conversation boundaries, XML injection patterns
- **Multi-model ensemble**: Run 2-3 small models, vote on classification
- **Confidence calibration**: Adjust thresholds per technique category

### Reducing false positives
- **Context-aware thresholds**: Higher threshold for text that mentions security topics
- **Negative signals**: Academic language, code context, documentation references → reduce confidence
- **Two-stage: coarse filter + fine classifier**: Quick check first, expensive model only if uncertain

## Experimentation Loop

LOOP FOREVER:

1. Run `python benchmark.py` to get current score.
2. Check which attacks are missed (especially Tier 3) and which cause false positives.
3. Modify `harness.py` — change model, preprocessing, thresholds, or ensemble.
4. Run benchmark again.
5. If improved → commit and keep. If not → revert.
6. Never stop.

## Logging

```
commit	score	status	description
a1b2c3d	0.466000	keep	baseline keyword heuristic (no BERT)
b2c3d4e	0.720000	keep	added protectai/deberta-v3-base-prompt-injection-v2
c3d4e5f	0.780000	keep	added Unicode NFKC normalization for homoglyph detection
d4e5f6g	0.810000	keep	ensemble: BERT + zero-width char detection + delimiter heuristic
```
