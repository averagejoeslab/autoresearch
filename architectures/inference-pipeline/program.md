# Inference Pipeline Architecture — Autoresearch Program

## Objective

Research how to structure multi-stage inference pipelines for agent runtime classification. This experiment is about **pipeline architecture** — preprocessing, routing, ensemble, calibration, cascading — not individual model quality.

All classifiers are simulated heuristics with realistic strengths and weaknesses. The research question is: **how do you combine them optimally?**

This parallels Karpathy's autoresearch:
- `harness.py` = `train.py` — pipeline configuration, detector implementations, ensemble logic
- `benchmark.py` = `prepare.py` — held-out evaluation, locked, do not modify

## What You're Researching

**Pipeline structure for classification.** In production, you need:

1. **Efficiency** — fast pre-filter catches 80% of cases cheaply, expensive analysis only for hard cases
2. **Accuracy** — ensemble of diverse detectors covers more attack surface than any single detector
3. **Calibration** — confidence scores must correlate with actual accuracy for downstream decision-making
4. **Robustness** — no single point of failure; if one detector is evaded, others compensate

The four simulated detectors have intentionally different strengths:

| Detector | Strength | Weakness |
|----------|----------|----------|
| **keyword** | Fast, high recall on known patterns | Brittle to paraphrasing, high FPR on security text |
| **structural** | Catches delimiter/format tricks | Misses semantic attacks entirely |
| **behavioral** | Catches social engineering, urgency | False positives on assertive writing |
| **statistical** | Catches encoding, obfuscation, homoglyphs | Misses well-formed English attacks |

No single detector can score well alone. The pipeline's job is to combine their outputs intelligently.

## The Score

```
score = 0.30 * detection_f1           — precision/recall balance
      + 0.25 * (1 - false_positive_rate) — avoiding false alarms
      + 0.20 * hard_negative_accuracy  — not flagging legitimate security discussions
      + 0.15 * detector_efficiency     — fewer detectors run = better (rewards cascading)
      + 0.10 * calibration_quality     — confidence correlates with actual accuracy (1 - ECE)
```

The multi-objective scoring rewards pipelines that are accurate, efficient, AND well-calibrated — just like production systems need to be.

## Setup

```bash
# No dependencies needed — pure Python
cd architectures/inference-pipeline
python benchmark.py
```

## The Four Pipeline Stages

### Stage 1: Preprocessing
Text normalization before detection. Options:
- `strip` — whitespace normalization
- `unicode_nfkc` — Unicode compatibility normalization
- `zero_width_strip` — remove zero-width characters
- `homoglyph_normalize` — replace Cyrillic/confusable characters with ASCII
- `lowercase_check` — no-op (placeholder for feature extraction flag)

**Key experiment:** Adding `zero_width_strip` and `homoglyph_normalize` should improve detection of obfuscation attacks (u14, u15) but has no cost. Why not always include them?

### Stage 2: Detectors
Four simulated heuristic classifiers, each with configurable weights:
- `keyword` (default weight 0.3) — regex pattern matching
- `structural` (default weight 0.3) — document structure analysis
- `behavioral` (default weight 0.2) — linguistic/pragmatic analysis
- `statistical` (default weight 0.2) — character distribution anomalies

**Key experiment:** Rebalance weights based on which detectors are most accurate on missed categories.

### Stage 3: Ensemble
How to combine detector outputs. Five strategies:
- `weighted_vote` — weighted average of scores (default)
- `majority` — flag if >50% of detectors flag
- `any_positive` — flag if ANY detector flags (high recall)
- `cascading` — run fast detectors first, skip expensive if confident
- `stacking` — logistic regression over detector outputs

**Key experiment:** Try `cascading` with keyword first — it should boost `detector_efficiency` (worth 0.15 of the score) while maintaining accuracy on easy cases.

### Stage 4: Calibration
Adjust raw confidence for reliability. Options:
- `none` — pass through raw scores
- `platt` — sigmoid rescaling
- `isotonic` — piecewise-linear calibration
- `temperature` — temperature scaling

**Key experiment:** Try `temperature` with T=1.5 to soften overconfident predictions, improving ECE.

## What You Can Modify

**`harness.py`** — everything in this file is fair game:

### Pipeline Configuration
| Parameter | Default | What it controls |
|-----------|---------|-----------------|
| `PREPROCESSING` | `["strip", "lowercase_check"]` | Which preprocessors run (and in what order) |
| `DETECTOR_WEIGHTS` | `{keyword: 0.3, structural: 0.3, behavioral: 0.2, statistical: 0.2}` | Relative detector importance |
| `ENSEMBLE_STRATEGY` | `"weighted_vote"` | How detector outputs are combined |
| `CASCADE_THRESHOLD` | `0.8` | Confidence threshold for cascade short-circuit |
| `CASCADE_ORDER` | `["keyword", "structural", "behavioral", "statistical"]` | Detector execution order in cascade mode |
| `CALIBRATION_METHOD` | `"none"` | Post-hoc calibration method |
| `CALIBRATION_TEMPERATURE` | `1.5` | Temperature parameter (if using temperature scaling) |
| `CONFIDENCE_THRESHOLD` | `0.5` | Final decision threshold |

### Detector Internals
Each detector's patterns, thresholds, and scoring logic can be modified:
- Keyword detector patterns and their individual scores
- Structural detector's delimiter patterns and scoring weights
- Behavioral detector's linguistic patterns and sensitivity
- Statistical detector's anomaly thresholds

### Preprocessor Logic
- Add new preprocessors to `PREPROCESSOR_REGISTRY`
- Modify existing preprocessors (e.g., expand homoglyph map)

### Ensemble and Calibration
- Modify ensemble functions (e.g., stacking coefficients)
- Tune calibration parameters (Platt scaling A/B, isotonic breakpoints, temperature)

## What You CANNOT Modify

- **`benchmark.py`** — held-out test data and scoring. This is the trust boundary.

## Experimentation Ideas

### Quick wins (try first)
1. **Add preprocessors**: Include `zero_width_strip` and `homoglyph_normalize` in `PREPROCESSING` — should catch u14 (homoglyph) and u15 (zero-width) for free
2. **Try cascading**: Set `ENSEMBLE_STRATEGY = "cascading"` — earns efficiency points (0.15 weight in score)
3. **Add temperature calibration**: Set `CALIBRATION_METHOD = "temperature"` — should improve ECE

### Medium effort
4. **Rebalance detector weights**: Analyze which categories are missed, upweight the detector that covers them
5. **Lower confidence threshold**: Try `CONFIDENCE_THRESHOLD = 0.4` to improve recall at the cost of precision
6. **Tune cascade order**: Put the most discriminative detector first in `CASCADE_ORDER`
7. **Try stacking ensemble**: `ENSEMBLE_STRATEGY = "stacking"` with tuned coefficients

### Deep changes
8. **Improve keyword patterns**: Add patterns for missed attack categories (emotional manipulation, completion attacks)
9. **Improve behavioral detector**: Add patterns for emotional manipulation, false urgency
10. **Improve structural detector**: Better code comment injection detection
11. **Tune isotonic calibration**: Adjust breakpoints based on observed score distribution
12. **Add new detector**: Create a fifth detector in `DETECTOR_REGISTRY` (e.g., semantic similarity to known attacks)

## Why Pipeline Architecture Matters

In production agent runtimes:
- **Latency budget**: You can't run every detector on every request. Cascading lets you spend compute only when needed.
- **Detector diversity**: Different attack types need different detection strategies. No single classifier handles everything.
- **Confidence reliability**: Downstream systems (routing, escalation, logging) need calibrated probabilities, not just binary flags.
- **Graceful degradation**: If one detector fails or is evaded, the ensemble still functions.
- **Cost optimization**: The efficiency metric directly maps to real-world compute costs.

## Experimentation Loop

LOOP FOREVER:

1. Run `python benchmark.py` to evaluate the pipeline.
2. Check the output: which categories are missed? Which cause false positives? What's the efficiency?
3. Modify `harness.py` — pipeline config, detector logic, ensemble strategy, calibration.
4. Run benchmark again.
5. If score improved, `git commit` and keep.
6. If score is same or worse, `git reset` and try something different.
7. Never stop.

## Logging

```
commit	score	status	description
a1b2c3d	0.XXXXXX	keep	baseline — weighted_vote, no calibration, minimal preprocessing
b2c3d4e	0.XXXXXX	keep	added zero_width_strip + homoglyph_normalize preprocessing
c3d4e5f	0.XXXXXX	keep	switched to cascading ensemble, keyword-first
d4e5f6g	0.XXXXXX	discard	any_positive ensemble — too many false positives
e5f6g7h	0.XXXXXX	keep	temperature calibration T=1.5, improved ECE
```
