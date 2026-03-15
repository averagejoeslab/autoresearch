# Automated Grading — Autoresearch Program

## Objective

Improve the `AutoGrader` class in `harness.py` so that it achieves the highest possible composite score on the benchmark. The benchmark measures:

1. **Pearson correlation** (40% weight): How well do auto-grader scores correlate with human scores? (Mapped from [-1,1] to [0,1]; negative correlation is penalized.)
2. **Mean absolute error** (30% weight): Reported as `1 - MAE` — lower error means higher score.
3. **Ranking agreement** (30% weight): Kendall's tau — does the auto-grader rank outputs in the same order as humans? (Mapped from [-1,1] to [0,1].)

## Setup

```bash
cd harnesses/verification/automated-grading
python benchmark.py
```

The benchmark prints `score: X.XXXXXX` at the end. Your goal is to maximize this score.

## What you can modify

- **`harness.py`** — the `AutoGrader` class and any helpers in this file. This is the ONLY file you may edit.
- You may add new methods, scoring heuristics, keyword tables, etc.
- You must preserve the public API: `grade(output: str, task: str, rubric: dict) -> dict`
- The returned dict must have keys: `"score"` (float 0-1), `"rationale"` (str), `"criteria_scores"` (dict[str, float]).

## What you CANNOT modify

- **`benchmark.py`** — this is the locked evaluation. Do not edit it.
- The `contracts/` package.

## Baseline

The baseline scores based on two heuristics:
1. **Keyword overlap** (50%): Fraction of non-stopword task words found in the output.
2. **Length score** (50%): Ramps up to 200 chars, plateaus at 200-2000, penalizes beyond 2000.

All criteria get the same score (undifferentiated). This produces some correlation with human scores (longer, more relevant outputs tend to be better) but misses semantic quality.

## Dataset Characteristics

The dataset has 30 outputs across 10 different tasks, with 3 quality levels each:
- **High quality** (human score >= 0.8): Detailed, accurate, complete answers.
- **Medium quality** (human score 0.4-0.7): Partial or superficial answers.
- **Low quality** (human score < 0.3): Minimal, wrong, or irrelevant answers.

The rubric has three weighted criteria:
- **Correctness** (40%): Factual accuracy and logical soundness.
- **Completeness** (30%): All parts of the task are addressed.
- **Clarity** (30%): Clear, well-organized, easy to understand.

## Experimentation Ideas

- **Specificity scoring**: Count specific details (numbers, proper nouns, technical terms) — high-quality answers tend to have more.
- **Structure detection**: Look for structured elements (lists, code blocks, step-by-step explanations) that indicate thoroughness.
- **Sentence count**: Longer answers with multiple sentences score higher on completeness.
- **Code quality heuristics**: For code outputs, check for function definitions, comments, edge case handling.
- **Vocabulary richness**: Type-token ratio or unique word count as a proxy for quality.
- **Criterion-specific scoring**: Score correctness, completeness, and clarity with different heuristics rather than one score for all.
- **Answer length calibration**: Map output length to score using a learned curve rather than a linear ramp.
- **Negative indicators**: Detect signs of low quality (very short, missing the topic, just a single word).

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
- No LLM calls — the grading must be purely algorithmic.
- `score` must be a float in [0, 1].
- `criteria_scores` must have an entry for each criterion in the rubric.
- `rationale` must be a non-empty string.
