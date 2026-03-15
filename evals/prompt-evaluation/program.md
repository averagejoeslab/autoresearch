# Prompt Evaluation -- Autoresearch Program

## Goal
Improve the `PromptEvaluator` class in `harness.py` so it correctly identifies which of two prompts is better (matching human labels) and assigns quality scores that correlate with human ratings.

## Fitness Signal
```
fitness = 0.5 * comparison_accuracy + 0.3 * quality_correlation + 0.2 * explanation_quality
```
Run `python benchmark.py` to get the current score.

## What You Can Change
- **Only** edit `harness.py` in this directory.
- Do **not** edit `benchmark.py` -- it is the locked evaluation.
- The class interface must remain:
  - `evaluate(prompt, task_type) -> dict` with keys: clarity, specificity, completeness, potential_ambiguity, overall, explanation
  - `compare(prompt_a, prompt_b, task_type) -> str` returning "A" or "B"
- No LLM/API calls allowed.

## Baseline
The baseline scores prompts by length and keyword presence. Longer prompts with more task-specific keywords score higher. This is a reasonable starting point but misses many quality signals.

## Improvement Ideas (ranked by likely impact)
1. **Specificity detection**: Look for concrete details: numbers, examples, constraints, named entities. Vague prompts ("do something") should score low.
2. **Structure analysis**: Check for clear instruction format: numbered steps, bullet points, explicit output format requirements.
3. **Constraint counting**: Count explicit constraints (word count, format, audience, etc.). More constraints = more specific.
4. **Ambiguity detection**: Beyond vague words, check for unclear pronouns, missing context, underspecified requirements.
5. **Task-type specialization**: Different task types have different quality markers. Coding prompts need input/output specs; writing prompts need audience/tone.
6. **Example presence**: Prompts with examples are generally better. Detect "e.g.", "for example", sample input/output.
7. **Explanation richness**: Include specific sub-scores and actionable feedback in the explanation string.

## Workflow
1. Run `python benchmark.py` to see the current score and comparison accuracy.
2. Look at which prompt pairs the evaluator gets wrong.
3. Identify patterns in misclassified pairs.
4. Add or improve quality signals in `evaluate()`.
5. The `compare()` method should benefit automatically if `evaluate()` improves.
6. Run benchmark again and iterate.

## Constraints
- Single `harness.py` file, no external dependencies.
- Explanation must be a non-empty string with quantitative details.
- All quality metrics must be in [0, 1] range.
