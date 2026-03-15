# LLM Judge -- Autoresearch Program

## Goal

Optimize the LLM-as-judge harness in `harness.py` so that it produces judgments that closely match human ground-truth evaluations across three domains: code quality, answer accuracy, and instruction following.

LLM-as-judge is used by over half of organizations evaluating agents, but accuracy varies significantly depending on how the judge prompt is designed. This experiment systematically improves the judge by iterating on every configurable knob in the harness.

## Fitness Signal

```
fitness = 0.30 * judgment_accuracy
        + 0.25 * calibration_quality
        + 0.20 * reasoning_quality
        + 0.15 * cross_domain_consistency
        + 0.10 * token_efficiency
```

Run `python benchmark.py` to get the current score.

## What You Can Change

- **Only** edit `harness.py` in this directory.
- Do **not** edit `benchmark.py` -- it is the locked evaluation.
- You may modify any of these configuration knobs:
  - `SYSTEM_PROMPT` -- the system-level instructions to the judge
  - `USER_TEMPLATE` -- the per-evaluation user prompt template
  - `FEW_SHOT_EXAMPLES` -- list of example judgments (input/output/judgment dicts)
  - `USE_CHAIN_OF_THOUGHT` -- whether to instruct the judge to reason step-by-step
  - `RESPONSE_FORMAT` -- output structure: `"json"`, `"yes_no"`, `"numeric_score"`, `"rubric"`
  - `CONFIDENCE_METHOD` -- how confidence is elicited: `"direct"`, `"self_reported"`, `"multi_pass"`
  - `TEMPERATURE` -- sampling temperature for the judge LLM
  - `MAX_JUDGE_TOKENS` -- max tokens for the judge response
- You may also improve the `LLMJudge` class methods: `_build_prompt`, `_parse_response`, and the heuristic/simulated logic.
- The public interface must remain: `judge(agent_output, task, reference) -> {"score": float, "pass": bool, "confidence": float, "reasoning": str}`.
- No external dependencies; pure Python stdlib only.

## Why This Matters

Automated evaluation is the bottleneck for agent development. Without reliable judges, teams cannot iterate quickly on agent quality. Better judge prompts lead to evaluations that agree more closely with human judgment, enabling faster and cheaper development cycles. The core tradeoff: more detailed judge prompts produce better accuracy but cost more tokens per evaluation.

## Concrete Experiments (ranked by likely impact)

1. **Rubric-based system prompt**: Add explicit scoring criteria to `SYSTEM_PROMPT` -- define what a score of 0.0, 0.5, and 1.0 looks like for each domain. This directly improves judgment accuracy.
2. **Few-shot example count and diversity**: Try 0, 2, 3, and 5 examples. Include examples that span pass, fail, and partial-credit cases. Diverse examples improve cross-domain consistency.
3. **Enable chain-of-thought**: Set `USE_CHAIN_OF_THOUGHT = True`. CoT helps the judge reason through ambiguous cases, improving accuracy on hard examples at the cost of more tokens.
4. **Response format tuning**: Compare `"json"` vs `"rubric"` vs `"numeric_score"`. JSON is most parseable. Rubric forces multi-criteria evaluation. Numeric is simplest.
5. **Multi-pass confidence**: Switch `CONFIDENCE_METHOD` to `"multi_pass"` -- the judge evaluates twice and reconciles. This improves calibration quality at the cost of 2x tokens.
6. **Domain-aware prompts**: Add domain-specific criteria to the user template (e.g., "for code: check correctness, efficiency, style; for answers: check factual accuracy, completeness").
7. **Token efficiency**: After other improvements stabilize, trim the prompt. Remove redundant instructions, shorten few-shot examples, aim for the sweet spot of 200-800 estimated tokens.
8. **Calibration instructions**: Add explicit guidance to the system prompt about when to report high vs low confidence. This directly targets the calibration_quality component.

## Workflow

1. Run `python benchmark.py` to see the current score and per-component breakdown.
2. Identify the weakest component (judgment accuracy, calibration, reasoning, consistency, or efficiency).
3. Pick one experiment from the list above that targets that component.
4. Edit `harness.py` to implement the change.
5. Run benchmark again. If the score improves, keep the change. If not, revert.
6. Repeat until diminishing returns.

## Constraints

- Must remain a single `harness.py` file.
- No external dependencies beyond Python stdlib.
- The `judge()` method signature must not change.
- All 30 benchmark examples must be evaluated; no short-circuiting.
