# Tool-Use Training Data -- Autoresearch Program

## Goal
Improve the `ToolDataGenerator` class in `harness.py` so it generates valid, diverse training examples for tool calling with good parameter coverage.

## Fitness Signal
```
fitness = 0.4 * validity_rate + 0.3 * diversity + 0.3 * param_coverage
```
Run `python benchmark.py` to get the current score.

## What You Can Change
- **Only** edit `harness.py` in this directory.
- Do **not** edit `benchmark.py` -- it is the locked evaluation.
- The class interface must remain:
  - `generate_example(tool_spec) -> dict` with keys: user_query, tool_call, tool_spec
  - `validate_example(example) -> bool`
- No LLM/API calls allowed.

## Baseline
The baseline generates "Use {tool_name}" as query and fills all params with type-appropriate placeholders. Validation checks structural completeness. This produces valid but unnatural, non-diverse examples.

## Improvement Ideas (ranked by likely impact)
1. **Natural queries**: Generate queries that describe what the user wants to accomplish, not just "Use {tool_name}". Use the description and parameter names to construct natural language.
2. **Realistic parameter values**: Instead of "placeholder", generate realistic values based on parameter name and type (e.g., "file_path" -> "/home/user/document.txt").
3. **Query templates**: Create a bank of query templates per tool category and vary them across generations.
4. **Parameter variation**: Vary which optional parameters are included across different examples of the same tool.
5. **Context-aware queries**: Use parameter names to inform the query (e.g., if tool has "language" param, mention language in the query).
6. **Validation strictness**: Add more validation checks: parameter type matching, value plausibility, query-tool alignment.

## Workflow
1. Run `python benchmark.py` to see the current score.
2. Note which components (validity, diversity, coverage) are weakest.
3. Improve `generate_example()` for more natural, varied examples.
4. Improve `validate_example()` for more thorough checking.
5. Run benchmark again and iterate.

## Constraints
- Single `harness.py` file, no external dependencies.
- Generated examples must be self-contained (no external data).
- Validation must be deterministic (same input = same output).
- `generate_example()` may be non-deterministic (varying queries is good for diversity).
