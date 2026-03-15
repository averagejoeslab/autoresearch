# Task Generation

## Goal

Improve the `TaskGenerator` class in `harness.py` to maximize the benchmark score. The benchmark is a meta-benchmark: it tests whether generated tasks can discriminate between simulated agents of different quality levels.

## Benchmark

`benchmark.py` runs 9 generation configs across domains (geography, arithmetic, color-theory, general) and difficulty levels (1-5). For each config, it:

1. Generates N tasks using your `TaskGenerator`
2. Tests 3 simulated agents against the tasks:
   - **Perfect agent**: always returns the exact expected answer
   - **Mediocre agent**: correct only for short/simple prompts, wrong for longer ones
   - **Bad agent**: almost always answers "I don't know"
3. Measures discrimination, difficulty correlation, and task validity

The fitness signal is:
```
score = 0.4 * discrimination + 0.3 * difficulty_correlation + 0.3 * validity
```

## Baseline

The current baseline fills templates with pre-defined factual Q&A pairs ("What is the capital of {country}?", "What is {n} + {m}?", etc.).

Baseline score: **~0.96**

## What to improve

The baseline scores high because the template tasks are valid and the simulated agents are designed to produce score spread. But the score ceiling exists because:

1. **Limited domain coverage**: Only geography, arithmetic, and color-theory templates exist. The "general" domain falls back to all templates.

2. **No difficulty variation**: All tasks within a batch get the same difficulty regardless of the `difficulty` parameter. Tasks at difficulty 5 should genuinely be harder.

3. **Template monotony**: Tasks follow only 3 patterns. More diverse task structures would be more discriminative.

### Improvement directions

1. **Difficulty-scaled templates**: Create templates that naturally scale. E.g., arithmetic: difficulty 1 = "2+3", difficulty 5 = "23*47+19". The mediocre agent will fail harder tasks.

2. **More domains**: Add templates for logic, language, ordering, comparison, etc.

3. **Answer format variation**: Mix in tasks with multi-word answers, boolean answers, and list answers. This makes the mediocre agent more likely to fail.

4. **Parameterized complexity**: Generate tasks whose complexity (prompt length, number of steps, answer format) scales with the difficulty parameter.

## Constraints

- Only modify `harness.py`
- Do not modify `benchmark.py`
- No external dependencies (stdlib only)
- No LLM calls -- the logic must be purely algorithmic
- The class must keep the same method signatures
- Tasks must have well-defined expected_answer values (the simulated agents do exact string matching)

## Running

```bash
python benchmark.py
```

Score is printed as `score: X.XXXXXX` on the last line.
