# Contamination Resistance

## Goal

Improve the `ContaminationGuard` class in `harness.py` to maximize the benchmark score. The benchmark tests creating task variants that prevent memorization while preserving the tested capability.

## Benchmark

`benchmark.py` uses 20 tasks (arithmetic, word problems, sequences, conversions, algebra) and generates 3 variants of each using different seeds. It evaluates:

- **Semantic preservation**: tags preserved, structural similarity, verify_equivalence returns True
- **Surface diversity**: prompts change from original, numbers change, variants differ from each other
- **Anti-memorization**: a memorization agent (exact prompt->answer lookup) fails on variants, and answers differ from originals

The fitness signal is:
```
score = 0.4 * semantic_preservation + 0.3 * surface_diversity + 0.3 * anti_memorization
```

## Baseline

The current baseline replaces all integers in the prompt/answer with different integers (deterministic via seed hashing). It verifies equivalence by checking that the non-numeric word structure matches.

Baseline score: **~0.85**

## What to improve

1. **Anti-memorization** (~0.55): The baseline changes numbers in prompts (preventing exact lookup) but does NOT update the expected answer to match the new numbers. For arithmetic tasks, changing "15 + 27" to "38 + 91" should also change the answer from "42" to "129".

2. **Surface diversity** (~0.94): Already good, but could improve by:
   - Changing word order or phrasing, not just numbers
   - Using synonyms ("What is" -> "Calculate", "Compute")
   - Varying sentence structure

3. **Semantic preservation** (~1.0): Already perfect. Must maintain this while improving other dimensions.

### Improvement directions

1. **Answer recomputation**: When numbers change in arithmetic/algebra tasks, recompute the expected answer. Detect the operation from the prompt (+, -, *, /) and apply it to the new numbers.

2. **Phrasing variation**: Maintain a small bank of equivalent phrasings ("What is X + Y?" = "Calculate X + Y" = "Compute the sum of X and Y"). Rotate based on seed.

3. **Structural transformation**: For word problems, swap the nouns ("apples" -> "oranges", "store" -> "shop") while preserving the mathematical structure.

4. **Better verify_equivalence**: Check that both tasks require the same operations/skills rather than just structural similarity.

## Constraints

- Only modify `harness.py`
- Do not modify `benchmark.py`
- No external dependencies (stdlib only)
- No LLM calls -- the logic must be purely algorithmic
- The class must keep the same method signatures

## Running

```bash
python benchmark.py
```

Score is printed as `score: X.XXXXXX` on the last line.
