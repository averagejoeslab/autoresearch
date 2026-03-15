# Window Packing — Autoresearch Program

## Objective

Improve the `WindowPacker` class in `harness.py` so that it achieves the highest possible composite score on the benchmark. The benchmark measures three things:

1. **Content coverage** (40% weight): Priority-weighted fraction of original content preserved.
2. **Priority alignment** (30% weight): Higher-priority sections should get higher coverage.
3. **Min section coverage** (30% weight): No section should be starved — the worst-covered section drags this down.

## Setup

```bash
cd harnesses/context/window-packing
python benchmark.py
```

The benchmark prints `score: X.XXXXXX` at the end. Your goal is to maximize this score.

## What you can modify

- **`harness.py`** — the `WindowPacker` class and any helpers in this file. This is the ONLY file you may edit.
- You may add new methods, change the allocation algorithm, add adaptive logic, etc.
- You must preserve the public API: `pack(sections: dict[str, list[dict]], total_budget: int) -> dict[str, list[dict]]`
- Token counting uses `len(content.split())` as a word-count proxy.

## What you CANNOT modify

- **`benchmark.py`** — this is the locked evaluation. Do not edit it.
- The `contracts/` package.

## Baseline

The baseline strategy uses fixed proportions:
- system: 10%, history: 50%, tools: 20%, current: 20%

It does NOT redistribute unused budget from sections that don't need their full allocation. This means budget is wasted when a section is small but allocated a large proportion.

## Section Priorities (from benchmark)

The benchmark uses these priority weights:
- system: 1.0 (highest)
- current: 0.9
- history: 0.7
- tools: 0.6 (lowest)

## Experimentation Ideas

- **Budget redistribution**: If a section uses less than its allocation, redistribute the surplus to other sections proportional to their priority.
- **Demand-based allocation**: First calculate how much each section needs, then allocate proportional to min(need, priority-weighted share).
- **Two-phase packing**: Phase 1 gives each section a minimum guarantee (e.g., 5% each). Phase 2 distributes the remaining budget by priority and demand.
- **Adaptive proportions**: Detect which section has the most content relative to its allocation and shift budget toward it.
- **Priority-weighted greedy**: Instead of fixed proportions, greedily allocate budget to sections in priority order until each section is satisfied or budget runs out.
- **Min-coverage guarantee**: Ensure every section gets at least some representation, then optimize the rest.

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
- No LLM calls — the packing must be purely algorithmic.
- The packed output must respect the total token budget (word count).
- Each section in the output must only contain entries from the corresponding input section.
- Entry order within a section should be preserved (or at minimum, the section should contain a valid subset of its original entries).
