# Delegation Decider

## Goal

Improve the `DelegationDecider` class in `harness.py` to maximize the benchmark score. The benchmark evaluates two capabilities:

1. **should_delegate(task, task_features) -> bool** -- deciding whether a task needs delegation
2. **plan_delegation(task, available_agents) -> list[dict]** -- creating effective subtask assignments

## Benchmark

`benchmark.py` runs 25 test cases covering:
- Simple tasks that should NOT be delegated (complexity 1-3)
- Complex tasks that SHOULD be delegated (complexity 4-5)
- Edge cases (empty tasks, borderline complexity)

The fitness signal is:
```
score = 0.5 * decision_accuracy + 0.3 * assignment_quality + 0.2 * efficiency
```

## Baseline

The current baseline:
- Delegates if `complexity > 3` (achieves ~100% decision accuracy on the current task set)
- Assigns the entire task to the first available agent as a single subtask (poor assignment quality)

Baseline score: **~0.93**

## What to improve

The main weakness is **assignment quality** (~0.76). The baseline:
- Creates only 1 subtask instead of breaking the task into meaningful sub-work
- Always assigns to the first agent regardless of capabilities
- Ignores agent specializations entirely

### Improvement directions

1. **Task decomposition in plan_delegation**: Parse the task description to identify distinct sub-activities (e.g., "build API, write docs, and deploy" should become 3 subtasks). Use keyword matching, sentence splitting, or conjunction analysis.

2. **Capability-based agent matching**: Match subtask keywords against agent capabilities. For example, a subtask mentioning "write documentation" should go to an agent with "writing" capability.

3. **Multi-feature delegation decisions**: Use more than just complexity -- incorporate `scope`, `requires_specialization`, and `estimated_time` for borderline cases.

4. **Priority and dependency assignment**: Set meaningful priorities and dependency chains rather than flat structures.

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
