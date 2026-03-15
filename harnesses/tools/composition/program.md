# Tool Composition – Autoresearch Program

## Objective

Improve the `ToolComposer` class in `harness.py` so that it correctly
chains 2-3 tools for multi-step tasks.  The benchmark has 20 tasks, each
with a ground-truth chain specifying the correct tool order and how
parameters should flow between steps.

## Fitness Signal

```
score = 0.4 * chain_accuracy + 0.3 * param_mapping + 0.3 * validity_rate
```

Higher is better.  Maximum possible score is 1.0.

- **chain_accuracy** – fraction of chain steps where the correct tool
  appears in the correct position.
- **param_mapping** – fraction of parameters correctly sourced (from the
  task input vs. from a previous step's output).
- **validity_rate** – fraction of chains that pass `validate_chain`.

## What You May Change

- **`harness.py`** – the `ToolComposer` class entirely.

## What You Must NOT Change

- **`benchmark.py`** – locked.
- The public API:
  - `plan_chain(task: str, available_tools: list[dict]) -> list[dict]`
  - `validate_chain(chain: list[dict]) -> bool`
  - Each chain step must have `tool` (str) and `inputs` (dict).
  - Parameter references use `{"from_step": int, "output_key": str}`.

## How Scoring Works

The benchmark provides a subset of tools for each task and an expected
chain.  For parameter mapping:

- If `param_sources[param] == "task"`, the predicted input should be a
  plain string (literal from the task).
- If `param_sources[param] == N` (int), the predicted input should be a
  dict `{"from_step": N, ...}` referencing step N's output.

## Approach Suggestions

1. **Run the benchmark** to see which chains the baseline gets right.
2. **Understand the pattern**: tasks are phrased as "do X, then Y, then
   Z" – parse the conjunctions ("and", "then", comma) to identify the
   step sequence.
3. **Match steps to tools**: for each identified sub-task, find the best
   matching tool from the available subset.
4. **Wire parameters**: the first step gets inputs from the task; later
   steps get data-carrying inputs from the previous step and config
   inputs from the task.
5. **Data-carrying vs config parameters**: learn which parameters carry
   data forward (e.g., `text`, `data`, `content`, `code`, `body`,
   `expression`) vs. which are configuration (e.g., `format`,
   `target_lang`, `language`, `path`, `to`, `subject`).
6. **Validate**: ensure all `from_step` references point backward.

## Constraints

- No network calls.
- No external files.
- Python standard library + numpy available.
