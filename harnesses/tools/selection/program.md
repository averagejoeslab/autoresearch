# Tool Selection – Autoresearch Program

## Objective

Improve the `ToolSelector` class in `harness.py` so that it:

1. Correctly decides **whether a tool is needed** (`should_use_tool`).
2. **Ranks the correct tool first** when a tool is needed (`select`).

The benchmark has 40 tasks: 25 need a specific tool, 15 do not need any
tool.

## Fitness Signal

```
score = 0.4 * selection_accuracy + 0.3 * (1 - false_positive_rate) + 0.3 * mrr
```

Higher is better.  Maximum possible score is 1.0.

- **selection_accuracy** – fraction of all 40 tasks handled correctly
  (right tool selected, or correctly identified as not needing a tool).
- **false_positive_rate** – fraction of no-tool tasks where
  `should_use_tool` incorrectly returned True.
- **mrr** – mean reciprocal rank of the correct tool across tasks that
  need one (1.0 for no-tool tasks correctly classified).

## What You May Change

- **`harness.py`** – the `ToolSelector` class entirely.

## What You Must NOT Change

- **`benchmark.py`** – locked.
- The public API:
  - `select(task: str, available_tools: list[dict]) -> list[dict]`
  - `should_use_tool(task: str) -> bool`
  - Each dict in the returned list must have a `name` field.

## Key Insight

The baseline always returns `True` for `should_use_tool`, so it gets 0%
on the 15 no-tool tasks.  That alone tanks the score.  A quick win is
building a classifier that can tell "knowledge / creative / reasoning"
tasks from "action / lookup / computation" tasks.

## Approach Suggestions

1. **Run the benchmark** to get the baseline score and identify failures.
2. **Improve `should_use_tool`**:
   - Look for action verbs: "search", "read", "write", "send",
     "calculate", "run", "translate", "schedule", "query", "analyze".
   - If none found, the task is likely a knowledge / creative task.
   - Check for keywords like "explain", "describe", "summarize",
     "compare", "write a haiku", "tell me" which suggest no tool needed.
3. **Improve `select`**:
   - TF-IDF or weighted keyword matching.
   - Give higher weight to tool name matches.
   - Use parameter names as additional signal.
4. **Iterate** – run benchmark after each change.

## Constraints

- No network calls.
- No external files.
- Python standard library + numpy available.
