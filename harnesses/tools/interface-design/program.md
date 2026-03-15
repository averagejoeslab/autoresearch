# Tool Interface Design – Autoresearch Program

## Objective

Improve the `ToolDescriber` class in `harness.py` so that its enhanced tool
descriptions lead a keyword-matching "agent" to select the correct tool
more often.  The benchmark has 10 tools and 30 tasks; each task has a
ground-truth correct tool.

## Fitness Signal

```
score = 0.6 * selection_accuracy + 0.2 * mean_reciprocal_rank + 0.2 * description_conciseness
```

Higher is better.  Maximum possible score is 1.0.

- **selection_accuracy** – fraction of tasks where the correct tool is
  ranked first.
- **mean_reciprocal_rank** – average of 1/rank for the correct tool.
- **description_conciseness** – penalises descriptions that are much longer
  than the originals (up to 5x before hitting 0).

## What You May Change

- **`harness.py`** – the `ToolDescriber.describe()` method and any helpers.

## What You Must NOT Change

- **`benchmark.py`** – locked.
- The `describe(tool_spec: dict) -> dict` signature.
- The returned dict must contain at least `name`, `description`, and
  `parameters`.

## How the Benchmark Works

The benchmark uses a simple keyword-overlap agent: for each task, it
tokenises the task description and each tool's text (name + description +
parameters + any extra fields like `examples`, `usage_hints`, `keywords`,
`synonyms`, `category`), then picks the tool with the most overlapping
words.

Your `describe()` method receives the raw tool spec and returns an enhanced
version.  You can add fields, expand descriptions, add synonyms, etc. –
anything that helps the keyword matcher pick the right tool.

## Approach Suggestions

1. **Run the benchmark** to see which tasks fail with the pass-through
   baseline.
2. **Study the failing tasks** – look at the task text and understand which
   keywords are missing from the correct tool's description.
3. **Add synonyms / keywords** – for each tool, add an `examples` list or
   `usage_hints` string containing words that users naturally use when
   requesting that tool.
4. **Expand parameter descriptions** – more descriptive parameter text
   gives the keyword matcher more to work with.
5. **Watch conciseness** – every extra word you add reduces the
   conciseness score.  Be targeted: add only high-value keywords.
6. **Avoid collisions** – if you add "file" to the web_search tool's
   description, it might steal tasks meant for file_reader.

## Constraints

- No network calls.
- No external files.
- `describe()` must be deterministic (same input → same output).
- Python standard library only.
