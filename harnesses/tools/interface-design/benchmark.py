#!/usr/bin/env python3
"""
Benchmark: Tool Interface Design
==================================
Evaluates whether ToolDescriber's enhanced descriptions improve tool
selection accuracy when a simple keyword-matching "agent" picks tools.

10 tools, 30 tasks with ground-truth correct tool labels.

Fitness = 0.6 * selection_accuracy + 0.2 * mean_reciprocal_rank + 0.2 * description_conciseness

Run:  python benchmark.py
"""

from __future__ import annotations

import os
import sys

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from harness import ToolDescriber  # noqa: E402

# ====================================================================
# 10 raw tool specifications
# ====================================================================
RAW_TOOLS: list[dict] = [
    {
        "name": "web_search",
        "description": "Search the web for information",
        "parameters": {"query": "search query string"},
        "category": "information",
    },
    {
        "name": "calculator",
        "description": "Perform mathematical calculations",
        "parameters": {"expression": "math expression to evaluate"},
        "category": "computation",
    },
    {
        "name": "file_reader",
        "description": "Read contents of a file from disk",
        "parameters": {"path": "file path to read"},
        "category": "filesystem",
    },
    {
        "name": "file_writer",
        "description": "Write content to a file on disk",
        "parameters": {"path": "file path", "content": "text to write"},
        "category": "filesystem",
    },
    {
        "name": "email_sender",
        "description": "Send an email message",
        "parameters": {"to": "recipient address", "subject": "email subject", "body": "email body"},
        "category": "communication",
    },
    {
        "name": "code_executor",
        "description": "Execute a snippet of code and return output",
        "parameters": {"language": "programming language", "code": "source code"},
        "category": "computation",
    },
    {
        "name": "database_query",
        "description": "Run a SQL query against the database",
        "parameters": {"sql": "SQL query string", "database": "database name"},
        "category": "data",
    },
    {
        "name": "image_generator",
        "description": "Generate an image from a text prompt",
        "parameters": {"prompt": "image description", "size": "image dimensions"},
        "category": "creative",
    },
    {
        "name": "translator",
        "description": "Translate text between languages",
        "parameters": {"text": "text to translate", "source_lang": "source language", "target_lang": "target language"},
        "category": "language",
    },
    {
        "name": "calendar_manager",
        "description": "Create, read, and manage calendar events",
        "parameters": {"action": "create/read/delete", "event": "event details"},
        "category": "productivity",
    },
]

assert len(RAW_TOOLS) == 10

# ====================================================================
# 30 tasks with ground-truth tool labels
# ====================================================================
TASKS: list[dict] = [
    # web_search (6 tasks)
    {"task": "Find the current weather in Tokyo", "correct_tool": "web_search"},
    {"task": "Look up the population of Brazil", "correct_tool": "web_search"},
    {"task": "Search for recent news about artificial intelligence", "correct_tool": "web_search"},
    {"task": "Who won the Nobel Prize in Physics last year", "correct_tool": "web_search"},
    {"task": "Find restaurants near Central Park", "correct_tool": "web_search"},
    {"task": "What is the stock price of Apple today", "correct_tool": "web_search"},

    # calculator (3 tasks)
    {"task": "What is 1547 multiplied by 382", "correct_tool": "calculator"},
    {"task": "Calculate the compound interest on 10000 at 5 percent for 3 years", "correct_tool": "calculator"},
    {"task": "Compute the square root of 2048", "correct_tool": "calculator"},

    # file_reader (3 tasks)
    {"task": "Show me the contents of config.yaml", "correct_tool": "file_reader"},
    {"task": "Open and display the README file", "correct_tool": "file_reader"},
    {"task": "Read the log file at /var/log/app.log", "correct_tool": "file_reader"},

    # file_writer (3 tasks)
    {"task": "Save this report as output.txt", "correct_tool": "file_writer"},
    {"task": "Create a new file called notes.md with the meeting summary", "correct_tool": "file_writer"},
    {"task": "Write the results to a CSV file", "correct_tool": "file_writer"},

    # email_sender (3 tasks)
    {"task": "Send an email to john@example.com about the project update", "correct_tool": "email_sender"},
    {"task": "Email the team about the meeting tomorrow", "correct_tool": "email_sender"},
    {"task": "Notify sarah@company.com that the report is ready", "correct_tool": "email_sender"},

    # code_executor (3 tasks)
    {"task": "Run this Python script and show the output", "correct_tool": "code_executor"},
    {"task": "Execute the following JavaScript code", "correct_tool": "code_executor"},
    {"task": "Test this function by running it with sample inputs", "correct_tool": "code_executor"},

    # database_query (3 tasks)
    {"task": "Get all users from the database who signed up last month", "correct_tool": "database_query"},
    {"task": "Query the orders table for transactions over 1000 dollars", "correct_tool": "database_query"},
    {"task": "Count the number of active subscriptions in the SQL database", "correct_tool": "database_query"},

    # image_generator (2 tasks)
    {"task": "Create a picture of a sunset over mountains", "correct_tool": "image_generator"},
    {"task": "Generate an illustration of a robot reading a book", "correct_tool": "image_generator"},

    # translator (2 tasks)
    {"task": "Translate this paragraph from English to Spanish", "correct_tool": "translator"},
    {"task": "Convert the following Japanese text to English", "correct_tool": "translator"},

    # calendar_manager (2 tasks)
    {"task": "Schedule a meeting for next Tuesday at 2pm", "correct_tool": "calendar_manager"},
    {"task": "Show my calendar events for this week", "correct_tool": "calendar_manager"},
]

assert len(TASKS) == 30


# ====================================================================
# Simple keyword-matching agent (simulates tool selection)
# ====================================================================

def _tokenize(text: str) -> set[str]:
    """Lowercase, split into word set."""
    return set(text.lower().split())


def _select_tool(task_description: str, tools: list[dict]) -> list[tuple[str, float]]:
    """Return tools ranked by keyword overlap with task.

    Returns list of (tool_name, score) sorted descending.
    """
    task_words = _tokenize(task_description)
    scored: list[tuple[str, float]] = []

    for tool in tools:
        text = f"{tool.get('name', '')} {tool.get('description', '')} "
        # Include any extra fields the describer might add
        for extra_key in ("examples", "usage_hints", "category", "synonyms", "keywords"):
            val = tool.get(extra_key)
            if isinstance(val, str):
                text += f" {val}"
            elif isinstance(val, list):
                text += " " + " ".join(str(v) for v in val)

        # Also include parameter names / descriptions
        params = tool.get("parameters", {})
        if isinstance(params, dict):
            for pk, pv in params.items():
                text += f" {pk} {pv}" if isinstance(pv, str) else f" {pk}"

        tool_words = _tokenize(text)
        overlap = len(task_words & tool_words)
        score = overlap / max(len(task_words), 1)
        scored.append((tool["name"], score))

    scored.sort(key=lambda t: -t[1])
    return scored


# ====================================================================
# Benchmark runner
# ====================================================================


def run_benchmark() -> float:
    describer = ToolDescriber()

    # Enhance all tool descriptions
    enhanced_tools = [describer.describe(tool) for tool in RAW_TOOLS]

    correct_count = 0
    reciprocal_ranks: list[float] = []
    total_desc_length = 0

    for task in TASKS:
        ranking = _select_tool(task["task"], enhanced_tools)
        correct_tool = task["correct_tool"]

        # Top-1 accuracy
        if ranking and ranking[0][0] == correct_tool:
            correct_count += 1

        # Mean Reciprocal Rank
        rr = 0.0
        for rank_idx, (name, _score) in enumerate(ranking):
            if name == correct_tool:
                rr = 1.0 / (rank_idx + 1)
                break
        reciprocal_ranks.append(rr)

    # Description conciseness: ratio of original length to enhanced length
    # (shorter enhanced = better, but minimum is original length)
    orig_total = sum(len(str(t)) for t in RAW_TOOLS)
    enhanced_total = sum(len(str(t)) for t in enhanced_tools)
    # Conciseness: 1.0 if same length or shorter, drops toward 0 as descriptions get 5x longer
    if enhanced_total <= orig_total:
        conciseness = 1.0
    else:
        ratio = enhanced_total / max(orig_total, 1)
        conciseness = max(0.0, 1.0 - (ratio - 1.0) / 4.0)

    n = len(TASKS)
    selection_accuracy = correct_count / n
    mrr = sum(reciprocal_ranks) / n

    fitness = 0.6 * selection_accuracy + 0.2 * mrr + 0.2 * conciseness

    print(f"selection_accuracy:   {selection_accuracy:.6f}")
    print(f"mean_reciprocal_rank: {mrr:.6f}")
    print(f"conciseness:          {conciseness:.6f}")
    print(f"enhanced_total_chars: {enhanced_total}")
    print(f"original_total_chars: {orig_total}")
    print(f"score: {fitness:.6f}")

    return fitness


if __name__ == "__main__":
    run_benchmark()
