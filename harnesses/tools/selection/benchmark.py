#!/usr/bin/env python3
"""
Benchmark: Tool Selection
==========================
Evaluates ToolSelector on 40 tasks: some need tools, some do not.

Measures selection accuracy, false-positive rate, and ranking quality.

Fitness = 0.4 * selection_accuracy + 0.3 * (1 - false_positive_rate) + 0.3 * mrr

Run:  python benchmark.py
"""

from __future__ import annotations

import os
import sys

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from harness import ToolSelector  # noqa: E402

# ====================================================================
# Available tools
# ====================================================================
AVAILABLE_TOOLS: list[dict] = [
    {"name": "web_search", "description": "Search the web for current information and facts"},
    {"name": "calculator", "description": "Evaluate mathematical expressions and perform arithmetic"},
    {"name": "file_reader", "description": "Read the contents of files from disk"},
    {"name": "file_writer", "description": "Write or save content to files on disk"},
    {"name": "code_runner", "description": "Execute code snippets in Python or JavaScript"},
    {"name": "database_query", "description": "Run SQL queries against a relational database"},
    {"name": "email_sender", "description": "Send email messages to recipients"},
    {"name": "translator", "description": "Translate text between different natural languages"},
    {"name": "image_analyzer", "description": "Analyze and describe the contents of images"},
    {"name": "calendar", "description": "Create and manage calendar events and schedules"},
]

# ====================================================================
# 40 tasks with ground-truth labels
# ====================================================================
# needs_tool: whether any tool is needed
# correct_tools: ordered list of acceptable tools (first is best)
TASKS: list[dict] = [
    # --- Tasks that NEED a specific tool (25) ---
    {"task": "What is 847 times 293", "needs_tool": True, "correct_tools": ["calculator"]},
    {"task": "Calculate the factorial of 12", "needs_tool": True, "correct_tools": ["calculator"]},
    {"task": "Compute 15 percent of 3200", "needs_tool": True, "correct_tools": ["calculator"]},
    {"task": "What is the current temperature in London", "needs_tool": True, "correct_tools": ["web_search"]},
    {"task": "Search for the latest news about climate change", "needs_tool": True, "correct_tools": ["web_search"]},
    {"task": "Find the population of Nigeria", "needs_tool": True, "correct_tools": ["web_search"]},
    {"task": "Look up the current exchange rate for EUR to USD", "needs_tool": True, "correct_tools": ["web_search"]},
    {"task": "Read the configuration file at /etc/app/config.json", "needs_tool": True, "correct_tools": ["file_reader"]},
    {"task": "Show me what is in the README.md file", "needs_tool": True, "correct_tools": ["file_reader"]},
    {"task": "Open and display the contents of data.csv", "needs_tool": True, "correct_tools": ["file_reader"]},
    {"task": "Save this summary as report.pdf", "needs_tool": True, "correct_tools": ["file_writer"]},
    {"task": "Write these results to output.json", "needs_tool": True, "correct_tools": ["file_writer"]},
    {"task": "Create a new file called notes.txt with today's notes", "needs_tool": True, "correct_tools": ["file_writer"]},
    {"task": "Run this Python script and show the output", "needs_tool": True, "correct_tools": ["code_runner"]},
    {"task": "Execute the following JavaScript function", "needs_tool": True, "correct_tools": ["code_runner"]},
    {"task": "Get all users who signed up in March from the users table", "needs_tool": True, "correct_tools": ["database_query"]},
    {"task": "Query the database for orders over 500 dollars", "needs_tool": True, "correct_tools": ["database_query"]},
    {"task": "Count active subscriptions in the SQL database", "needs_tool": True, "correct_tools": ["database_query"]},
    {"task": "Send an email to alice@example.com about the meeting", "needs_tool": True, "correct_tools": ["email_sender"]},
    {"task": "Email the marketing team the quarterly report", "needs_tool": True, "correct_tools": ["email_sender"]},
    {"task": "Translate this paragraph into French", "needs_tool": True, "correct_tools": ["translator"]},
    {"task": "Convert this Japanese text to English", "needs_tool": True, "correct_tools": ["translator"]},
    {"task": "What objects are in this photograph", "needs_tool": True, "correct_tools": ["image_analyzer"]},
    {"task": "Describe what is shown in the uploaded image", "needs_tool": True, "correct_tools": ["image_analyzer"]},
    {"task": "Schedule a team meeting for Friday at 3pm", "needs_tool": True, "correct_tools": ["calendar"]},

    # --- Tasks that DO NOT need a tool (15) ---
    {"task": "What is the meaning of the word ephemeral", "needs_tool": False, "correct_tools": []},
    {"task": "Explain the concept of recursion in programming", "needs_tool": False, "correct_tools": []},
    {"task": "Write a haiku about autumn leaves", "needs_tool": False, "correct_tools": []},
    {"task": "Summarize the main ideas of stoic philosophy", "needs_tool": False, "correct_tools": []},
    {"task": "What are the pros and cons of remote work", "needs_tool": False, "correct_tools": []},
    {"task": "Explain how photosynthesis works", "needs_tool": False, "correct_tools": []},
    {"task": "What is the difference between a list and a tuple in Python", "needs_tool": False, "correct_tools": []},
    {"task": "Tell me a joke about programmers", "needs_tool": False, "correct_tools": []},
    {"task": "Describe the water cycle in simple terms", "needs_tool": False, "correct_tools": []},
    {"task": "What are best practices for writing clean code", "needs_tool": False, "correct_tools": []},
    {"task": "Explain object-oriented programming principles", "needs_tool": False, "correct_tools": []},
    {"task": "What is the capital of France", "needs_tool": False, "correct_tools": []},
    {"task": "Rewrite this sentence to be more concise", "needs_tool": False, "correct_tools": []},
    {"task": "Compare and contrast TCP and UDP protocols", "needs_tool": False, "correct_tools": []},
    {"task": "Generate a creative name for a coffee shop", "needs_tool": False, "correct_tools": []},
]

assert len(TASKS) == 40


# ====================================================================
# Benchmark runner
# ====================================================================


def run_benchmark() -> float:
    selector = ToolSelector()

    correct_selections = 0
    false_positives = 0
    no_tool_tasks = 0
    reciprocal_ranks: list[float] = []

    for task in TASKS:
        task_str = task["task"]
        needs_tool = task["needs_tool"]
        correct_tools = task["correct_tools"]

        # Check should_use_tool
        predicted_use = selector.should_use_tool(task_str)

        if not needs_tool:
            no_tool_tasks += 1
            if predicted_use:
                false_positives += 1
            # For no-tool tasks, selection accuracy is based on should_use_tool
            if not predicted_use:
                correct_selections += 1
            # MRR is 0 for no-tool tasks (no correct tool to find)
            reciprocal_ranks.append(0.0 if predicted_use else 1.0)
            continue

        # Task needs a tool – check if the right one was selected
        ranking = selector.select(task_str, AVAILABLE_TOOLS)

        if ranking:
            top_tool = ranking[0].get("name", "")
            if top_tool in correct_tools:
                correct_selections += 1

            # MRR
            rr = 0.0
            for rank_idx, tool_entry in enumerate(ranking):
                if tool_entry.get("name", "") in correct_tools:
                    rr = 1.0 / (rank_idx + 1)
                    break
            reciprocal_ranks.append(rr)
        else:
            reciprocal_ranks.append(0.0)

    n = len(TASKS)
    selection_accuracy = correct_selections / n
    false_positive_rate = false_positives / max(no_tool_tasks, 1)
    mrr = sum(reciprocal_ranks) / n

    fitness = (
        0.4 * selection_accuracy
        + 0.3 * (1.0 - false_positive_rate)
        + 0.3 * mrr
    )

    print(f"selection_accuracy:   {selection_accuracy:.6f}")
    print(f"false_positive_rate:  {false_positive_rate:.6f}")
    print(f"mean_reciprocal_rank: {mrr:.6f}")
    print(f"tasks_needing_tool:   {n - no_tool_tasks}")
    print(f"tasks_no_tool:        {no_tool_tasks}")
    print(f"false_positives:      {false_positives}")
    print(f"score: {fitness:.6f}")

    return fitness


if __name__ == "__main__":
    run_benchmark()
