#!/usr/bin/env python3
"""
Benchmark: Tool Composition
=============================
Evaluates ToolComposer on 20 multi-step tasks requiring 2-3 chained tool
calls.  Measures chain correctness (right tools, right order), parameter
mapping accuracy, and structural validity.

Fitness = 0.4 * chain_accuracy + 0.3 * param_mapping + 0.3 * validity_rate

Run:  python benchmark.py
"""

from __future__ import annotations

import os
import sys

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from harness import ToolComposer  # noqa: E402

# ====================================================================
# Available tools (shared across tasks)
# ====================================================================
TOOLS: list[dict] = [
    {"name": "web_search", "description": "Search the web for information", "parameters": {"query": "search string"}},
    {"name": "summarizer", "description": "Summarize a long text into key points", "parameters": {"text": "text to summarize"}},
    {"name": "translator", "description": "Translate text between languages", "parameters": {"text": "text to translate", "target_lang": "target language"}},
    {"name": "file_reader", "description": "Read contents of a file", "parameters": {"path": "file path"}},
    {"name": "file_writer", "description": "Write content to a file", "parameters": {"path": "file path", "content": "content to write"}},
    {"name": "calculator", "description": "Evaluate mathematical expressions", "parameters": {"expression": "math expression"}},
    {"name": "formatter", "description": "Format text as markdown, HTML, or plain text", "parameters": {"text": "text to format", "format": "output format"}},
    {"name": "data_parser", "description": "Parse structured data from CSV, JSON, or XML", "parameters": {"data": "raw data string", "format": "data format"}},
    {"name": "email_sender", "description": "Send an email message", "parameters": {"to": "recipient", "subject": "subject", "body": "body"}},
    {"name": "code_runner", "description": "Execute code and return output", "parameters": {"code": "source code", "language": "programming language"}},
]

# ====================================================================
# 20 tasks with ground-truth chains
# ====================================================================
# Each task specifies:
#   task          – natural language description
#   tools_subset  – which tools are available for this task
#   expected_chain – ground truth: list of {tool, param_sources}
#       param_sources maps param_name -> "task" | step_index (int)

TASKS: list[dict] = [
    # 2-step chains
    {
        "task": "Search the web for Python tutorials and summarize the results",
        "tools_subset": ["web_search", "summarizer"],
        "expected_chain": [
            {"tool": "web_search", "param_sources": {"query": "task"}},
            {"tool": "summarizer", "param_sources": {"text": 0}},
        ],
    },
    {
        "task": "Read the report file and translate it to Spanish",
        "tools_subset": ["file_reader", "translator"],
        "expected_chain": [
            {"tool": "file_reader", "param_sources": {"path": "task"}},
            {"tool": "translator", "param_sources": {"text": 0, "target_lang": "task"}},
        ],
    },
    {
        "task": "Search for climate data and write the results to a file",
        "tools_subset": ["web_search", "file_writer"],
        "expected_chain": [
            {"tool": "web_search", "param_sources": {"query": "task"}},
            {"tool": "file_writer", "param_sources": {"path": "task", "content": 0}},
        ],
    },
    {
        "task": "Read the CSV file and parse the structured data",
        "tools_subset": ["file_reader", "data_parser"],
        "expected_chain": [
            {"tool": "file_reader", "param_sources": {"path": "task"}},
            {"tool": "data_parser", "param_sources": {"data": 0, "format": "task"}},
        ],
    },
    {
        "task": "Read the document and format it as markdown",
        "tools_subset": ["file_reader", "formatter"],
        "expected_chain": [
            {"tool": "file_reader", "param_sources": {"path": "task"}},
            {"tool": "formatter", "param_sources": {"text": 0, "format": "task"}},
        ],
    },
    {
        "task": "Search for financial data and calculate the total",
        "tools_subset": ["web_search", "calculator"],
        "expected_chain": [
            {"tool": "web_search", "param_sources": {"query": "task"}},
            {"tool": "calculator", "param_sources": {"expression": 0}},
        ],
    },
    {
        "task": "Run the analysis code and email the output",
        "tools_subset": ["code_runner", "email_sender"],
        "expected_chain": [
            {"tool": "code_runner", "param_sources": {"code": "task", "language": "task"}},
            {"tool": "email_sender", "param_sources": {"to": "task", "subject": "task", "body": 0}},
        ],
    },
    {
        "task": "Read the data file and run code to process it",
        "tools_subset": ["file_reader", "code_runner"],
        "expected_chain": [
            {"tool": "file_reader", "param_sources": {"path": "task"}},
            {"tool": "code_runner", "param_sources": {"code": "task", "language": "task"}},
        ],
    },
    {
        "task": "Translate the text to French and save it to a file",
        "tools_subset": ["translator", "file_writer"],
        "expected_chain": [
            {"tool": "translator", "param_sources": {"text": "task", "target_lang": "task"}},
            {"tool": "file_writer", "param_sources": {"path": "task", "content": 0}},
        ],
    },
    {
        "task": "Parse the JSON data and summarize the contents",
        "tools_subset": ["data_parser", "summarizer"],
        "expected_chain": [
            {"tool": "data_parser", "param_sources": {"data": "task", "format": "task"}},
            {"tool": "summarizer", "param_sources": {"text": 0}},
        ],
    },

    # 3-step chains
    {
        "task": "Search the web for AI research papers, summarize them, and save to a file",
        "tools_subset": ["web_search", "summarizer", "file_writer"],
        "expected_chain": [
            {"tool": "web_search", "param_sources": {"query": "task"}},
            {"tool": "summarizer", "param_sources": {"text": 0}},
            {"tool": "file_writer", "param_sources": {"path": "task", "content": 1}},
        ],
    },
    {
        "task": "Read the report, translate it to German, and send it by email",
        "tools_subset": ["file_reader", "translator", "email_sender"],
        "expected_chain": [
            {"tool": "file_reader", "param_sources": {"path": "task"}},
            {"tool": "translator", "param_sources": {"text": 0, "target_lang": "task"}},
            {"tool": "email_sender", "param_sources": {"to": "task", "subject": "task", "body": 1}},
        ],
    },
    {
        "task": "Search for statistics, calculate the average, and format as a table",
        "tools_subset": ["web_search", "calculator", "formatter"],
        "expected_chain": [
            {"tool": "web_search", "param_sources": {"query": "task"}},
            {"tool": "calculator", "param_sources": {"expression": 0}},
            {"tool": "formatter", "param_sources": {"text": 1, "format": "task"}},
        ],
    },
    {
        "task": "Read the CSV file, parse the data, and summarize it",
        "tools_subset": ["file_reader", "data_parser", "summarizer"],
        "expected_chain": [
            {"tool": "file_reader", "param_sources": {"path": "task"}},
            {"tool": "data_parser", "param_sources": {"data": 0, "format": "task"}},
            {"tool": "summarizer", "param_sources": {"text": 1}},
        ],
    },
    {
        "task": "Search for news articles, translate them to Japanese, and save to disk",
        "tools_subset": ["web_search", "translator", "file_writer"],
        "expected_chain": [
            {"tool": "web_search", "param_sources": {"query": "task"}},
            {"tool": "translator", "param_sources": {"text": 0, "target_lang": "task"}},
            {"tool": "file_writer", "param_sources": {"path": "task", "content": 1}},
        ],
    },
    {
        "task": "Read the code file, run it, and email the output",
        "tools_subset": ["file_reader", "code_runner", "email_sender"],
        "expected_chain": [
            {"tool": "file_reader", "param_sources": {"path": "task"}},
            {"tool": "code_runner", "param_sources": {"code": 0, "language": "task"}},
            {"tool": "email_sender", "param_sources": {"to": "task", "subject": "task", "body": 1}},
        ],
    },
    {
        "task": "Parse the XML data, format as HTML, and write to file",
        "tools_subset": ["data_parser", "formatter", "file_writer"],
        "expected_chain": [
            {"tool": "data_parser", "param_sources": {"data": "task", "format": "task"}},
            {"tool": "formatter", "param_sources": {"text": 0, "format": "task"}},
            {"tool": "file_writer", "param_sources": {"path": "task", "content": 1}},
        ],
    },
    {
        "task": "Search for product reviews, summarize them, and email to the team",
        "tools_subset": ["web_search", "summarizer", "email_sender"],
        "expected_chain": [
            {"tool": "web_search", "param_sources": {"query": "task"}},
            {"tool": "summarizer", "param_sources": {"text": 0}},
            {"tool": "email_sender", "param_sources": {"to": "task", "subject": "task", "body": 1}},
        ],
    },
    {
        "task": "Read the log file, run analysis code on it, and save results",
        "tools_subset": ["file_reader", "code_runner", "file_writer"],
        "expected_chain": [
            {"tool": "file_reader", "param_sources": {"path": "task"}},
            {"tool": "code_runner", "param_sources": {"code": 0, "language": "task"}},
            {"tool": "file_writer", "param_sources": {"path": "task", "content": 1}},
        ],
    },
    {
        "task": "Search for recipes, format as a list, and translate to Italian",
        "tools_subset": ["web_search", "formatter", "translator"],
        "expected_chain": [
            {"tool": "web_search", "param_sources": {"query": "task"}},
            {"tool": "formatter", "param_sources": {"text": 0, "format": "task"}},
            {"tool": "translator", "param_sources": {"text": 1, "target_lang": "task"}},
        ],
    },
]

assert len(TASKS) == 20


# ====================================================================
# Scoring helpers
# ====================================================================

def _tool_order_score(predicted_chain: list[dict], expected_chain: list[dict]) -> float:
    """Score how well the predicted tool order matches the expected order.

    Returns fraction of expected tools that appear in the correct position.
    """
    if not expected_chain:
        return 1.0 if not predicted_chain else 0.0
    if not predicted_chain:
        return 0.0

    expected_tools = [step["tool"] for step in expected_chain]
    predicted_tools = [step.get("tool", "") for step in predicted_chain]

    # Check exact positional match
    matches = 0
    for i, exp_tool in enumerate(expected_tools):
        if i < len(predicted_tools) and predicted_tools[i] == exp_tool:
            matches += 1

    return matches / len(expected_tools)


def _param_mapping_score(predicted_chain: list[dict], expected_chain: list[dict]) -> float:
    """Score whether parameters are correctly sourced (from task vs from previous step)."""
    if not expected_chain or not predicted_chain:
        return 0.0

    total_params = 0
    correct_params = 0

    for step_idx, exp_step in enumerate(expected_chain):
        if step_idx >= len(predicted_chain):
            total_params += len(exp_step.get("param_sources", {}))
            continue

        pred_step = predicted_chain[step_idx]
        pred_inputs = pred_step.get("inputs", {})

        for param_name, source in exp_step.get("param_sources", {}).items():
            total_params += 1
            pred_val = pred_inputs.get(param_name)
            if pred_val is None:
                continue

            if source == "task":
                # Should be a literal / task-derived value (not a step reference)
                if isinstance(pred_val, str):
                    correct_params += 1
            elif isinstance(source, int):
                # Should be a reference to a previous step
                if isinstance(pred_val, dict) and pred_val.get("from_step") == source:
                    correct_params += 1

    return correct_params / max(total_params, 1)


# ====================================================================
# Benchmark runner
# ====================================================================


def run_benchmark() -> float:
    composer = ToolComposer()

    chain_scores: list[float] = []
    param_scores: list[float] = []
    validity_scores: list[float] = []

    for task in TASKS:
        # Build the tool subset
        subset_names = set(task["tools_subset"])
        available = [t for t in TOOLS if t["name"] in subset_names]

        predicted_chain = composer.plan_chain(task["task"], available)
        expected_chain = task["expected_chain"]

        # 1. Chain accuracy (tool order)
        chain_scores.append(_tool_order_score(predicted_chain, expected_chain))

        # 2. Parameter mapping
        param_scores.append(_param_mapping_score(predicted_chain, expected_chain))

        # 3. Structural validity
        validity_scores.append(1.0 if composer.validate_chain(predicted_chain) else 0.0)

    n = len(TASKS)
    chain_accuracy = sum(chain_scores) / n
    param_mapping = sum(param_scores) / n
    validity_rate = sum(validity_scores) / n

    fitness = 0.4 * chain_accuracy + 0.3 * param_mapping + 0.3 * validity_rate

    print(f"chain_accuracy:   {chain_accuracy:.6f}")
    print(f"param_mapping:    {param_mapping:.6f}")
    print(f"validity_rate:    {validity_rate:.6f}")
    print(f"score: {fitness:.6f}")

    return fitness


if __name__ == "__main__":
    run_benchmark()
