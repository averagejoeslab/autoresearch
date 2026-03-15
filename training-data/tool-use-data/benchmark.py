"""
Benchmark: Tool-Use Training Data

Evaluates how well the tool data generator creates valid, diverse
training examples for tool calling.
20 tasks over 10 tool specifications.

Fitness signal: 0.4 * validity_rate + 0.3 * diversity + 0.3 * param_coverage

Usage:
    python benchmark.py
"""

from __future__ import annotations

import os
import sys

# Allow importing from contracts/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from harness import ToolDataGenerator

# ── 10 Tool Specifications ───────────────────────────────────────────
TOOL_SPECS: list[dict] = [
    {
        "name": "web_search",
        "description": "Search the web for information",
        "parameters": {
            "query": {"type": "string", "required": True},
            "max_results": {"type": "integer", "required": False},
            "language": {"type": "string", "required": False},
        },
    },
    {
        "name": "read_file",
        "description": "Read the contents of a file",
        "parameters": {
            "file_path": {"type": "string", "required": True},
            "encoding": {"type": "string", "required": False},
        },
    },
    {
        "name": "write_file",
        "description": "Write content to a file",
        "parameters": {
            "file_path": {"type": "string", "required": True},
            "content": {"type": "string", "required": True},
            "append": {"type": "boolean", "required": False},
        },
    },
    {
        "name": "run_command",
        "description": "Execute a shell command",
        "parameters": {
            "command": {"type": "string", "required": True},
            "timeout": {"type": "integer", "required": False},
            "working_dir": {"type": "string", "required": False},
        },
    },
    {
        "name": "send_email",
        "description": "Send an email message",
        "parameters": {
            "to": {"type": "string", "required": True},
            "subject": {"type": "string", "required": True},
            "body": {"type": "string", "required": True},
            "cc": {"type": "string", "required": False},
        },
    },
    {
        "name": "database_query",
        "description": "Execute a SQL query against a database",
        "parameters": {
            "query": {"type": "string", "required": True},
            "database": {"type": "string", "required": True},
            "params": {"type": "array", "required": False},
        },
    },
    {
        "name": "http_request",
        "description": "Make an HTTP request to a URL",
        "parameters": {
            "url": {"type": "string", "required": True},
            "method": {"type": "string", "required": True},
            "headers": {"type": "object", "required": False},
            "body": {"type": "string", "required": False},
        },
    },
    {
        "name": "create_chart",
        "description": "Create a data visualization chart",
        "parameters": {
            "data": {"type": "array", "required": True},
            "chart_type": {"type": "string", "required": True},
            "title": {"type": "string", "required": False},
            "x_label": {"type": "string", "required": False},
            "y_label": {"type": "string", "required": False},
        },
    },
    {
        "name": "translate_text",
        "description": "Translate text from one language to another",
        "parameters": {
            "text": {"type": "string", "required": True},
            "source_language": {"type": "string", "required": True},
            "target_language": {"type": "string", "required": True},
        },
    },
    {
        "name": "resize_image",
        "description": "Resize an image to specified dimensions",
        "parameters": {
            "image_path": {"type": "string", "required": True},
            "width": {"type": "integer", "required": True},
            "height": {"type": "integer", "required": True},
            "maintain_aspect": {"type": "boolean", "required": False},
        },
    },
]

assert len(TOOL_SPECS) == 10, f"Expected 10 tool specs, got {len(TOOL_SPECS)}"


def run_benchmark() -> float:
    """Run the tool-use data benchmark and return fitness score."""
    generator = ToolDataGenerator()

    # Generate 2 examples per tool spec = 20 tasks
    all_examples: list[dict] = []
    for spec in TOOL_SPECS:
        for _ in range(2):
            example = generator.generate_example(spec)
            all_examples.append(example)

    assert len(all_examples) == 20, f"Expected 20 examples, got {len(all_examples)}"

    # --- Component 1: Validity rate (0.4 weight) ---
    valid_count = 0
    validity_details: list[bool] = []
    for example in all_examples:
        is_valid = generator.validate_example(example)
        validity_details.append(is_valid)
        if is_valid:
            valid_count += 1

    validity_rate = valid_count / len(all_examples)

    # Also test with some known-bad examples
    bad_examples = [
        # Missing user_query
        {"tool_call": {"name": "web_search", "arguments": {"query": "test"}}, "tool_spec": TOOL_SPECS[0]},
        # Wrong tool name
        {"user_query": "search", "tool_call": {"name": "wrong_name", "arguments": {}}, "tool_spec": TOOL_SPECS[0]},
        # Missing required param
        {"user_query": "send email", "tool_call": {"name": "send_email", "arguments": {"to": "x"}}, "tool_spec": TOOL_SPECS[4]},
        # Extra param not in spec
        {"user_query": "search", "tool_call": {"name": "web_search", "arguments": {"query": "test", "fake_param": "x"}}, "tool_spec": TOOL_SPECS[0]},
        # Empty query
        {"user_query": "", "tool_call": {"name": "web_search", "arguments": {"query": "test"}}, "tool_spec": TOOL_SPECS[0]},
    ]
    bad_rejections = 0
    for bad in bad_examples:
        if not generator.validate_example(bad):
            bad_rejections += 1
    rejection_rate = bad_rejections / len(bad_examples)

    # Combined validity score
    validity_score = 0.7 * validity_rate + 0.3 * rejection_rate

    # --- Component 2: Diversity (0.3 weight) ---
    # Check diversity of generated queries
    queries = [ex.get("user_query", "") for ex in all_examples]

    # Unique query ratio
    unique_queries = len(set(queries))
    uniqueness = unique_queries / max(len(queries), 1)

    # Average query length (longer = more natural = better, up to a point)
    avg_len = sum(len(q.split()) for q in queries) / max(len(queries), 1)
    # Optimal: 5-15 words
    if avg_len < 3:
        length_quality = 0.2
    elif avg_len <= 15:
        length_quality = 0.5 + 0.5 * min(1.0, (avg_len - 3) / 12)
    else:
        length_quality = max(0.3, 1.0 - (avg_len - 15) * 0.05)

    # Do queries mention the tool's purpose (not just "Use tool_name")?
    purpose_mentions = 0
    for ex in all_examples:
        query = ex.get("user_query", "").lower()
        spec = ex.get("tool_spec", {})
        desc = spec.get("description", "").lower()
        # Check if query contains words from the description
        desc_words = set(desc.split()) - {"a", "an", "the", "to", "of", "for"}
        query_words = set(query.split())
        if len(desc_words & query_words) >= 2:
            purpose_mentions += 1
    purpose_rate = purpose_mentions / max(len(all_examples), 1)

    diversity_score = 0.3 * uniqueness + 0.3 * length_quality + 0.4 * purpose_rate

    # --- Component 3: Parameter coverage (0.3 weight) ---
    # Check how many parameters are filled in generated examples
    total_params = 0
    filled_params = 0
    required_covered = 0
    total_required = 0
    optional_covered = 0
    total_optional = 0

    for ex in all_examples:
        spec = ex.get("tool_spec", {})
        call = ex.get("tool_call", {})
        args = call.get("arguments", {})
        params = spec.get("parameters", {})

        for pname, pinfo in params.items():
            total_params += 1
            is_required = pinfo.get("required", True) if isinstance(pinfo, dict) else True
            if is_required:
                total_required += 1
                if pname in args:
                    required_covered += 1
                    filled_params += 1
            else:
                total_optional += 1
                if pname in args:
                    optional_covered += 1
                    filled_params += 1

    required_rate = required_covered / max(total_required, 1)
    optional_rate = optional_covered / max(total_optional, 1)
    overall_fill = filled_params / max(total_params, 1)

    # Check type correctness
    type_correct = 0
    type_total = 0
    type_map = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "array": list,
        "object": dict,
    }
    for ex in all_examples:
        spec = ex.get("tool_spec", {})
        call = ex.get("tool_call", {})
        args = call.get("arguments", {})
        params = spec.get("parameters", {})
        for pname, pinfo in params.items():
            if pname in args:
                type_total += 1
                expected_type = type_map.get(
                    pinfo.get("type", "string") if isinstance(pinfo, dict) else "string"
                )
                if expected_type and isinstance(args[pname], expected_type):
                    type_correct += 1

    type_accuracy = type_correct / max(type_total, 1)

    param_coverage = (
        0.4 * required_rate
        + 0.2 * optional_rate
        + 0.2 * overall_fill
        + 0.2 * type_accuracy
    )

    # --- Final fitness ---
    fitness = (
        0.4 * validity_score
        + 0.3 * diversity_score
        + 0.3 * param_coverage
    )

    # Print diagnostics
    print("=== Tool-Use Data Benchmark ===")
    print(f"Tool specs: {len(TOOL_SPECS)}")
    print(f"Examples generated: {len(all_examples)}")
    print()
    print(f"Validity:       {validity_score:.4f}")
    print(f"  generated valid: {valid_count}/{len(all_examples)}")
    print(f"  bad rejected: {bad_rejections}/{len(bad_examples)}")
    print()
    print(f"Diversity:      {diversity_score:.4f}")
    print(f"  unique queries: {unique_queries}/{len(queries)}")
    print(f"  avg query len: {avg_len:.1f} words")
    print(f"  purpose mentions: {purpose_mentions}/{len(all_examples)}")
    print()
    print(f"Param coverage: {param_coverage:.4f}")
    print(f"  required: {required_covered}/{total_required}")
    print(f"  optional: {optional_covered}/{total_optional}")
    print(f"  type accuracy: {type_correct}/{type_total}")
    print()
    print(f"score: {fitness:.6f}")
    return fitness


if __name__ == "__main__":
    run_benchmark()
