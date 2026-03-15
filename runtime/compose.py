"""
Runtime composition configuration.

Defines which harness variant to use for each primitive, the pipeline
execution order, and runtime configuration. This is the MUTABLE file
that the meta-agent edits during composition search.

Usage:
    python compose.py          # prints the current configuration
    python compose.py --check  # validates the configuration

This file is edited by the composition search meta-agent.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ┌──────────────────────────────────────────────────────────────────────
# │ VARIANT_CONFIG -- which harness variant to use per primitive
# │ The meta-agent modifies this dict to try different variant combos.
# │ Keys: primitive directory (relative to project root)
# │ Values: variant name or "baseline" for the original harness.py
# └──────────────────────────────────────────────────────────────────────
VARIANT_CONFIG: dict[str, str] = {
    # -- Harnesses --
    "harnesses/tools/selection": "baseline",
    "harnesses/tools/composition": "baseline",
    "harnesses/tools/interface-design": "baseline",
    "harnesses/context/window-packing": "baseline",
    "harnesses/context/compaction": "baseline",
    "harnesses/context/jit-loading": "baseline",
    "harnesses/memory/retrieval": "baseline",
    "harnesses/memory/consolidation": "baseline",
    "harnesses/planning/recovery": "baseline",
    "harnesses/planning/decomposition": "baseline",
    "harnesses/planning/strategy-selection": "baseline",
    "harnesses/verification/error-recovery": "baseline",
    "harnesses/verification/self-check": "baseline",
    # -- Evals --
    "evals/grading-strategies": "baseline",
    "evals/reliability-metrics": "baseline",
    "evals/prompt-evaluation": "baseline",
    # -- Training data --
    "training-data/agent-trajectories": "baseline",
    "training-data/tool-use-data": "baseline",
    # -- Benchmarks --
    "benchmarks/difficulty-calibration": "baseline",
    "benchmarks/task-generation": "baseline",
}

# ┌──────────────────────────────────────────────────────────────────────
# │ PIPELINE_ORDER -- execution order of primitives in the agent loop
# │ Each entry is a primitive directory. The agent runtime processes
# │ them in this order per turn.
# └──────────────────────────────────────────────────────────────────────
PIPELINE_ORDER: list[str] = [
    # 1. Planning
    "harnesses/planning/decomposition",
    "harnesses/planning/strategy-selection",
    # 2. Context management
    "harnesses/context/window-packing",
    "harnesses/context/compaction",
    "harnesses/context/jit-loading",
    # 3. Tool use
    "harnesses/tools/interface-design",
    "harnesses/tools/selection",
    "harnesses/tools/composition",
    # 4. Memory
    "harnesses/memory/retrieval",
    "harnesses/memory/consolidation",
    # 5. Verification
    "harnesses/verification/self-check",
    "harnesses/verification/error-recovery",
    # 6. Recovery (if needed)
    "harnesses/planning/recovery",
]

# ┌──────────────────────────────────────────────────────────────────────
# │ RUNTIME_CONFIG -- general runtime parameters
# └──────────────────────────────────────────────────────────────────────
RUNTIME_CONFIG: dict = {
    "max_turns": 20,
    "timeout_seconds": 300,
    "early_stop_on_success": True,
    "retry_on_error": True,
    "max_retries": 3,
    "log_level": "INFO",
    "parallel_primitives": False,
}


def validate_config() -> list[str]:
    """Validate the current configuration. Returns a list of issues."""
    issues = []

    # Check that all pipeline entries exist in variant config
    for prim in PIPELINE_ORDER:
        if prim not in VARIANT_CONFIG:
            issues.append(f"Pipeline entry '{prim}' not in VARIANT_CONFIG")

    # Check that all variant config directories exist
    for prim_dir in VARIANT_CONFIG:
        full_path = PROJECT_ROOT / prim_dir
        if not full_path.exists():
            issues.append(f"Directory not found: {prim_dir}")
        elif not (full_path / "harness.py").exists():
            issues.append(f"No harness.py in: {prim_dir}")

    # Check pipeline has no duplicates
    seen = set()
    for prim in PIPELINE_ORDER:
        if prim in seen:
            issues.append(f"Duplicate in pipeline: {prim}")
        seen.add(prim)

    # Check runtime config values
    if RUNTIME_CONFIG.get("max_turns", 0) < 1:
        issues.append("max_turns must be >= 1")
    if RUNTIME_CONFIG.get("timeout_seconds", 0) < 1:
        issues.append("timeout_seconds must be >= 1")
    if RUNTIME_CONFIG.get("max_retries", 0) < 0:
        issues.append("max_retries must be >= 0")

    return issues


def print_config() -> None:
    """Print the current configuration in a readable format."""
    print("=" * 70)
    print("  AUTORESEARCH -- Composition Configuration")
    print("=" * 70)
    print()

    print("[Variant Config]")
    for prim, variant in sorted(VARIANT_CONFIG.items()):
        marker = "*" if variant != "baseline" else " "
        print(f"  {marker} {prim:45s}  -> {variant}")
    print()

    print("[Pipeline Order]")
    for i, prim in enumerate(PIPELINE_ORDER, 1):
        variant = VARIANT_CONFIG.get(prim, "???")
        print(f"  {i:2d}. {prim:45s}  ({variant})")
    print()

    print("[Runtime Config]")
    for key, val in sorted(RUNTIME_CONFIG.items()):
        print(f"  {key:30s}  = {val}")
    print()

    # Validate
    issues = validate_config()
    if issues:
        print("[Validation Issues]")
        for issue in issues:
            print(f"  WARNING: {issue}")
    else:
        print("[Validation: OK]")

    non_baseline = sum(1 for v in VARIANT_CONFIG.values() if v != "baseline")
    print()
    print(f"Total primitives:    {len(VARIANT_CONFIG)}")
    print(f"Non-baseline:        {non_baseline}")
    print(f"Pipeline length:     {len(PIPELINE_ORDER)}")
    print()
    print("=" * 70)


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Composition configuration")
    parser.add_argument("--check", action="store_true", help="Validate config and exit")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    if args.json:
        config = {
            "variant_config": VARIANT_CONFIG,
            "pipeline_order": PIPELINE_ORDER,
            "runtime_config": RUNTIME_CONFIG,
        }
        print(json.dumps(config, indent=2))
    elif args.check:
        issues = validate_config()
        if issues:
            for issue in issues:
                print(f"ERROR: {issue}", file=sys.stderr)
            sys.exit(1)
        else:
            print("Configuration is valid.")
    else:
        print_config()


if __name__ == "__main__":
    main()
