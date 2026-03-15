"""
Runtime preparation -- scan all harness subfolders for results and identify best variants.

Scans the project tree for harness.py files, checks for results.tsv alongside them,
and prints a summary of available primitives with their scores.

Usage:
    python prepare.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Directories to scan for harness subfolders
SCAN_DIRS = [
    PROJECT_ROOT / "harnesses",
    PROJECT_ROOT / "evals",
    PROJECT_ROOT / "training-data",
    PROJECT_ROOT / "benchmarks",
]


def find_harnesses() -> list[dict]:
    """Find all harness.py files and their associated results."""
    harnesses = []

    for scan_dir in SCAN_DIRS:
        if not scan_dir.exists():
            continue
        for root, dirs, files in os.walk(scan_dir):
            if "harness.py" in files:
                harness_path = Path(root) / "harness.py"
                results_path = Path(root) / "results.tsv"
                rel_path = harness_path.relative_to(PROJECT_ROOT)

                entry = {
                    "path": str(rel_path),
                    "directory": str(Path(root).relative_to(PROJECT_ROOT)),
                    "has_results": results_path.exists(),
                    "best_score": None,
                    "variant_count": 0,
                    "best_variant": None,
                }

                if results_path.exists():
                    entry.update(_parse_results(results_path))

                harnesses.append(entry)

    # Sort by directory name for consistent output
    harnesses.sort(key=lambda h: h["directory"])
    return harnesses


def _parse_results(results_path: Path) -> dict:
    """Parse a results.tsv file to find the best variant and score."""
    best_score = None
    best_variant = None
    variant_count = 0

    try:
        with open(results_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue

            variant_count += 1
            try:
                score = float(parts[1])
                if best_score is None or score > best_score:
                    best_score = score
                    best_variant = parts[0]
            except (ValueError, IndexError):
                continue

    except (OSError, IOError):
        pass

    return {
        "best_score": best_score,
        "variant_count": variant_count,
        "best_variant": best_variant,
    }


def print_summary(harnesses: list[dict]) -> None:
    """Print a formatted summary of all discovered primitives."""
    print("=" * 70)
    print("  AUTORESEARCH -- Primitive Discovery Report")
    print("=" * 70)
    print()

    # Group by top-level directory
    groups: dict[str, list[dict]] = {}
    for h in harnesses:
        top_dir = h["directory"].split("/")[0] if "/" in h["directory"] else h["directory"]
        groups.setdefault(top_dir, []).append(h)

    total_primitives = len(harnesses)
    with_results = sum(1 for h in harnesses if h["has_results"])

    for group_name, entries in sorted(groups.items()):
        print(f"[{group_name}]")
        for h in entries:
            status = ""
            if h["has_results"]:
                score_str = f"{h['best_score']:.4f}" if h["best_score"] is not None else "N/A"
                variant_str = f"({h['variant_count']} variants)"
                best_str = f"  best: {h['best_variant']}" if h["best_variant"] else ""
                status = f"  score={score_str} {variant_str}{best_str}"
            else:
                status = "  (no results yet)"
            print(f"  {h['directory']:45s}{status}")
        print()

    print("-" * 70)
    print(f"Total primitives: {total_primitives}")
    print(f"With results:     {with_results}")
    print(f"Without results:  {total_primitives - with_results}")
    print()

    if with_results > 0:
        scored = [h for h in harnesses if h["best_score"] is not None]
        if scored:
            avg_score = sum(h["best_score"] for h in scored) / len(scored)
            best = max(scored, key=lambda h: h["best_score"])
            worst = min(scored, key=lambda h: h["best_score"])
            print(f"Average best score: {avg_score:.4f}")
            print(f"Highest:            {best['directory']} ({best['best_score']:.4f})")
            print(f"Lowest:             {worst['directory']} ({worst['best_score']:.4f})")
    else:
        print("No results available yet. Run benchmarks first.")

    print()
    print("=" * 70)


def main() -> None:
    """Entry point."""
    harnesses = find_harnesses()
    print_summary(harnesses)


if __name__ == "__main__":
    main()
