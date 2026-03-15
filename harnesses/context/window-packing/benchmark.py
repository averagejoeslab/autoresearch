#!/usr/bin/env python3
"""
Benchmark: Window Packing
=========================

20 tasks across three categories:
  - Balanced allocation   (8 tasks): priority-weighted content coverage
  - Overflow handling      (6 tasks): graceful degradation when content >> budget
  - Adaptive allocation    (6 tasks): task-appropriate section sizing

Fitness signal:
  score = 0.4 * content_coverage
        + 0.3 * priority_alignment
        + 0.3 * min_section_coverage
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from contracts import AgentMessage  # noqa: E402
from harness import WindowPacker  # noqa: E402

# ======================================================================
# Helpers
# ======================================================================

_WC = lambda t: len(t.split())  # noqa: E731

SECTION_PRIORITIES = {
    "system": 1.0,
    "current": 0.9,
    "history": 0.7,
    "tools": 0.6,
}


def _make_entry(content: str) -> dict:
    return {"content": content}


def _section_words(entries: list[dict]) -> int:
    return sum(_WC(e.get("content", "")) for e in entries)


def _gen_section(prefix: str, count: int, words_each: int) -> list[dict]:
    """Generate *count* entries each approximately *words_each* words."""
    word_pool = (
        "alpha bravo charlie delta echo foxtrot golf hotel india juliet "
        "kilo lima mike november oscar papa quebec romeo sierra tango "
        "uniform victor whiskey xray yankee zulu"
    ).split()
    entries = []
    for i in range(count):
        words = []
        words.append(f"{prefix}_{i}:")
        while len(words) < words_each:
            words.append(word_pool[len(words) % len(word_pool)])
        entries.append(_make_entry(" ".join(words)))
    return entries


# ======================================================================
# Task generators
# ======================================================================

def _gen_balanced_tasks() -> list[dict]:
    """8 tasks with varying section sizes; budget = 60% of total."""
    tasks = []
    configs = [
        # (system_count, hist_count, tool_count, current_count, words_each)
        (1, 20, 5, 3, 15),
        (1, 10, 10, 5, 20),
        (2, 30, 3, 2, 12),
        (1, 15, 8, 4, 18),
        (1, 25, 6, 2, 10),
        (1, 12, 12, 6, 14),
        (2, 20, 4, 3, 16),
        (1, 18, 7, 5, 13),
    ]
    for sc, hc, tc, cc, we in configs:
        sections = {
            "system": _gen_section("sys", sc, we),
            "history": _gen_section("hist", hc, we),
            "tools": _gen_section("tool", tc, we),
            "current": _gen_section("curr", cc, we),
        }
        total = sum(_section_words(v) for v in sections.values())
        budget = int(total * 0.60)
        tasks.append({"type": "balanced", "sections": sections, "budget": budget})
    return tasks


def _gen_overflow_tasks() -> list[dict]:
    """6 tasks where total content exceeds budget by ~3x."""
    tasks = []
    configs = [
        (1, 40, 10, 5, 20),
        (2, 50, 15, 8, 15),
        (1, 30, 20, 10, 25),
        (1, 60, 5, 3, 18),
        (2, 35, 12, 6, 22),
        (1, 45, 8, 4, 20),
    ]
    for sc, hc, tc, cc, we in configs:
        sections = {
            "system": _gen_section("sys", sc, we),
            "history": _gen_section("hist", hc, we),
            "tools": _gen_section("tool", tc, we),
            "current": _gen_section("curr", cc, we),
        }
        total = sum(_section_words(v) for v in sections.values())
        budget = int(total / 3.0)  # budget is ~1/3 of total
        tasks.append({"type": "overflow", "sections": sections, "budget": budget})
    return tasks


def _gen_adaptive_tasks() -> list[dict]:
    """6 tasks where one section is clearly more important.

    'priority_section' marks which section should get the most budget.
    """
    tasks = []

    # History-heavy tasks (debugging a long conversation)
    for _ in range(2):
        sections = {
            "system": _gen_section("sys", 1, 10),
            "history": _gen_section("hist", 40, 20),
            "tools": _gen_section("tool", 3, 10),
            "current": _gen_section("curr", 2, 10),
        }
        total = sum(_section_words(v) for v in sections.values())
        budget = int(total * 0.50)
        tasks.append({
            "type": "adaptive",
            "sections": sections,
            "budget": budget,
            "priority_section": "history",
        })

    # Tool-heavy tasks (lots of search results to integrate)
    for _ in range(2):
        sections = {
            "system": _gen_section("sys", 1, 10),
            "history": _gen_section("hist", 5, 15),
            "tools": _gen_section("tool", 30, 20),
            "current": _gen_section("curr", 2, 12),
        }
        total = sum(_section_words(v) for v in sections.values())
        budget = int(total * 0.50)
        tasks.append({
            "type": "adaptive",
            "sections": sections,
            "budget": budget,
            "priority_section": "tools",
        })

    # Current-heavy tasks (complex current instructions)
    for _ in range(2):
        sections = {
            "system": _gen_section("sys", 1, 10),
            "history": _gen_section("hist", 5, 10),
            "tools": _gen_section("tool", 3, 10),
            "current": _gen_section("curr", 15, 25),
        }
        total = sum(_section_words(v) for v in sections.values())
        budget = int(total * 0.50)
        tasks.append({
            "type": "adaptive",
            "sections": sections,
            "budget": budget,
            "priority_section": "current",
        })

    return tasks


# ======================================================================
# Scoring functions
# ======================================================================

def _coverage(original: list[dict], packed: list[dict]) -> float:
    """Fraction of original words that appear in packed output."""
    orig_words = _section_words(original)
    if orig_words == 0:
        return 1.0
    packed_words = _section_words(packed)
    return min(packed_words / orig_words, 1.0)


def _score_balanced(task: dict, packed: dict[str, list[dict]]) -> tuple[float, float, float]:
    """Return (content_coverage, priority_alignment, min_section_coverage)."""
    sections = task["sections"]

    # Content coverage: priority-weighted average
    weighted_cov = 0.0
    total_priority = 0.0
    coverages = {}
    for name in sections:
        cov = _coverage(sections[name], packed.get(name, []))
        coverages[name] = cov
        pri = SECTION_PRIORITIES.get(name, 0.5)
        weighted_cov += cov * pri
        total_priority += pri
    content_coverage = weighted_cov / total_priority if total_priority else 0.0

    # Priority alignment: correlation between priority and coverage
    # Higher-priority sections should get higher coverage.
    section_names = sorted(sections.keys())
    priorities = [SECTION_PRIORITIES.get(n, 0.5) for n in section_names]
    covs = [coverages[n] for n in section_names]
    # Simple: check if rank ordering of coverage matches priority ordering
    pri_rank = _rank_order(priorities)
    cov_rank = _rank_order(covs)
    # Kendall-tau-like: fraction of concordant pairs
    n = len(section_names)
    concordant = 0
    total_pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            total_pairs += 1
            if (pri_rank[i] - pri_rank[j]) * (cov_rank[i] - cov_rank[j]) >= 0:
                concordant += 1
    priority_alignment = concordant / total_pairs if total_pairs else 1.0

    # Min section coverage: the worst-covered section
    min_cov = min(coverages.values()) if coverages else 0.0

    return content_coverage, priority_alignment, min_cov


def _rank_order(values: list[float]) -> list[int]:
    """Return rank order (0=lowest)."""
    indexed = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0] * len(values)
    for rank, (idx, _) in enumerate(indexed):
        ranks[idx] = rank
    return ranks


def _score_overflow(task: dict, packed: dict[str, list[dict]]) -> tuple[float, float, float]:
    """Same metrics but emphasis on min-section being > 0."""
    return _score_balanced(task, packed)


def _score_adaptive(task: dict, packed: dict[str, list[dict]]) -> tuple[float, float, float]:
    """Check that the priority section gets disproportionate coverage."""
    sections = task["sections"]
    priority_section = task["priority_section"]

    coverages = {}
    for name in sections:
        coverages[name] = _coverage(sections[name], packed.get(name, []))

    # Content coverage (simple average)
    content_coverage = sum(coverages.values()) / len(coverages) if coverages else 0.0

    # Priority alignment: does the priority section get the highest coverage
    # among non-system sections?
    non_system = {k: v for k, v in coverages.items() if k != "system"}
    if non_system:
        best_section = max(non_system, key=non_system.get)
        priority_alignment = 1.0 if best_section == priority_section else 0.3
    else:
        priority_alignment = 0.0

    # Min section coverage
    min_cov = min(coverages.values()) if coverages else 0.0

    return content_coverage, priority_alignment, min_cov


# ======================================================================
# Budget verification
# ======================================================================

def _packed_total_words(packed: dict[str, list[dict]]) -> int:
    return sum(_section_words(v) for v in packed.values())


# ======================================================================
# Main benchmark runner
# ======================================================================

def run_benchmark() -> float:
    packer = WindowPacker()

    balanced_tasks = _gen_balanced_tasks()
    overflow_tasks = _gen_overflow_tasks()
    adaptive_tasks = _gen_adaptive_tasks()

    all_coverage: list[float] = []
    all_priority: list[float] = []
    all_min_cov: list[float] = []

    print("=" * 60)
    print("Window Packing Benchmark Results")
    print("=" * 60)

    for label, tasks, scorer in [
        ("Balanced", balanced_tasks, _score_balanced),
        ("Overflow", overflow_tasks, _score_overflow),
        ("Adaptive", adaptive_tasks, _score_adaptive),
    ]:
        print(f"\n  {label} ({len(tasks)} tasks):")
        for i, task in enumerate(tasks):
            packed = packer.pack(task["sections"], task["budget"])
            # Budget compliance check
            used = _packed_total_words(packed)
            assert used <= task["budget"] + 5, (
                f"{label} task {i}: budget violated {used} > {task['budget']}"
            )
            cov, pri, mc = scorer(task, packed)
            all_coverage.append(cov)
            all_priority.append(pri)
            all_min_cov.append(mc)
            extra = ""
            if task.get("priority_section"):
                extra = f"  (priority={task['priority_section']})"
            print(f"    task {i}: cov={cov:.4f} pri={pri:.4f} min={mc:.4f}{extra}")

    avg_cov = sum(all_coverage) / len(all_coverage)
    avg_pri = sum(all_priority) / len(all_priority)
    avg_min = sum(all_min_cov) / len(all_min_cov)

    final = 0.4 * avg_cov + 0.3 * avg_pri + 0.3 * avg_min

    print("\n" + "-" * 60)
    print(f"  Avg content coverage:   {avg_cov:.4f}")
    print(f"  Avg priority alignment: {avg_pri:.4f}")
    print(f"  Avg min section cov:    {avg_min:.4f}")
    print(f"  Composite: 0.4*{avg_cov:.4f} + 0.3*{avg_pri:.4f} + 0.3*{avg_min:.4f}")
    print(f"score: {final:.6f}")
    return final


if __name__ == "__main__":
    run_benchmark()
