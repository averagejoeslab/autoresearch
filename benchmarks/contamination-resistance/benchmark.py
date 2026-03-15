#!/usr/bin/env python3
"""
Benchmark: Contamination Resistance
=====================================
20 tasks, 3 variants each. Tests that variants preserve semantics
while changing surface form.

Fitness = 0.4 * semantic_preservation + 0.3 * surface_diversity + 0.3 * anti_memorization

Run:  python benchmark.py
"""

from __future__ import annotations

import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from harness import ContaminationGuard  # noqa: E402

# ── Inline tasks ────────────────────────────────────────────────────

TASKS: list[dict] = [
    # Arithmetic tasks (numbers should change)
    {"prompt": "What is 15 + 27?",       "expected_answer": "42",   "tags": ["arithmetic", "addition"]},
    {"prompt": "What is 100 - 37?",      "expected_answer": "63",   "tags": ["arithmetic", "subtraction"]},
    {"prompt": "What is 8 * 9?",         "expected_answer": "72",   "tags": ["arithmetic", "multiplication"]},
    {"prompt": "What is 144 / 12?",      "expected_answer": "12",   "tags": ["arithmetic", "division"]},
    {"prompt": "What is 25 % of 200?",   "expected_answer": "50",   "tags": ["arithmetic", "percentage"]},
    # Word problems with numbers
    {"prompt": "A store has 50 apples. If 23 are sold, how many remain?", "expected_answer": "27", "tags": ["word-problem", "subtraction"]},
    {"prompt": "A train travels 120 miles in 2 hours. What is its speed in mph?", "expected_answer": "60", "tags": ["word-problem", "division"]},
    {"prompt": "If 3 workers can build 6 chairs in 4 days, how many chairs can 3 workers build in 8 days?", "expected_answer": "12", "tags": ["word-problem", "proportion"]},
    {"prompt": "A rectangle has length 15 and width 8. What is its area?", "expected_answer": "120", "tags": ["geometry", "area"]},
    {"prompt": "A circle has radius 7. What is its diameter?", "expected_answer": "14", "tags": ["geometry", "diameter"]},
    # Sequence/pattern tasks
    {"prompt": "What is the next number: 2, 4, 8, 16, ?", "expected_answer": "32", "tags": ["sequence", "geometric"]},
    {"prompt": "What is the next number: 1, 4, 9, 16, 25, ?", "expected_answer": "36", "tags": ["sequence", "squares"]},
    {"prompt": "What is the 10th Fibonacci number?", "expected_answer": "55", "tags": ["sequence", "fibonacci"]},
    # Conversion tasks
    {"prompt": "Convert 32 degrees Fahrenheit to Celsius", "expected_answer": "0", "tags": ["conversion", "temperature"]},
    {"prompt": "How many inches are in 5 feet?", "expected_answer": "60", "tags": ["conversion", "length"]},
    # Logic tasks with numbers
    {"prompt": "If x = 7 and y = 3, what is x + y?", "expected_answer": "10", "tags": ["algebra", "evaluation"]},
    {"prompt": "Solve for x: 2x + 4 = 12", "expected_answer": "4", "tags": ["algebra", "equation"]},
    {"prompt": "What is the average of 10, 20, and 30?", "expected_answer": "20", "tags": ["statistics", "mean"]},
    {"prompt": "How many prime numbers are between 1 and 20?", "expected_answer": "8", "tags": ["number-theory", "primes"]},
    {"prompt": "What is 2 to the power of 8?", "expected_answer": "256", "tags": ["arithmetic", "exponentiation"]},
]

VARIANT_SEEDS = [42, 137, 999]


def _extract_numbers(text: str) -> list[str]:
    """Extract all numbers from text."""
    return re.findall(r"\b\d+\b", text)


def _jaccard_similarity(a: set, b: set) -> float:
    """Jaccard similarity between two sets."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _word_set(text: str) -> set[str]:
    """Get lowercase word set from text."""
    return {w.lower() for w in re.findall(r"[a-zA-Z]+", text)}


def run_benchmark() -> float:
    guard = ContaminationGuard()

    semantic_scores: list[float] = []
    diversity_scores: list[float] = []
    anti_memo_scores: list[float] = []

    for task in TASKS:
        variants: list[dict] = []
        for seed in VARIANT_SEEDS:
            try:
                variant = guard.transform(task, seed)
            except Exception:
                variant = dict(task)  # fallback: unchanged
            variants.append(variant)

        # ── Semantic preservation ────────────────────────────────────
        sem_score = 0.0
        for variant in variants:
            # 1. Tags preserved (0.4 weight)
            if set(task.get("tags", [])) == set(variant.get("tags", [])):
                sem_score += 0.4
            else:
                tag_overlap = len(set(task.get("tags", [])) & set(variant.get("tags", [])))
                total_tags = max(len(task.get("tags", [])), 1)
                sem_score += 0.4 * (tag_overlap / total_tags)

            # 2. Structure preserved - same non-numeric words (0.3 weight)
            orig_structure = re.sub(r"\b\d+\b", "NUM", task["prompt"])
            var_structure = re.sub(r"\b\d+\b", "NUM", variant.get("prompt", ""))
            struct_sim = _jaccard_similarity(_word_set(orig_structure), _word_set(var_structure))
            sem_score += 0.3 * struct_sim

            # 3. verify_equivalence returns True (0.3 weight)
            try:
                equiv = guard.verify_equivalence(task, variant)
                sem_score += 0.3 if equiv else 0.0
            except Exception:
                pass

        semantic_scores.append(sem_score / len(VARIANT_SEEDS))

        # ── Surface diversity ────────────────────────────────────────
        div_score = 0.0
        all_prompts = [task["prompt"]] + [v.get("prompt", "") for v in variants]

        # Check that variants differ from original
        for variant in variants:
            orig_prompt = task["prompt"]
            var_prompt = variant.get("prompt", "")

            if orig_prompt != var_prompt:
                div_score += 0.5  # different from original

                # Check that numbers actually changed
                orig_nums = set(_extract_numbers(orig_prompt))
                var_nums = set(_extract_numbers(var_prompt))
                if orig_nums and orig_nums != var_nums:
                    div_score += 0.5  # numbers changed
                elif not orig_nums:
                    div_score += 0.25  # no numbers to change, partial credit

        # Check that variants differ from each other
        pairwise_different = 0
        pairwise_total = 0
        for i in range(len(variants)):
            for j in range(i + 1, len(variants)):
                pairwise_total += 1
                if variants[i].get("prompt", "") != variants[j].get("prompt", ""):
                    pairwise_different += 1

        if pairwise_total > 0:
            pairwise_bonus = pairwise_different / pairwise_total
        else:
            pairwise_bonus = 0.0

        total_div = (div_score / len(VARIANT_SEEDS) + pairwise_bonus) / 2.0
        diversity_scores.append(min(1.0, total_div))

        # ── Anti-memorization ────────────────────────────────────────
        # Simulate a memorization agent: it has the original prompt->answer
        # memorized. Can it solve the variant by exact lookup?
        memo_score = 0.0
        memorized = {task["prompt"].strip().lower(): task["expected_answer"].strip().lower()}

        for variant in variants:
            var_prompt = variant.get("prompt", "").strip().lower()
            var_answer = variant.get("expected_answer", "").strip().lower()
            orig_answer = task["expected_answer"].strip().lower()

            # Check if memorized lookup works
            lookup_result = memorized.get(var_prompt, None)

            if lookup_result is None:
                # Memorization fails (good!) - prompt changed enough
                memo_score += 0.5
            else:
                # The exact prompt is memorized, no protection
                memo_score += 0.0

            # Also check: is the answer different from original?
            if var_answer != orig_answer:
                memo_score += 0.5  # answer changed too
            else:
                memo_score += 0.0  # same answer, partial memorization risk

        anti_memo_scores.append(memo_score / len(VARIANT_SEEDS))

    avg_semantic = sum(semantic_scores) / len(semantic_scores)
    avg_diversity = sum(diversity_scores) / len(diversity_scores)
    avg_anti_memo = sum(anti_memo_scores) / len(anti_memo_scores)

    fitness = (
        0.4 * avg_semantic
        + 0.3 * avg_diversity
        + 0.3 * avg_anti_memo
    )

    print(f"semantic_preservation: {avg_semantic:.4f}")
    print(f"surface_diversity:     {avg_diversity:.4f}")
    print(f"anti_memorization:     {avg_anti_memo:.4f}")
    print(f"score: {fitness:.6f}")
    return fitness


if __name__ == "__main__":
    run_benchmark()
