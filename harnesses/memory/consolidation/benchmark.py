#!/usr/bin/env python3
"""
Benchmark: Memory Consolidation
================================
Evaluates MemoryConsolidator on 20 tasks testing the ability to generalise
from specific episodes to patterns (episodic -> semantic memory).

Fitness = 0.6 * generalization_accuracy + 0.4 * specificity_preservation

Run:  python benchmark.py
"""

from __future__ import annotations

import os
import sys

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from harness import MemoryConsolidator  # noqa: E402

# ====================================================================
# Inline dataset: 20 episodes across multiple categories
# ====================================================================
EPISODES: list[dict] = [
    # Coffee orders (7 episodes) – user mostly orders latte
    {"category": "coffee_order", "attributes": {"drink": "latte", "size": "medium", "milk": "oat"}, "description": "User ordered a medium oat milk latte on Monday morning"},
    {"category": "coffee_order", "attributes": {"drink": "latte", "size": "medium", "milk": "oat"}, "description": "User ordered a medium oat milk latte on Tuesday morning"},
    {"category": "coffee_order", "attributes": {"drink": "latte", "size": "large", "milk": "oat"}, "description": "User ordered a large oat milk latte on Wednesday morning"},
    {"category": "coffee_order", "attributes": {"drink": "cappuccino", "size": "medium", "milk": "whole"}, "description": "User ordered a medium whole milk cappuccino on Thursday"},
    {"category": "coffee_order", "attributes": {"drink": "latte", "size": "medium", "milk": "oat"}, "description": "User ordered a medium oat milk latte on Friday morning"},
    {"category": "coffee_order", "attributes": {"drink": "latte", "size": "medium", "milk": "oat"}, "description": "User ordered a medium oat milk latte on Saturday"},
    {"category": "coffee_order", "attributes": {"drink": "latte", "size": "medium", "milk": "almond"}, "description": "User ordered a medium almond milk latte on Sunday"},

    # Meeting preferences (5 episodes) – user prefers morning, 30 min
    {"category": "meeting_pref", "attributes": {"time_of_day": "morning", "duration": "30min", "format": "video"}, "description": "User scheduled a 30 min morning video meeting"},
    {"category": "meeting_pref", "attributes": {"time_of_day": "morning", "duration": "30min", "format": "video"}, "description": "User had a 30 min morning video call with team"},
    {"category": "meeting_pref", "attributes": {"time_of_day": "afternoon", "duration": "60min", "format": "in-person"}, "description": "User attended 60 min afternoon in-person meeting"},
    {"category": "meeting_pref", "attributes": {"time_of_day": "morning", "duration": "30min", "format": "video"}, "description": "User set up a 30 min morning video meeting"},
    {"category": "meeting_pref", "attributes": {"time_of_day": "morning", "duration": "30min", "format": "phone"}, "description": "User had a 30 min morning phone call"},

    # Lunch choices (4 episodes) – user mostly eats salad
    {"category": "lunch_choice", "attributes": {"food": "salad", "location": "desk", "time": "12:00"}, "description": "User had salad at desk at noon"},
    {"category": "lunch_choice", "attributes": {"food": "salad", "location": "cafeteria", "time": "12:30"}, "description": "User had salad at cafeteria at 12:30"},
    {"category": "lunch_choice", "attributes": {"food": "sandwich", "location": "desk", "time": "12:00"}, "description": "User had sandwich at desk at noon"},
    {"category": "lunch_choice", "attributes": {"food": "salad", "location": "desk", "time": "12:00"}, "description": "User had salad at desk at noon"},

    # Exercise (4 episodes) – user mostly runs in the evening
    {"category": "exercise", "attributes": {"activity": "running", "time_of_day": "evening", "duration": "30min"}, "description": "User went running in the evening for 30 minutes"},
    {"category": "exercise", "attributes": {"activity": "running", "time_of_day": "evening", "duration": "45min"}, "description": "User went running in the evening for 45 minutes"},
    {"category": "exercise", "attributes": {"activity": "yoga", "time_of_day": "morning", "duration": "60min"}, "description": "User did yoga in the morning for 60 minutes"},
    {"category": "exercise", "attributes": {"activity": "running", "time_of_day": "evening", "duration": "30min"}, "description": "User went running in the evening for 30 minutes"},
]

assert len(EPISODES) == 20, f"Expected 20 episodes, got {len(EPISODES)}"

# ====================================================================
# 20 benchmark tasks
# ====================================================================
# Each task tests either generalization accuracy or specificity preservation.
# generalization tasks: check if the consolidator found the right pattern
# specificity tasks: check if specific episodes are still accessible

TASKS: list[dict] = [
    # --- Generalization accuracy tasks (1-12) ---
    # Coffee patterns
    {
        "type": "generalization",
        "question": "What does the user usually order for coffee?",
        "expected_attribute": "drink",
        "expected_value": "latte",
        "category": "coffee_order",
    },
    {
        "type": "generalization",
        "question": "What size coffee does the user prefer?",
        "expected_attribute": "size",
        "expected_value": "medium",
        "category": "coffee_order",
    },
    {
        "type": "generalization",
        "question": "What type of milk does the user usually choose?",
        "expected_attribute": "milk",
        "expected_value": "oat",
        "category": "coffee_order",
    },
    # Meeting patterns
    {
        "type": "generalization",
        "question": "When does the user prefer meetings?",
        "expected_attribute": "time_of_day",
        "expected_value": "morning",
        "category": "meeting_pref",
    },
    {
        "type": "generalization",
        "question": "How long are the user meetings usually?",
        "expected_attribute": "duration",
        "expected_value": "30min",
        "category": "meeting_pref",
    },
    {
        "type": "generalization",
        "question": "What meeting format does the user prefer?",
        "expected_attribute": "format",
        "expected_value": "video",
        "category": "meeting_pref",
    },
    # Lunch patterns
    {
        "type": "generalization",
        "question": "What does the user eat for lunch?",
        "expected_attribute": "food",
        "expected_value": "salad",
        "category": "lunch_choice",
    },
    {
        "type": "generalization",
        "question": "Where does the user eat lunch?",
        "expected_attribute": "location",
        "expected_value": "desk",
        "category": "lunch_choice",
    },
    {
        "type": "generalization",
        "question": "What time does the user eat lunch?",
        "expected_attribute": "time",
        "expected_value": "12:00",
        "category": "lunch_choice",
    },
    # Exercise patterns
    {
        "type": "generalization",
        "question": "What exercise does the user usually do?",
        "expected_attribute": "activity",
        "expected_value": "running",
        "category": "exercise",
    },
    {
        "type": "generalization",
        "question": "When does the user exercise?",
        "expected_attribute": "time_of_day",
        "expected_value": "evening",
        "category": "exercise",
    },
    {
        "type": "generalization",
        "question": "How long does the user exercise?",
        "expected_attribute": "duration",
        "expected_value": "30min",
        "category": "exercise",
    },

    # --- Specificity preservation tasks (13-20) ---
    # Check that the consolidator still knows about minority / specific events
    {
        "type": "specificity",
        "question": "Did the user ever order a cappuccino?",
        "category": "coffee_order",
        "expected_specific_value": "cappuccino",
        "attribute": "drink",
    },
    {
        "type": "specificity",
        "question": "Did the user ever use almond milk?",
        "category": "coffee_order",
        "expected_specific_value": "almond",
        "attribute": "milk",
    },
    {
        "type": "specificity",
        "question": "Did the user ever have an in-person meeting?",
        "category": "meeting_pref",
        "expected_specific_value": "in-person",
        "attribute": "format",
    },
    {
        "type": "specificity",
        "question": "Did the user ever have a 60 minute meeting?",
        "category": "meeting_pref",
        "expected_specific_value": "60min",
        "attribute": "duration",
    },
    {
        "type": "specificity",
        "question": "Did the user ever eat a sandwich for lunch?",
        "category": "lunch_choice",
        "expected_specific_value": "sandwich",
        "attribute": "food",
    },
    {
        "type": "specificity",
        "question": "Did the user ever eat at the cafeteria?",
        "category": "lunch_choice",
        "expected_specific_value": "cafeteria",
        "attribute": "location",
    },
    {
        "type": "specificity",
        "question": "Did the user ever do yoga?",
        "category": "exercise",
        "expected_specific_value": "yoga",
        "attribute": "activity",
    },
    {
        "type": "specificity",
        "question": "Did the user ever exercise for 45 minutes?",
        "category": "exercise",
        "expected_specific_value": "45min",
        "attribute": "duration",
    },
]

assert len(TASKS) == 20, f"Expected 20 tasks, got {len(TASKS)}"


# ====================================================================
# Benchmark runner
# ====================================================================


def run_benchmark() -> float:
    consolidator = MemoryConsolidator()

    # Load episodes
    for ep in EPISODES:
        consolidator.add_episode(ep)

    # Run consolidation
    generalizations = consolidator.consolidate()

    # Build lookup for quick access
    gen_by_cat: dict[str, dict] = {g["category"]: g for g in generalizations}

    generalization_scores: list[float] = []
    specificity_scores: list[float] = []

    for task in TASKS:
        if task["type"] == "generalization":
            cat = task["category"]
            gen = gen_by_cat.get(cat)
            if gen is None:
                generalization_scores.append(0.0)
                continue

            predicted_val = gen.get("attributes", {}).get(task["expected_attribute"])
            expected_val = task["expected_value"]

            if predicted_val is not None and str(predicted_val).lower() == str(expected_val).lower():
                generalization_scores.append(1.0)
            else:
                generalization_scores.append(0.0)

        elif task["type"] == "specificity":
            # Check if the specific/minority event is still accessible
            cat = task["category"]
            expected_specific = task["expected_specific_value"]
            attr = task["attribute"]

            # Check 1: Can we find it in the raw episodes?
            found_in_episodes = any(
                ep.get("category") == cat
                and str(ep.get("attributes", {}).get(attr, "")).lower() == expected_specific.lower()
                for ep in consolidator.episodes
            )

            # Check 2: Does the query_general response reference it?
            answer = consolidator.query_general(task["question"])
            found_in_answer = expected_specific.lower() in answer.lower()

            # Score: 0.5 for preserving episodes, 0.5 for surfacing in queries
            score = 0.0
            if found_in_episodes:
                score += 0.5
            if found_in_answer:
                score += 0.5
            specificity_scores.append(score)

    gen_accuracy = sum(generalization_scores) / max(len(generalization_scores), 1)
    spec_preservation = sum(specificity_scores) / max(len(specificity_scores), 1)

    fitness = 0.6 * gen_accuracy + 0.4 * spec_preservation

    print(f"generalization_accuracy:    {gen_accuracy:.6f}")
    print(f"specificity_preservation:   {spec_preservation:.6f}")
    print(f"generalization_tasks:       {len(generalization_scores)}")
    print(f"specificity_tasks:          {len(specificity_scores)}")
    print(f"score: {fitness:.6f}")

    return fitness


if __name__ == "__main__":
    run_benchmark()
