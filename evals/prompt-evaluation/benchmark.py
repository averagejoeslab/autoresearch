"""
Benchmark: Prompt Evaluation

Evaluates how well the prompt evaluator identifies better prompts
from labeled pairs across different task types.
25 tasks with human-labeled prompt comparisons.

Fitness signal: 0.5 * comparison_accuracy + 0.3 * quality_correlation + 0.2 * explanation_quality

Usage:
    python benchmark.py
"""

from __future__ import annotations

import math
import os
import sys

# Allow importing from contracts/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from harness import PromptEvaluator

# ── 25 prompt comparison tasks ───────────────────────────────────────
# (prompt_a, prompt_b, task_type, human_winner, human_quality_a, human_quality_b)
# human_winner: "A" or "B" (which prompt is better according to humans)
# human_quality_{a,b}: 0.0-1.0 human quality rating
TASKS: list[tuple[str, str, str, str, float, float]] = [
    # --- Coding tasks ---
    (
        "Write a function",
        "Write a Python function called `merge_sorted` that takes two sorted lists of integers as input and returns a single sorted list. Handle edge cases like empty lists. Include type hints and a docstring.",
        "coding", "B", 0.15, 0.95,
    ),
    (
        "Implement binary search in Python. The function should take a sorted list and a target value, returning the index if found or -1 if not found.",
        "Do binary search",
        "coding", "A", 0.85, 0.10,
    ),
    (
        "Make a class",
        "Create a Python class `Stack` with methods push(item), pop(), peek(), and is_empty(). Use a list internally. Raise an IndexError on pop/peek when empty.",
        "coding", "B", 0.10, 0.90,
    ),
    (
        "Write code to sort a list of dictionaries by a specific key. The function should accept the list and key name as parameters and return the sorted list.",
        "Sort some dicts or something, you know what I mean",
        "coding", "A", 0.80, 0.15,
    ),
    (
        "Implement a function that validates email addresses using regex. It should return True for valid emails and False otherwise. Test with: user@example.com, invalid@, @domain.com",
        "Implement a function that checks if a string is a valid email address.",
        "coding", "A", 0.90, 0.60,
    ),

    # --- Writing tasks ---
    (
        "Write something about climate change",
        "Write a 500-word persuasive essay arguing for increased investment in renewable energy. Target audience: policymakers. Include at least three specific data points and address the counterargument of economic cost.",
        "writing", "B", 0.20, 0.95,
    ),
    (
        "Compose a professional email to a client explaining a project delay of two weeks. Maintain a positive tone, provide a brief reason, and propose a revised timeline. Keep it under 200 words.",
        "Write an email about being late",
        "writing", "A", 0.90, 0.15,
    ),
    (
        "Write a short story",
        "Write a 1000-word short story in the style of magical realism set in a small coastal town. The protagonist should discover an ordinary object with extraordinary properties. Focus on sensory details.",
        "writing", "B", 0.10, 0.85,
    ),
    (
        "Write a product description for a wireless Bluetooth speaker. Highlight its waterproof rating (IPX7), 20-hour battery life, and 360-degree sound. Keep the tone enthusiastic but not hyperbolic. Aim for 100-150 words.",
        "Describe a speaker",
        "writing", "A", 0.92, 0.12,
    ),
    (
        "Write a blog post about healthy eating habits for busy professionals. Include 5 actionable tips with brief explanations.",
        "Write about food and health stuff, maybe some tips or whatever",
        "writing", "A", 0.80, 0.20,
    ),

    # --- Analysis tasks ---
    (
        "Analyze data",
        "Analyze the quarterly revenue data for Q1-Q4 2024: [$2.1M, $2.3M, $1.9M, $2.8M]. Identify the overall trend, explain the Q3 dip, and project Q1 2025 revenue with your reasoning.",
        "analysis", "B", 0.10, 0.90,
    ),
    (
        "Compare the environmental impact of electric vehicles vs traditional combustion engines. Consider manufacturing, operation, and end-of-life disposal. Present findings in a structured format with evidence.",
        "Compare electric and gas cars",
        "analysis", "A", 0.88, 0.25,
    ),
    (
        "Evaluate the pros and cons of remote work based on recent studies. Consider productivity, employee wellbeing, company culture, and cost factors. Provide a balanced conclusion.",
        "What do you think about remote work?",
        "analysis", "A", 0.85, 0.30,
    ),
    (
        "Look at this data and tell me what you see",
        "Analyze the correlation between the following paired data points: [(1,2), (3,5), (5,11), (7,14), (9,20)]. Calculate the correlation coefficient, identify whether the relationship is linear or exponential, and explain your methodology.",
        "analysis", "B", 0.10, 0.92,
    ),
    (
        "Evaluate the effectiveness of three marketing strategies: social media ads ($5K spend, 200 conversions), email campaigns ($2K spend, 150 conversions), and influencer partnerships ($8K spend, 300 conversions). Calculate ROI and recommend the best allocation of a $15K budget.",
        "Which marketing is best?",
        "analysis", "A", 0.93, 0.15,
    ),

    # --- Math tasks ---
    (
        "Solve for x",
        "Solve the quadratic equation 2x^2 + 5x - 3 = 0. Show your work step by step using the quadratic formula. Express the answer as exact fractions.",
        "math", "B", 0.05, 0.90,
    ),
    (
        "Prove that the sum of the first n positive integers equals n(n+1)/2 using mathematical induction. Clearly state the base case, inductive hypothesis, and inductive step.",
        "Prove the sum formula",
        "math", "A", 0.92, 0.20,
    ),
    (
        "Calculate the derivative of f(x) = x^3 * ln(x). Show each step, identify which differentiation rules you use, and simplify the final answer.",
        "Find the derivative of x^3 * ln(x)",
        "math", "A", 0.88, 0.55,
    ),
    (
        "Do some probability",
        "A bag contains 5 red, 3 blue, and 2 green marbles. Calculate the probability of drawing 2 red marbles in succession without replacement. Show your work using combinatorics.",
        "math", "B", 0.05, 0.90,
    ),
    (
        "Find the volume of the solid obtained by rotating y = x^2 from x = 0 to x = 2 around the x-axis. Use the disk method. Show the integral setup and evaluation steps.",
        "Find a volume",
        "math", "A", 0.90, 0.08,
    ),

    # --- QA tasks ---
    (
        "Explain how photosynthesis works in 3-5 sentences. Include the role of chlorophyll, the inputs (CO2, water, light), and the outputs (glucose, oxygen). Use simple language suitable for a high school student.",
        "Tell me about photosynthesis",
        "qa", "A", 0.90, 0.30,
    ),
    (
        "What is AI",
        "Explain the difference between narrow AI (ANI) and artificial general intelligence (AGI). Provide two real-world examples of each. Discuss current limitations that prevent ANI systems from achieving AGI.",
        "qa", "B", 0.15, 0.90,
    ),
    (
        "Describe the process of how a bill becomes a law in the United States. Cover the key stages from introduction to presidential signature. Keep the explanation concise but complete.",
        "How does a bill become a law?",
        "qa", "A", 0.85, 0.40,
    ),
    (
        "Explain something about black holes or whatever, I guess",
        "Explain how black holes form from stellar collapse. Describe the Schwarzschild radius, event horizon, and singularity. Explain why nothing can escape once past the event horizon using the concept of escape velocity.",
        "qa", "B", 0.08, 0.92,
    ),
    (
        "What causes seasons on Earth? Explain using the concepts of axial tilt and orbital position. Clarify the common misconception that seasons are caused by distance from the Sun.",
        "Why do we have seasons?",
        "qa", "A", 0.88, 0.35,
    ),
]

assert len(TASKS) == 25, f"Expected 25 tasks, got {len(TASKS)}"


def pearson_correlation(x: list[float], y: list[float]) -> float:
    """Compute Pearson correlation coefficient."""
    n = len(x)
    if n < 2:
        return 0.0
    mx = sum(x) / n
    my = sum(y) / n
    sx = math.sqrt(sum((xi - mx) ** 2 for xi in x) / n)
    sy = math.sqrt(sum((yi - my) ** 2 for yi in y) / n)
    if sx == 0 or sy == 0:
        return 0.0
    cov = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y)) / n
    return cov / (sx * sy)


def run_benchmark() -> float:
    """Run the prompt evaluation benchmark and return fitness score."""
    evaluator = PromptEvaluator()

    # --- Component 1: Comparison accuracy (0.5 weight) ---
    correct = 0
    total = len(TASKS)
    for prompt_a, prompt_b, task_type, human_winner, _, _ in TASKS:
        predicted = evaluator.compare(prompt_a, prompt_b, task_type)
        if predicted == human_winner:
            correct += 1

    comparison_accuracy = correct / total

    # --- Component 2: Quality correlation (0.3 weight) ---
    # Check if the evaluator's overall scores correlate with human quality ratings
    evaluator_scores: list[float] = []
    human_scores: list[float] = []
    for prompt_a, prompt_b, task_type, _, hq_a, hq_b in TASKS:
        eval_a = evaluator.evaluate(prompt_a, task_type)
        eval_b = evaluator.evaluate(prompt_b, task_type)
        evaluator_scores.append(eval_a["overall"])
        human_scores.append(hq_a)
        evaluator_scores.append(eval_b["overall"])
        human_scores.append(hq_b)

    raw_corr = pearson_correlation(evaluator_scores, human_scores)
    # Map from [-1, 1] to [0, 1]
    quality_correlation = (raw_corr + 1.0) / 2.0

    # --- Component 3: Explanation quality (0.2 weight) ---
    # Check that explanations are non-empty and contain useful info
    explanation_scores: list[float] = []
    for prompt_a, prompt_b, task_type, _, _, _ in TASKS:
        eval_result = evaluator.evaluate(prompt_a, task_type)
        explanation = eval_result.get("explanation", "")
        if not explanation:
            explanation_scores.append(0.0)
            continue

        # Score explanation quality
        e_score = 0.0
        # Has content?
        if len(explanation) > 10:
            e_score += 0.3
        # Contains numbers (quantitative)?
        if any(c.isdigit() for c in explanation):
            e_score += 0.3
        # Mentions specific metrics?
        metric_words = ["word", "sentence", "keyword", "length", "term"]
        if any(w in explanation.lower() for w in metric_words):
            e_score += 0.4

        explanation_scores.append(min(1.0, e_score))

    explanation_quality = sum(explanation_scores) / max(len(explanation_scores), 1)

    # --- Final fitness ---
    fitness = (
        0.5 * comparison_accuracy
        + 0.3 * quality_correlation
        + 0.2 * explanation_quality
    )

    # Print diagnostics
    print("=== Prompt Evaluation Benchmark ===")
    print(f"Tasks: {len(TASKS)}")
    print()
    print(f"Comparison accuracy: {comparison_accuracy:.4f} ({correct}/{total})")
    print(f"Quality correlation: {quality_correlation:.4f} (raw={raw_corr:+.4f})")
    print(f"Explanation quality: {explanation_quality:.4f}")
    print()
    print(f"score: {fitness:.6f}")
    return fitness


if __name__ == "__main__":
    run_benchmark()
