"""
Benchmark: LLM Judge Quality

Evaluates how well the LLM judge harness agrees with ground-truth human
judgments across three evaluation domains:
  1. Code quality judging (10 examples)
  2. Answer accuracy judging (10 examples)
  3. Instruction following judging (10 examples)

Fitness signal:
  score = 0.30 * judgment_accuracy
        + 0.25 * calibration_quality
        + 0.20 * reasoning_quality
        + 0.15 * cross_domain_consistency
        + 0.10 * token_efficiency

Usage:
    python benchmark.py
"""

from __future__ import annotations

import math
import os
import sys

# Allow importing from contracts/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from harness import LLMJudge

# ── Data structures ────────────────────────────────────────────────

# Each example:
#   task:                  str  -- the original task given to the agent
#   agent_output:          str  -- the agent's response
#   reference:             str  -- reference / correct answer (may be empty)
#   ground_truth_pass:     bool -- should the judge say pass or fail?
#   ground_truth_score:    float -- gold-standard score 0-1
#   domain:                str  -- "code", "answer", "instruction"
#   criteria_keywords:     list[str] -- keywords a good explanation should mention

EXAMPLES: list[dict] = [
    # ================================================================
    # DOMAIN 1: Code quality judging (10 examples)
    # ================================================================
    {
        "task": "Write a Python function that returns the factorial of a non-negative integer.",
        "agent_output": "def factorial(n):\n    if n == 0:\n        return 1\n    return n * factorial(n - 1)",
        "reference": "def factorial(n):\n    if n < 0: raise ValueError\n    if n <= 1: return 1\n    return n * factorial(n-1)",
        "ground_truth_pass": True,
        "ground_truth_score": 0.85,
        "domain": "code",
        "criteria_keywords": ["recursive", "base case", "correct", "negative"],
    },
    {
        "task": "Write a Python function that reverses a string.",
        "agent_output": "def reverse(s):\n    return s[::-1]",
        "reference": "def reverse_string(s: str) -> str:\n    return s[::-1]",
        "ground_truth_pass": True,
        "ground_truth_score": 0.90,
        "domain": "code",
        "criteria_keywords": ["correct", "slice", "concise"],
    },
    {
        "task": "Implement binary search on a sorted list. Return the index or -1.",
        "agent_output": "def binary_search(arr, target):\n    for i, v in enumerate(arr):\n        if v == target:\n            return i\n    return -1",
        "reference": "def binary_search(arr, target):\n    lo, hi = 0, len(arr)-1\n    while lo <= hi:\n        mid = (lo+hi)//2\n        if arr[mid] == target: return mid\n        elif arr[mid] < target: lo = mid+1\n        else: hi = mid-1\n    return -1",
        "ground_truth_pass": False,
        "ground_truth_score": 0.25,
        "domain": "code",
        "criteria_keywords": ["linear", "not binary", "O(n)", "incorrect algorithm"],
    },
    {
        "task": "Write a function to check if a string is a palindrome, ignoring case and spaces.",
        "agent_output": "def is_palindrome(s):\n    s = s.lower().replace(' ', '')\n    return s == s[::-1]",
        "reference": "def is_palindrome(s):\n    cleaned = ''.join(c.lower() for c in s if c.isalnum())\n    return cleaned == cleaned[::-1]",
        "ground_truth_pass": True,
        "ground_truth_score": 0.80,
        "domain": "code",
        "criteria_keywords": ["palindrome", "case", "spaces", "partial"],
    },
    {
        "task": "Write a function that merges two sorted lists into one sorted list.",
        "agent_output": "def merge(a, b):\n    return sorted(a + b)",
        "reference": "def merge(a, b):\n    result, i, j = [], 0, 0\n    while i < len(a) and j < len(b):\n        if a[i] <= b[j]: result.append(a[i]); i += 1\n        else: result.append(b[j]); j += 1\n    result.extend(a[i:]); result.extend(b[j:])\n    return result",
        "ground_truth_pass": True,
        "ground_truth_score": 0.55,
        "domain": "code",
        "criteria_keywords": ["correct output", "not O(n)", "sorted", "inefficient"],
    },
    {
        "task": "Implement a stack with push, pop, and peek methods.",
        "agent_output": "class Stack:\n    def __init__(self):\n        self.items = []\n    def push(self, item):\n        self.items.append(item)\n    def pop(self):\n        return self.items.pop()\n    def peek(self):\n        return self.items[-1]",
        "reference": "class Stack with push/pop/peek, raising IndexError on empty.",
        "ground_truth_pass": True,
        "ground_truth_score": 0.75,
        "domain": "code",
        "criteria_keywords": ["stack", "push", "pop", "error handling", "empty"],
    },
    {
        "task": "Write a function that counts word frequencies in a string.",
        "agent_output": "def word_freq(s):\n    counts = {}\n    for word in s.split():\n        counts[word] = counts.get(word, 0) + 1\n    return counts",
        "reference": "def word_freq(text):\n    words = text.lower().split()\n    return {w: words.count(w) for w in set(words)}",
        "ground_truth_pass": True,
        "ground_truth_score": 0.80,
        "domain": "code",
        "criteria_keywords": ["frequency", "dictionary", "case"],
    },
    {
        "task": "Write a Python function to compute the nth Fibonacci number.",
        "agent_output": "def fib(n):\n    return fib(n-1) + fib(n-2)",
        "reference": "def fib(n):\n    if n <= 1: return n\n    a, b = 0, 1\n    for _ in range(2, n+1): a, b = b, a+b\n    return b",
        "ground_truth_pass": False,
        "ground_truth_score": 0.10,
        "domain": "code",
        "criteria_keywords": ["infinite recursion", "no base case", "crash", "error"],
    },
    {
        "task": "Write a function that flattens a nested list.",
        "agent_output": "I'm not sure how to do this but maybe try using itertools?",
        "reference": "def flatten(lst):\n    result = []\n    for item in lst:\n        if isinstance(item, list): result.extend(flatten(item))\n        else: result.append(item)\n    return result",
        "ground_truth_pass": False,
        "ground_truth_score": 0.05,
        "domain": "code",
        "criteria_keywords": ["no code", "no implementation", "incomplete"],
    },
    {
        "task": "Write a function that removes duplicates from a list while preserving order.",
        "agent_output": "def remove_dupes(lst):\n    seen = set()\n    result = []\n    for x in lst:\n        if x not in seen:\n            seen.add(x)\n            result.append(x)\n    return result",
        "reference": "def remove_duplicates(lst):\n    seen = set()\n    return [x for x in lst if not (x in seen or seen.add(x))]",
        "ground_truth_pass": True,
        "ground_truth_score": 0.95,
        "domain": "code",
        "criteria_keywords": ["correct", "order", "set", "efficient"],
    },

    # ================================================================
    # DOMAIN 2: Answer accuracy judging (10 examples)
    # ================================================================
    {
        "task": "What is the speed of light in a vacuum?",
        "agent_output": "The speed of light in a vacuum is approximately 299,792,458 meters per second.",
        "reference": "299,792,458 m/s",
        "ground_truth_pass": True,
        "ground_truth_score": 0.95,
        "domain": "answer",
        "criteria_keywords": ["correct", "precise", "units"],
    },
    {
        "task": "What is the chemical formula for glucose?",
        "agent_output": "C6H12O6",
        "reference": "C6H12O6",
        "ground_truth_pass": True,
        "ground_truth_score": 1.0,
        "domain": "answer",
        "criteria_keywords": ["exact", "correct", "formula"],
    },
    {
        "task": "Who wrote the novel '1984'?",
        "agent_output": "George Orwell wrote 1984, published in 1949.",
        "reference": "George Orwell",
        "ground_truth_pass": True,
        "ground_truth_score": 0.95,
        "domain": "answer",
        "criteria_keywords": ["correct", "author", "additional info"],
    },
    {
        "task": "What is the boiling point of water at sea level in Celsius?",
        "agent_output": "Water boils at 212 degrees.",
        "reference": "100 degrees Celsius",
        "ground_truth_pass": False,
        "ground_truth_score": 0.30,
        "domain": "answer",
        "criteria_keywords": ["wrong unit", "fahrenheit", "not celsius"],
    },
    {
        "task": "What is the capital of Australia?",
        "agent_output": "Sydney is the capital of Australia.",
        "reference": "Canberra",
        "ground_truth_pass": False,
        "ground_truth_score": 0.10,
        "domain": "answer",
        "criteria_keywords": ["incorrect", "common mistake", "sydney"],
    },
    {
        "task": "How many chromosomes do humans have?",
        "agent_output": "Humans have 46 chromosomes, arranged in 23 pairs.",
        "reference": "46 chromosomes (23 pairs)",
        "ground_truth_pass": True,
        "ground_truth_score": 1.0,
        "domain": "answer",
        "criteria_keywords": ["correct", "pairs", "complete"],
    },
    {
        "task": "What is the largest planet in our solar system?",
        "agent_output": "Saturn is the largest planet.",
        "reference": "Jupiter",
        "ground_truth_pass": False,
        "ground_truth_score": 0.05,
        "domain": "answer",
        "criteria_keywords": ["incorrect", "saturn", "wrong planet"],
    },
    {
        "task": "What year did World War II end?",
        "agent_output": "World War II ended in 1945 with the surrender of both Germany in May and Japan in September.",
        "reference": "1945",
        "ground_truth_pass": True,
        "ground_truth_score": 0.95,
        "domain": "answer",
        "criteria_keywords": ["correct", "1945", "detailed"],
    },
    {
        "task": "What is the square root of 144?",
        "agent_output": "The square root of 144 is 14.",
        "reference": "12",
        "ground_truth_pass": False,
        "ground_truth_score": 0.0,
        "domain": "answer",
        "criteria_keywords": ["incorrect", "math error", "wrong"],
    },
    {
        "task": "Explain what DNA stands for.",
        "agent_output": "DNA stands for deoxyribonucleic acid. It is the molecule that carries genetic information in living organisms.",
        "reference": "Deoxyribonucleic acid",
        "ground_truth_pass": True,
        "ground_truth_score": 0.95,
        "domain": "answer",
        "criteria_keywords": ["correct", "expanded", "explanation"],
    },

    # ================================================================
    # DOMAIN 3: Instruction following judging (10 examples)
    # ================================================================
    {
        "task": "Write exactly 3 bullet points summarizing the benefits of exercise.",
        "agent_output": "- Improves cardiovascular health\n- Boosts mental well-being and reduces stress\n- Helps maintain a healthy weight",
        "reference": "3 bullet points about exercise benefits",
        "ground_truth_pass": True,
        "ground_truth_score": 1.0,
        "domain": "instruction",
        "criteria_keywords": ["3 bullets", "format correct", "on topic"],
    },
    {
        "task": "Respond in exactly one sentence. What is machine learning?",
        "agent_output": "Machine learning is a subset of artificial intelligence where computers learn patterns from data to make predictions or decisions without being explicitly programmed.",
        "reference": "One sentence definition of ML",
        "ground_truth_pass": True,
        "ground_truth_score": 0.95,
        "domain": "instruction",
        "criteria_keywords": ["one sentence", "complete", "accurate"],
    },
    {
        "task": "List 5 programming languages. Use a numbered list.",
        "agent_output": "Python, Java, C++, JavaScript, and Rust are popular programming languages.",
        "reference": "1. Python\n2. Java\n3. C++\n4. JavaScript\n5. Rust",
        "ground_truth_pass": False,
        "ground_truth_score": 0.40,
        "domain": "instruction",
        "criteria_keywords": ["not numbered", "wrong format", "content correct"],
    },
    {
        "task": "Write a haiku (5-7-5 syllable pattern) about the ocean.",
        "agent_output": "Waves crash on the shore\nSalty breeze whispers softly\nSea gulls cry above",
        "reference": "A haiku with 5-7-5 syllables about ocean",
        "ground_truth_pass": True,
        "ground_truth_score": 0.90,
        "domain": "instruction",
        "criteria_keywords": ["haiku", "syllable", "ocean", "format"],
    },
    {
        "task": "Explain photosynthesis in exactly 2 paragraphs. Do not use the word 'sunlight'.",
        "agent_output": "Photosynthesis is the process by which plants convert sunlight, water, and carbon dioxide into glucose and oxygen. It occurs in the chloroplasts of plant cells.\n\nThe process has two stages: light reactions and the Calvin cycle. These work together to produce the energy plants need to grow.",
        "reference": "2 paragraphs about photosynthesis without the word 'sunlight'",
        "ground_truth_pass": False,
        "ground_truth_score": 0.35,
        "domain": "instruction",
        "criteria_keywords": ["used sunlight", "forbidden word", "constraint violated"],
    },
    {
        "task": "Translate 'hello' into French, Spanish, and German. Format as a table.",
        "agent_output": "| Language | Translation |\n|----------|-------------|\n| French   | Bonjour     |\n| Spanish  | Hola        |\n| German   | Hallo       |",
        "reference": "Table with French/Spanish/German translations of hello",
        "ground_truth_pass": True,
        "ground_truth_score": 1.0,
        "domain": "instruction",
        "criteria_keywords": ["table", "correct translations", "all languages"],
    },
    {
        "task": "Write a 50-word summary of climate change. Count must be exactly 50 words.",
        "agent_output": "Climate change refers to long-term shifts in global temperatures and weather patterns. Human activities, especially burning fossil fuels, have been the main driver since the 1800s. Rising temperatures cause ice caps to melt, sea levels to rise, and extreme weather events to intensify. Urgent action is needed to reduce emissions.",
        "reference": "Exactly 50-word summary of climate change",
        "ground_truth_pass": True,
        "ground_truth_score": 0.85,
        "domain": "instruction",
        "criteria_keywords": ["word count", "approximately 50", "on topic"],
    },
    {
        "task": "Answer the following question using ONLY 'yes' or 'no': Is Python an interpreted language?",
        "agent_output": "Yes, Python is generally considered an interpreted language, though technically it compiles to bytecode first.",
        "reference": "yes",
        "ground_truth_pass": False,
        "ground_truth_score": 0.30,
        "domain": "instruction",
        "criteria_keywords": ["too verbose", "not yes/no only", "extra text"],
    },
    {
        "task": "Write the numbers 1-5 in reverse order, separated by commas.",
        "agent_output": "5, 4, 3, 2, 1",
        "reference": "5, 4, 3, 2, 1",
        "ground_truth_pass": True,
        "ground_truth_score": 1.0,
        "domain": "instruction",
        "criteria_keywords": ["exact", "correct order", "format"],
    },
    {
        "task": "Describe a cat in exactly 3 adjectives separated by commas. Nothing else.",
        "agent_output": "Cats are wonderful animals. They are fluffy, independent, and curious.",
        "reference": "fluffy, independent, curious",
        "ground_truth_pass": False,
        "ground_truth_score": 0.40,
        "domain": "instruction",
        "criteria_keywords": ["extra text", "not just adjectives", "partial"],
    },
]

assert len(EXAMPLES) == 30, f"Expected 30 examples, got {len(EXAMPLES)}"
assert sum(1 for e in EXAMPLES if e["domain"] == "code") == 10
assert sum(1 for e in EXAMPLES if e["domain"] == "answer") == 10
assert sum(1 for e in EXAMPLES if e["domain"] == "instruction") == 10


# ── Helper functions ───────────────────────────────────────────────

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


def expected_calibration_error(
    confidences: list[float],
    accuracies: list[bool],
    n_bins: int = 5,
) -> float:
    """Compute Expected Calibration Error (ECE).

    Groups predictions into bins by confidence, then measures the gap
    between average confidence and actual accuracy in each bin.
    """
    if not confidences:
        return 1.0

    bins: list[list[tuple[float, bool]]] = [[] for _ in range(n_bins)]
    for conf, acc in zip(confidences, accuracies):
        idx = min(int(conf * n_bins), n_bins - 1)
        bins[idx].append((conf, acc))

    ece = 0.0
    total = len(confidences)
    for b in bins:
        if not b:
            continue
        avg_conf = sum(c for c, _ in b) / len(b)
        avg_acc = sum(1.0 for _, a in b if a) / len(b)
        ece += (len(b) / total) * abs(avg_conf - avg_acc)

    return ece


# ── Main benchmark ─────────────────────────────────────────────────

def run_benchmark() -> float:
    """Run the LLM judge benchmark and return fitness score."""
    judge = LLMJudge()

    # Collect judgments
    judge_scores: list[float] = []
    gt_scores: list[float] = []
    confidences: list[float] = []
    pass_correct: list[bool] = []
    reasoning_scores: list[float] = []
    domain_accuracies: dict[str, list[float]] = {
        "code": [], "answer": [], "instruction": [],
    }

    for ex in EXAMPLES:
        result = judge.judge(
            agent_output=ex["agent_output"],
            task=ex["task"],
            reference=ex["reference"],
        )

        j_score = result["score"]
        j_pass = result["pass"]
        j_conf = result["confidence"]
        j_reasoning = result.get("reasoning", "")

        judge_scores.append(j_score)
        gt_scores.append(ex["ground_truth_score"])
        confidences.append(j_conf)

        # Did the judge's pass/fail agree with ground truth?
        is_correct = j_pass == ex["ground_truth_pass"]
        pass_correct.append(is_correct)
        domain_accuracies[ex["domain"]].append(1.0 if is_correct else 0.0)

        # Score the reasoning quality: does it mention task-relevant criteria?
        r_score = 0.0
        if j_reasoning:
            reasoning_lower = j_reasoning.lower()
            criteria = ex["criteria_keywords"]
            if criteria:
                matched = sum(1 for kw in criteria if kw in reasoning_lower)
                r_score = matched / len(criteria)
        reasoning_scores.append(r_score)

    # ── Component 1: Judgment accuracy (0.30) ──────────────────────
    # Correlation between judge scores and ground truth scores
    raw_corr = pearson_correlation(judge_scores, gt_scores)
    # Map [-1, 1] -> [0, 1]
    judgment_accuracy = (raw_corr + 1.0) / 2.0

    # ── Component 2: Calibration quality (0.25) ────────────────────
    # 1 - ECE: well-calibrated confidence
    ece = expected_calibration_error(confidences, pass_correct)
    calibration_quality = 1.0 - ece

    # ── Component 3: Reasoning quality (0.20) ──────────────────────
    reasoning_quality = sum(reasoning_scores) / max(len(reasoning_scores), 1)

    # ── Component 4: Cross-domain consistency (0.15) ───────────────
    domain_accs = {}
    for domain, accs in domain_accuracies.items():
        domain_accs[domain] = sum(accs) / max(len(accs), 1)
    if domain_accs:
        min_acc = min(domain_accs.values())
        max_acc = max(domain_accs.values())
        cross_domain_consistency = min_acc / max_acc if max_acc > 0 else 0.0
    else:
        cross_domain_consistency = 0.0

    # ── Component 5: Token efficiency (0.10) ───────────────────────
    # Estimate prompt tokens from the harness configuration
    import harness
    total_chars = len(harness.SYSTEM_PROMPT) + len(harness.USER_TEMPLATE)
    for ex in harness.FEW_SHOT_EXAMPLES:
        total_chars += sum(len(v) for v in ex.values())
    est_tokens = total_chars // 4
    token_efficiency = max(0.0, min(1.0, 1.0 - est_tokens / 2000))

    # ── Final fitness ──────────────────────────────────────────────
    fitness = (
        0.30 * judgment_accuracy
        + 0.25 * calibration_quality
        + 0.20 * reasoning_quality
        + 0.15 * cross_domain_consistency
        + 0.10 * token_efficiency
    )

    # Print diagnostics
    print("=== LLM Judge Benchmark ===")
    print(f"Examples: {len(EXAMPLES)} (10 code, 10 answer, 10 instruction)")
    print()

    print(f"Judgment accuracy:         {judgment_accuracy:.4f}  (raw correlation={raw_corr:+.4f})")
    print(f"Calibration quality:       {calibration_quality:.4f}  (ECE={ece:.4f})")
    print(f"Reasoning quality:         {reasoning_quality:.4f}")
    print(f"Cross-domain consistency:  {cross_domain_consistency:.4f}")
    for domain, acc in sorted(domain_accs.items()):
        print(f"  {domain:15s} accuracy = {acc:.4f}")
    print(f"Token efficiency:          {token_efficiency:.4f}  (~{est_tokens} tokens)")
    print()

    pass_acc = sum(1 for c in pass_correct if c) / max(len(pass_correct), 1)
    print(f"Pass/fail accuracy:        {pass_acc:.4f} ({sum(1 for c in pass_correct if c)}/{len(pass_correct)})")
    print()
    print(f"score: {fitness:.6f}")
    return fitness


if __name__ == "__main__":
    run_benchmark()
