"""
Benchmark: Agent Trajectory Curation

Evaluates how well the trajectory curator scores and selects
high-quality agent execution traces for training data.
20 tasks with a pool of 50 trajectories.

Fitness signal: 0.5 * scoring_correlation + 0.3 * selection_quality + 0.2 * diversity

Usage:
    python benchmark.py
"""

from __future__ import annotations

import math
import os
import sys

# Allow importing from contracts/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from harness import TrajectoryCurator

# ── Trajectory pool: 50 trajectories with ground-truth quality ────────
# Each trajectory is a list of steps; each step: {"action": str, "result": str, "success": bool}
# ground_truth_quality: 0.0 - 1.0 (human-rated quality)
# category: "efficient", "correct_verbose", "partial", "failing", "wandering"


def _make_trajectory(
    n_steps: int,
    success_rate: float,
    final_success: bool,
) -> list[dict]:
    """Generate a synthetic trajectory."""
    steps = []
    for i in range(n_steps):
        is_last = i == n_steps - 1
        success = final_success if is_last else (
            (i + 1) / n_steps >= (1.0 - success_rate)
        )
        steps.append({
            "action": f"step_{i}_action",
            "result": f"step_{i}_result{'_ok' if success else '_fail'}",
            "success": success,
        })
    return steps


# Build 50 trajectories with ground truth
TRAJECTORY_POOL: list[tuple[list[dict], float, str]] = [
    # --- Efficient & correct (5 trajectories, quality 0.85-1.0) ---
    (_make_trajectory(1, 1.0, True), 1.0, "efficient"),
    (_make_trajectory(2, 1.0, True), 0.95, "efficient"),
    (_make_trajectory(2, 1.0, True), 0.93, "efficient"),
    (_make_trajectory(3, 1.0, True), 0.90, "efficient"),
    (_make_trajectory(3, 0.9, True), 0.85, "efficient"),

    # --- Correct but verbose (10 trajectories, quality 0.5-0.75) ---
    (_make_trajectory(5, 0.8, True), 0.75, "correct_verbose"),
    (_make_trajectory(6, 0.7, True), 0.70, "correct_verbose"),
    (_make_trajectory(7, 0.8, True), 0.68, "correct_verbose"),
    (_make_trajectory(8, 0.7, True), 0.65, "correct_verbose"),
    (_make_trajectory(8, 0.6, True), 0.60, "correct_verbose"),
    (_make_trajectory(10, 0.8, True), 0.58, "correct_verbose"),
    (_make_trajectory(10, 0.7, True), 0.55, "correct_verbose"),
    (_make_trajectory(12, 0.6, True), 0.52, "correct_verbose"),
    (_make_trajectory(12, 0.7, True), 0.50, "correct_verbose"),
    (_make_trajectory(15, 0.5, True), 0.48, "correct_verbose"),

    # --- Partial success (10 trajectories, quality 0.25-0.45) ---
    (_make_trajectory(3, 0.5, False), 0.45, "partial"),
    (_make_trajectory(4, 0.6, False), 0.42, "partial"),
    (_make_trajectory(5, 0.4, False), 0.40, "partial"),
    (_make_trajectory(5, 0.5, False), 0.38, "partial"),
    (_make_trajectory(6, 0.3, False), 0.35, "partial"),
    (_make_trajectory(7, 0.4, False), 0.33, "partial"),
    (_make_trajectory(8, 0.3, False), 0.30, "partial"),
    (_make_trajectory(8, 0.5, False), 0.28, "partial"),
    (_make_trajectory(10, 0.3, False), 0.27, "partial"),
    (_make_trajectory(10, 0.4, False), 0.25, "partial"),

    # --- Wandering / wasteful (15 trajectories, quality 0.05-0.20) ---
    (_make_trajectory(15, 0.3, True), 0.20, "wandering"),
    (_make_trajectory(18, 0.2, True), 0.18, "wandering"),
    (_make_trajectory(20, 0.3, True), 0.17, "wandering"),
    (_make_trajectory(20, 0.2, False), 0.15, "wandering"),
    (_make_trajectory(22, 0.2, True), 0.14, "wandering"),
    (_make_trajectory(25, 0.1, False), 0.13, "wandering"),
    (_make_trajectory(25, 0.2, True), 0.12, "wandering"),
    (_make_trajectory(28, 0.1, False), 0.11, "wandering"),
    (_make_trajectory(30, 0.2, True), 0.10, "wandering"),
    (_make_trajectory(30, 0.1, False), 0.09, "wandering"),
    (_make_trajectory(32, 0.1, False), 0.08, "wandering"),
    (_make_trajectory(35, 0.1, True), 0.07, "wandering"),
    (_make_trajectory(35, 0.0, False), 0.06, "wandering"),
    (_make_trajectory(40, 0.1, False), 0.05, "wandering"),
    (_make_trajectory(40, 0.0, False), 0.05, "wandering"),

    # --- Complete failures (10 trajectories, quality 0.0-0.04) ---
    (_make_trajectory(1, 0.0, False), 0.04, "failing"),
    (_make_trajectory(2, 0.0, False), 0.03, "failing"),
    (_make_trajectory(3, 0.0, False), 0.03, "failing"),
    (_make_trajectory(5, 0.0, False), 0.02, "failing"),
    (_make_trajectory(5, 0.0, False), 0.02, "failing"),
    (_make_trajectory(8, 0.0, False), 0.01, "failing"),
    (_make_trajectory(10, 0.0, False), 0.01, "failing"),
    (_make_trajectory(15, 0.0, False), 0.01, "failing"),
    (_make_trajectory(20, 0.0, False), 0.0, "failing"),
    (_make_trajectory(25, 0.0, False), 0.0, "failing"),
]

assert len(TRAJECTORY_POOL) == 50, f"Expected 50 trajectories, got {len(TRAJECTORY_POOL)}"


def spearman_correlation(x: list[float], y: list[float]) -> float:
    """Compute Spearman rank correlation."""
    n = len(x)
    if n < 2:
        return 0.0

    def _rank(vals: list[float]) -> list[float]:
        indexed = sorted(enumerate(vals), key=lambda t: t[1])
        ranks = [0.0] * n
        for rank, (orig_i, _) in enumerate(indexed):
            ranks[orig_i] = float(rank)
        return ranks

    rx = _rank(x)
    ry = _rank(y)
    # Pearson on ranks
    mx = sum(rx) / n
    my = sum(ry) / n
    sx = math.sqrt(sum((r - mx) ** 2 for r in rx) / n)
    sy = math.sqrt(sum((r - my) ** 2 for r in ry) / n)
    if sx == 0 or sy == 0:
        return 0.0
    cov = sum((rx[i] - mx) * (ry[i] - my) for i in range(n)) / n
    return cov / (sx * sy)


def run_benchmark() -> float:
    """Run the trajectory curation benchmark and return fitness score."""
    curator = TrajectoryCurator()

    trajectories = [t[0] for t in TRAJECTORY_POOL]
    ground_truth = [t[1] for t in TRAJECTORY_POOL]
    categories = [t[2] for t in TRAJECTORY_POOL]

    # --- Component 1: Scoring correlation (0.5 weight) ---
    curator_scores = [curator.score_trajectory(t) for t in trajectories]
    scoring_correlation_raw = spearman_correlation(curator_scores, ground_truth)
    # Map [-1, 1] to [0, 1]
    scoring_correlation = (scoring_correlation_raw + 1.0) / 2.0

    # --- Component 2: Selection quality (0.3 weight) ---
    # Select top 10 trajectories. Compare to ground truth top 10.
    budget = 10
    selected_indices = curator.select_trajectories(trajectories, budget)

    # Ground truth best 10 by quality
    gt_ranked = sorted(range(len(ground_truth)), key=lambda i: -ground_truth[i])
    gt_top = set(gt_ranked[:budget])

    selected_set = set(selected_indices)

    # Overlap with ground truth selection
    overlap = len(selected_set & gt_top)
    overlap_score = overlap / budget

    # Average quality of selected vs average quality of best possible
    if selected_indices:
        selected_avg_quality = sum(ground_truth[i] for i in selected_indices) / len(selected_indices)
    else:
        selected_avg_quality = 0.0
    best_avg_quality = sum(ground_truth[i] for i in gt_ranked[:budget]) / budget
    quality_ratio = selected_avg_quality / max(best_avg_quality, 0.001)

    selection_quality = 0.5 * overlap_score + 0.5 * min(1.0, quality_ratio)

    # --- Component 3: Diversity (0.2 weight) ---
    # Do the selected trajectories cover different categories?
    if selected_indices:
        selected_categories = set(categories[i] for i in selected_indices)
        all_categories = set(categories)
        category_coverage = len(selected_categories) / len(all_categories)

        # Also check step-count diversity
        selected_lengths = [len(trajectories[i]) for i in selected_indices]
        if len(selected_lengths) > 1:
            mean_len = sum(selected_lengths) / len(selected_lengths)
            variance = sum((l - mean_len) ** 2 for l in selected_lengths) / len(selected_lengths)
            std_dev = math.sqrt(variance)
            # Higher std_dev = more diverse (up to a point)
            length_diversity = min(1.0, std_dev / 10.0)
        else:
            length_diversity = 0.0
    else:
        category_coverage = 0.0
        length_diversity = 0.0

    diversity = 0.6 * category_coverage + 0.4 * length_diversity

    # --- Final fitness ---
    fitness = (
        0.5 * scoring_correlation
        + 0.3 * selection_quality
        + 0.2 * diversity
    )

    # Print diagnostics
    print("=== Agent Trajectory Curation Benchmark ===")
    print(f"Trajectory pool: {len(TRAJECTORY_POOL)}")
    print(f"Budget: {budget}")
    print()
    print(f"Scoring correlation: {scoring_correlation:.4f} (raw={scoring_correlation_raw:+.4f})")
    print(f"Selection quality:   {selection_quality:.4f}")
    print(f"  overlap: {overlap}/{budget}")
    print(f"  quality ratio: {quality_ratio:.4f}")
    print(f"Diversity:           {diversity:.4f}")
    print(f"  categories: {len(selected_categories) if selected_indices else 0}/{len(set(categories))}")
    print()
    print(f"score: {fitness:.6f}")
    return fitness


if __name__ == "__main__":
    run_benchmark()
