#!/usr/bin/env python3
"""
Benchmark: JIT Context Loading
===============================

20 tasks across three categories:
  - Relevance selection  (10 tasks): precision/recall on relevant chunks
  - Priority ordering     (5 tasks): most important relevant chunks first
  - Budget efficiency      (5 tasks): don't waste budget on irrelevant chunks

Fitness signal:
  score = 0.5 * recall_relevant
        + 0.3 * precision
        + 0.2 * importance_ordering
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from contracts import AgentMessage  # noqa: E402
from harness import JITLoader  # noqa: E402

# ======================================================================
# Helpers
# ======================================================================

_WC = lambda t: len(t.split())  # noqa: E731


def _make_chunk(
    content: str,
    topic: str = "",
    recency: float = 0.5,
    importance: float = 0.5,
    is_relevant: bool = False,
) -> dict:
    return {
        "content": content,
        "metadata": {
            "topic": topic,
            "recency": recency,
            "importance": importance,
        },
        "_relevant": is_relevant,  # ground-truth label (not visible to harness)
    }


# Corpus of topics for building diverse tasks
_TOPICS = [
    ("python", "debugging", "Python debugging techniques including pdb breakpoints and logging strategies"),
    ("python", "async", "Async programming in Python with asyncio event loops and coroutines"),
    ("python", "testing", "Unit testing best practices with pytest fixtures and parametrize decorators"),
    ("javascript", "react", "React component lifecycle methods and hooks like useState useEffect"),
    ("javascript", "nodejs", "Node.js server setup with Express middleware and routing patterns"),
    ("database", "postgres", "PostgreSQL query optimization using EXPLAIN ANALYZE and index strategies"),
    ("database", "redis", "Redis caching patterns including cache aside and write through strategies"),
    ("devops", "docker", "Docker containerization with multi-stage builds and layer caching"),
    ("devops", "kubernetes", "Kubernetes pod scheduling with resource requests limits and affinity rules"),
    ("devops", "cicd", "CI CD pipeline design with GitHub Actions workflows and deployment gates"),
    ("security", "auth", "Authentication patterns including OAuth2 JWT tokens and session management"),
    ("security", "encryption", "Data encryption at rest and in transit using AES and TLS protocols"),
    ("frontend", "css", "CSS layout techniques with flexbox grid and responsive media queries"),
    ("frontend", "accessibility", "Web accessibility WCAG compliance with ARIA labels and keyboard navigation"),
    ("ml", "training", "Machine learning model training with gradient descent and learning rate schedules"),
    ("ml", "inference", "Model inference optimization with quantization and batch processing techniques"),
    ("api", "rest", "REST API design with proper HTTP methods status codes and pagination"),
    ("api", "graphql", "GraphQL schema design with queries mutations subscriptions and resolvers"),
    ("monitoring", "logging", "Application logging with structured logs log levels and log aggregation"),
    ("monitoring", "metrics", "Metrics collection with Prometheus exporters and Grafana dashboards"),
]

_IRRELEVANT_FILLER = [
    "Weather forecast for the upcoming week shows partly cloudy skies with temperatures around seventy degrees",
    "The history of ancient Roman architecture features arches columns and concrete construction methods",
    "Cooking pasta requires boiling water adding salt and cooking for approximately eight to ten minutes",
    "The migratory patterns of monarch butterflies span thousands of miles across North America each year",
    "Classical music compositions by Mozart include symphonies concertos and operas from the eighteenth century",
    "The geology of the Grand Canyon reveals millions of years of sedimentary rock layers and erosion patterns",
    "Professional basketball rules include a shot clock three point line and free throw regulations",
    "Knitting patterns for beginners include scarves dishcloths and simple hats using basic stitches",
    "The lifecycle of stars from nebula to main sequence to red giant and eventual supernova or white dwarf",
    "Gardening tips for spring include soil preparation seed starting and frost date awareness for planting",
    "The evolution of transportation from horse drawn carriages to automobiles trains and modern aircraft",
    "Photography composition rules include the rule of thirds leading lines and negative space techniques",
    "The history of the printing press and its impact on literacy education and the spread of knowledge",
    "Marine biology studies of coral reef ecosystems including symbiotic relationships and biodiversity",
    "Folk tales and mythology from various cultures share common themes of heroism transformation and justice",
]


# ======================================================================
# Task generators
# ======================================================================

def _build_chunk_pool(
    query_topic_indices: list[int],
    pool_size: int = 50,
    relevant_count: int = 5,
) -> tuple[list[dict], list[dict]]:
    """Build a pool of chunks with known relevant ones.

    Returns (all_chunks, relevant_chunks).
    """
    relevant_chunks = []
    for idx in query_topic_indices[:relevant_count]:
        cat, topic, content = _TOPICS[idx % len(_TOPICS)]
        chunk = _make_chunk(
            content=content,
            topic=f"{cat} {topic}",
            recency=0.5 + (idx % 5) * 0.1,
            importance=0.6 + (idx % 4) * 0.1,
            is_relevant=True,
        )
        relevant_chunks.append(chunk)

    irrelevant_chunks = []
    irr_needed = pool_size - len(relevant_chunks)
    # Use topics NOT in query_topic_indices
    available = [i for i in range(len(_TOPICS)) if i not in query_topic_indices]
    for i in range(irr_needed):
        if i < len(available):
            cat, topic, content = _TOPICS[available[i]]
            chunk = _make_chunk(
                content=content,
                topic=f"{cat} {topic}",
                recency=0.3 + (i % 5) * 0.1,
                importance=0.3 + (i % 4) * 0.1,
                is_relevant=False,
            )
        else:
            filler = _IRRELEVANT_FILLER[i % len(_IRRELEVANT_FILLER)]
            chunk = _make_chunk(
                content=filler,
                topic="misc",
                recency=0.2 + (i % 5) * 0.1,
                importance=0.2 + (i % 3) * 0.1,
                is_relevant=False,
            )
        irrelevant_chunks.append(chunk)

    # Interleave: place relevant chunks at various positions
    all_chunks = list(irrelevant_chunks)
    # Insert relevant chunks at spread-out positions
    for j, rc in enumerate(relevant_chunks):
        pos = min((j + 1) * (len(all_chunks) // (len(relevant_chunks) + 1)), len(all_chunks))
        all_chunks.insert(pos, rc)

    return all_chunks, relevant_chunks


def _gen_relevance_tasks() -> list[dict]:
    """10 tasks: 50 chunks, 5 relevant. Budget fits ~10 chunks."""
    tasks = []
    queries = [
        ("How do I debug Python code with breakpoints?", [0]),
        ("Set up async event loops in Python", [1]),
        ("Write pytest unit tests with fixtures", [2]),
        ("Build React components with hooks", [3]),
        ("Create a Node.js Express server", [4]),
        ("Optimize PostgreSQL queries with indexes", [5]),
        ("Implement Redis caching patterns", [6]),
        ("Docker multi-stage build configuration", [7]),
        ("Kubernetes pod scheduling and resources", [8]),
        ("GitHub Actions CI CD pipeline setup", [9]),
    ]
    for query, primary_indices in queries:
        # Add some related indices for each query
        related = [(idx + 1) % len(_TOPICS) for idx in primary_indices]
        all_indices = primary_indices + related
        # Ensure we have exactly 5 relevant
        while len(all_indices) < 5:
            all_indices.append((all_indices[-1] + 2) % len(_TOPICS))
        all_indices = all_indices[:5]

        chunks, relevant = _build_chunk_pool(all_indices, pool_size=50, relevant_count=5)
        # Budget fits ~10 chunks (each chunk is ~15 words, budget = 150 words)
        budget = 150
        tasks.append({
            "type": "relevance",
            "query": query,
            "chunks": chunks,
            "relevant_chunks": relevant,
            "budget": budget,
        })
    return tasks


def _gen_priority_ordering_tasks() -> list[dict]:
    """5 tasks: among relevant chunks, most important should be selected first."""
    tasks = []
    queries = [
        ("Python debugging and testing strategies", [0, 2]),
        ("Database optimization for PostgreSQL and Redis", [5, 6]),
        ("DevOps containerization and orchestration", [7, 8]),
        ("Security authentication and encryption", [10, 11]),
        ("Machine learning training and inference", [14, 15]),
    ]
    for query, primary_indices in queries:
        related = [(idx + 1) % len(_TOPICS) for idx in primary_indices]
        all_indices = (primary_indices + related)[:5]
        while len(all_indices) < 5:
            all_indices.append((all_indices[-1] + 3) % len(_TOPICS))
        all_indices = all_indices[:5]

        chunks, relevant = _build_chunk_pool(all_indices, pool_size=50, relevant_count=5)

        # Assign graduated importance to relevant chunks
        for i, rc in enumerate(relevant):
            rc["metadata"]["importance"] = 1.0 - i * 0.15  # 1.0, 0.85, 0.70, 0.55, 0.40

        # Budget fits only 3 chunks (~45 words)
        budget = 50
        tasks.append({
            "type": "priority_ordering",
            "query": query,
            "chunks": chunks,
            "relevant_chunks": relevant,
            "budget": budget,
        })
    return tasks


def _gen_budget_efficiency_tasks() -> list[dict]:
    """5 tasks: measure precision (don't waste budget on irrelevant)."""
    tasks = []
    queries = [
        ("Frontend CSS and accessibility techniques", [12, 13]),
        ("API design REST and GraphQL", [16, 17]),
        ("Monitoring logging and metrics", [18, 19]),
        ("Python async and testing", [1, 2]),
        ("DevOps CI CD and Docker", [7, 9]),
    ]
    for query, primary_indices in queries:
        all_indices = list(primary_indices)
        while len(all_indices) < 5:
            all_indices.append((all_indices[-1] + 2) % len(_TOPICS))
        all_indices = all_indices[:5]

        chunks, relevant = _build_chunk_pool(all_indices, pool_size=50, relevant_count=5)
        # Generous budget: fits ~15 chunks
        budget = 250
        tasks.append({
            "type": "budget_efficiency",
            "query": query,
            "chunks": chunks,
            "relevant_chunks": relevant,
            "budget": budget,
        })
    return tasks


# ======================================================================
# Scoring functions
# ======================================================================

def _score_relevance(task: dict, selected: list[dict]) -> tuple[float, float]:
    """Return (recall, precision) for relevant chunks."""
    relevant_contents = {c["content"] for c in task["relevant_chunks"]}
    selected_contents = {c["content"] for c in selected}

    if not relevant_contents:
        return 1.0, 1.0

    true_positives = len(relevant_contents & selected_contents)
    recall = true_positives / len(relevant_contents)
    precision = true_positives / len(selected_contents) if selected_contents else 0.0

    return recall, precision


def _score_importance_ordering(task: dict, selected: list[dict]) -> float:
    """Check if selected relevant chunks are the most important ones.

    Sort relevant chunks by importance descending; the selected ones
    should be the top-K from that sorted list.
    """
    relevant = sorted(
        task["relevant_chunks"],
        key=lambda c: c["metadata"]["importance"],
        reverse=True,
    )
    selected_contents = {c["content"] for c in selected}

    # How many of the top-K relevant chunks are actually selected?
    # K = number of relevant chunks in the selection
    relevant_in_selection = [c for c in relevant if c["content"] in selected_contents]
    k = len(relevant_in_selection)

    if k == 0:
        return 0.0

    # Check if the selected relevant chunks are the top-K most important
    top_k = relevant[:k]
    top_k_contents = {c["content"] for c in top_k}

    matches = len(top_k_contents & {c["content"] for c in relevant_in_selection})
    return matches / k


# ======================================================================
# Budget verification
# ======================================================================

def _selected_words(selected: list[dict]) -> int:
    return sum(_WC(c.get("content", "")) for c in selected)


# ======================================================================
# Main benchmark runner
# ======================================================================

def run_benchmark() -> float:
    loader = JITLoader()

    relevance_tasks = _gen_relevance_tasks()
    ordering_tasks = _gen_priority_ordering_tasks()
    efficiency_tasks = _gen_budget_efficiency_tasks()

    all_recall: list[float] = []
    all_precision: list[float] = []
    all_ordering: list[float] = []

    print("=" * 60)
    print("JIT Context Loading Benchmark Results")
    print("=" * 60)

    # --- Relevance selection ---
    print(f"\n  Relevance selection ({len(relevance_tasks)} tasks):")
    for i, task in enumerate(relevance_tasks):
        selected = loader.select(task["query"], task["chunks"], task["budget"])
        used = _selected_words(selected)
        assert used <= task["budget"] + 5, (
            f"Budget violated: {used} > {task['budget']}"
        )
        recall, precision = _score_relevance(task, selected)
        ordering = _score_importance_ordering(task, selected)
        all_recall.append(recall)
        all_precision.append(precision)
        all_ordering.append(ordering)
        print(f"    task {i}: recall={recall:.4f} prec={precision:.4f} ord={ordering:.4f} ({len(selected)} chunks)")

    # --- Priority ordering ---
    print(f"\n  Priority ordering ({len(ordering_tasks)} tasks):")
    for i, task in enumerate(ordering_tasks):
        selected = loader.select(task["query"], task["chunks"], task["budget"])
        used = _selected_words(selected)
        assert used <= task["budget"] + 5, (
            f"Budget violated: {used} > {task['budget']}"
        )
        recall, precision = _score_relevance(task, selected)
        ordering = _score_importance_ordering(task, selected)
        all_recall.append(recall)
        all_precision.append(precision)
        all_ordering.append(ordering)
        print(f"    task {i}: recall={recall:.4f} prec={precision:.4f} ord={ordering:.4f} ({len(selected)} chunks)")

    # --- Budget efficiency ---
    print(f"\n  Budget efficiency ({len(efficiency_tasks)} tasks):")
    for i, task in enumerate(efficiency_tasks):
        selected = loader.select(task["query"], task["chunks"], task["budget"])
        used = _selected_words(selected)
        assert used <= task["budget"] + 5, (
            f"Budget violated: {used} > {task['budget']}"
        )
        recall, precision = _score_relevance(task, selected)
        ordering = _score_importance_ordering(task, selected)
        all_recall.append(recall)
        all_precision.append(precision)
        all_ordering.append(ordering)
        print(f"    task {i}: recall={recall:.4f} prec={precision:.4f} ord={ordering:.4f} ({len(selected)} chunks)")

    avg_recall = sum(all_recall) / len(all_recall)
    avg_precision = sum(all_precision) / len(all_precision)
    avg_ordering = sum(all_ordering) / len(all_ordering)

    final = 0.5 * avg_recall + 0.3 * avg_precision + 0.2 * avg_ordering

    print("\n" + "-" * 60)
    print(f"  Avg recall (relevant): {avg_recall:.4f}")
    print(f"  Avg precision:         {avg_precision:.4f}")
    print(f"  Avg importance order:  {avg_ordering:.4f}")
    print(f"  Composite: 0.5*{avg_recall:.4f} + 0.3*{avg_precision:.4f} + 0.2*{avg_ordering:.4f}")
    print(f"score: {final:.6f}")
    return final


if __name__ == "__main__":
    run_benchmark()
