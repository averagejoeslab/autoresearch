#!/usr/bin/env python3
"""
Benchmark: Result Aggregation
===============================
15 tasks testing merging of sub-agent outputs into coherent answers.

Fitness = 0.4 * completeness + 0.3 * contradiction_handling + 0.3 * coherence

Run:  python benchmark.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from harness import ResultAggregator  # noqa: E402

# ── Inline test data ────────────────────────────────────────────────
# Each case has sub-agent results and ground-truth expectations.
# key_facts: facts that MUST appear in merged output
# contradictions: pairs of conflicting claims (good aggregator resolves them)
# expected_sections: minimum number of distinct information sections

CASES: list[dict] = [
    # 1. Complementary results, no contradictions
    {
        "task": "Summarize the Q3 financial results",
        "results": [
            {"agent": "analyst", "subtask": "revenue analysis", "output": "Revenue grew 15% year-over-year to $2.3 billion, driven by cloud services.", "confidence": 0.9},
            {"agent": "analyst2", "subtask": "expense analysis", "output": "Operating expenses increased 8% to $1.8 billion due to R&D investment.", "confidence": 0.85},
            {"agent": "writer", "subtask": "profit summary", "output": "Net profit was $320 million, up from $280 million in Q3 last year.", "confidence": 0.95},
        ],
        "key_facts": ["15%", "$2.3 billion", "8%", "$1.8 billion", "$320 million"],
        "contradictions": [],
        "expected_sections": 3,
    },
    # 2. Overlapping results with agreement
    {
        "task": "What programming language should we use for the backend?",
        "results": [
            {"agent": "coder", "subtask": "language evaluation", "output": "Python is recommended for its ecosystem and developer productivity.", "confidence": 0.8},
            {"agent": "architect", "subtask": "architecture review", "output": "Python with FastAPI provides excellent performance and Python has strong library support.", "confidence": 0.9},
        ],
        "key_facts": ["Python", "FastAPI"],
        "contradictions": [],
        "expected_sections": 2,
    },
    # 3. Contradictory results
    {
        "task": "Is the current deployment stable?",
        "results": [
            {"agent": "monitor", "subtask": "uptime check", "output": "Uptime is 99.9% over the last 30 days. No major incidents detected.", "confidence": 0.95},
            {"agent": "tester", "subtask": "load testing", "output": "The system fails under 500 concurrent users. Memory leaks detected in worker processes.", "confidence": 0.88},
            {"agent": "support", "subtask": "user feedback", "output": "Users report intermittent slowdowns during peak hours between 2-4pm.", "confidence": 0.7},
        ],
        "key_facts": ["99.9%", "500 concurrent users", "memory leak", "peak hours"],
        "contradictions": [("stable with 99.9% uptime", "fails under load")],
        "expected_sections": 3,
    },
    # 4. Single result
    {
        "task": "What is the status of the Jenkins pipeline?",
        "results": [
            {"agent": "devops", "subtask": "pipeline check", "output": "The Jenkins pipeline is green. Last build passed 10 minutes ago. All 342 tests pass.", "confidence": 0.99},
        ],
        "key_facts": ["green", "342 tests"],
        "contradictions": [],
        "expected_sections": 1,
    },
    # 5. Empty results
    {
        "task": "Analyze the customer churn data",
        "results": [],
        "key_facts": [],
        "contradictions": [],
        "expected_sections": 0,
    },
    # 6. Low confidence results mixed with high confidence
    {
        "task": "What caused the production outage last night?",
        "results": [
            {"agent": "investigator", "subtask": "log analysis", "output": "Database connection pool exhaustion at 02:15 UTC caused cascading failures.", "confidence": 0.92},
            {"agent": "guesser", "subtask": "initial hypothesis", "output": "Possibly a network issue or DNS failure.", "confidence": 0.2},
            {"agent": "monitor", "subtask": "metrics review", "output": "Connection pool reached 100% at 02:14 UTC, CPU normal, memory normal.", "confidence": 0.88},
        ],
        "key_facts": ["connection pool", "02:15 UTC", "cascading failures"],
        "contradictions": [("database connection pool exhaustion", "network issue or DNS failure")],
        "expected_sections": 3,
    },
    # 7. Deeply contradictory recommendations
    {
        "task": "Should we migrate to microservices?",
        "results": [
            {"agent": "architect_a", "subtask": "pro analysis", "output": "Yes, microservices will improve scalability and team autonomy. Expected 6 month migration.", "confidence": 0.75},
            {"agent": "architect_b", "subtask": "con analysis", "output": "No, the team size of 4 developers makes microservices overhead unjustifiable. Stay monolithic.", "confidence": 0.82},
            {"agent": "researcher", "subtask": "industry research", "output": "Companies under 10 engineers typically see 40% productivity decrease after microservices migration.", "confidence": 0.9},
        ],
        "key_facts": ["scalability", "team autonomy", "4 developers", "40% productivity decrease"],
        "contradictions": [("yes microservices", "no stay monolithic")],
        "expected_sections": 3,
    },
    # 8. Hierarchical/dependent subtasks
    {
        "task": "Create a project plan for the new feature",
        "results": [
            {"agent": "planner", "subtask": "requirements gathering", "output": "Requirements: user login, dashboard view, export to CSV. Estimated 3 weeks total.", "confidence": 0.85},
            {"agent": "designer", "subtask": "UI design", "output": "Dashboard mockups completed. Uses card-based layout with filter sidebar.", "confidence": 0.9},
            {"agent": "estimator", "subtask": "effort estimation", "output": "Backend: 5 days. Frontend: 5 days. Testing: 3 days. Buffer: 2 days. Total: 15 working days.", "confidence": 0.7},
        ],
        "key_facts": ["user login", "dashboard", "CSV", "3 weeks", "card-based layout", "15 working days"],
        "contradictions": [],
        "expected_sections": 3,
    },
    # 9. Duplicate information across agents
    {
        "task": "What version of Python should we use?",
        "results": [
            {"agent": "coder1", "subtask": "compatibility check", "output": "Python 3.11 is recommended. All dependencies are compatible.", "confidence": 0.9},
            {"agent": "coder2", "subtask": "performance benchmark", "output": "Python 3.11 shows 10-60% speedup over 3.10. Python 3.11 is the best choice.", "confidence": 0.85},
            {"agent": "devops", "subtask": "deployment check", "output": "Python 3.11 Docker images are available. Our CI supports 3.11.", "confidence": 0.95},
        ],
        "key_facts": ["Python 3.11", "10-60% speedup", "Docker", "CI"],
        "contradictions": [],
        "expected_sections": 3,
    },
    # 10. Mixed quality outputs
    {
        "task": "Review the security of the authentication system",
        "results": [
            {"agent": "security", "subtask": "vulnerability scan", "output": "Found 2 critical issues: SQL injection in /api/search (CVE-2024-1234) and weak password hashing (MD5).", "confidence": 0.95},
            {"agent": "compliance", "subtask": "compliance check", "output": "GDPR: non-compliant (no data deletion endpoint). SOC2: pending audit.", "confidence": 0.8},
            {"agent": "intern", "subtask": "code review", "output": "The code looks fine to me.", "confidence": 0.3},
        ],
        "key_facts": ["SQL injection", "CVE-2024-1234", "MD5", "GDPR", "non-compliant"],
        "contradictions": [("critical issues found", "code looks fine")],
        "expected_sections": 3,
    },
    # 11. Numerical data aggregation
    {
        "task": "Report website traffic for January",
        "results": [
            {"agent": "analytics", "subtask": "page views", "output": "Total page views: 1.2 million. Unique visitors: 340,000. Bounce rate: 45%.", "confidence": 0.95},
            {"agent": "marketing", "subtask": "campaign performance", "output": "Email campaign drove 80,000 visits. Social media contributed 120,000 visits.", "confidence": 0.85},
            {"agent": "seo", "subtask": "organic traffic", "output": "Organic search: 150,000 visits, up 25% from December. Top keyword: 'cloud hosting'.", "confidence": 0.9},
        ],
        "key_facts": ["1.2 million", "340,000", "45%", "80,000", "120,000", "150,000", "25%"],
        "contradictions": [],
        "expected_sections": 3,
    },
    # 12. Partial failures - some agents returned errors
    {
        "task": "Check all microservices health",
        "results": [
            {"agent": "checker1", "subtask": "auth service", "output": "Auth service healthy. Response time 45ms.", "confidence": 0.95},
            {"agent": "checker2", "subtask": "payment service", "output": "ERROR: Payment service unreachable. Connection timeout after 30s.", "confidence": 0.99},
            {"agent": "checker3", "subtask": "notification service", "output": "Notification service degraded. Queue backlog of 15,000 messages.", "confidence": 0.85},
        ],
        "key_facts": ["healthy", "45ms", "unreachable", "timeout", "degraded", "15,000"],
        "contradictions": [],
        "expected_sections": 3,
    },
    # 13. Long-form results needing summarization
    {
        "task": "Explain how our caching system works",
        "results": [
            {"agent": "backend", "subtask": "cache architecture", "output": "We use a two-tier caching strategy. L1 is in-process LRU cache with 1000 entry limit and 5 minute TTL. L2 is Redis cluster with 3 nodes, 64GB total memory, and 1 hour TTL. Cache invalidation uses pub/sub pattern.", "confidence": 0.9},
            {"agent": "frontend", "subtask": "client caching", "output": "Browser caching uses service workers for static assets with 24 hour cache. API responses cached with ETag validation. CDN caches at edge locations with 30 minute TTL.", "confidence": 0.85},
        ],
        "key_facts": ["two-tier", "LRU", "Redis", "pub/sub", "service workers", "ETag", "CDN"],
        "contradictions": [],
        "expected_sections": 2,
    },
    # 14. Contradictory metrics
    {
        "task": "Is user engagement improving?",
        "results": [
            {"agent": "product", "subtask": "engagement metrics", "output": "Daily active users up 20%. Average session duration increased from 4 to 6 minutes.", "confidence": 0.85},
            {"agent": "retention", "subtask": "retention analysis", "output": "7-day retention dropped from 45% to 38%. Churn rate increased by 15%.", "confidence": 0.9},
            {"agent": "support", "subtask": "satisfaction survey", "output": "NPS score improved from 32 to 41. Customer satisfaction at 78%.", "confidence": 0.8},
        ],
        "key_facts": ["20%", "4 to 6 minutes", "45% to 38%", "churn", "NPS", "41"],
        "contradictions": [("engagement improving", "retention dropping")],
        "expected_sections": 3,
    },
    # 15. All agents agree perfectly
    {
        "task": "What time zone should the server use?",
        "results": [
            {"agent": "devops", "subtask": "server config", "output": "UTC is the standard for server timestamps. All logs should use UTC.", "confidence": 0.95},
            {"agent": "backend", "subtask": "application config", "output": "Application stores all timestamps in UTC. Conversion to local time happens at display layer.", "confidence": 0.9},
            {"agent": "dba", "subtask": "database config", "output": "PostgreSQL timezone is set to UTC. All timestamp columns use TIMESTAMPTZ.", "confidence": 0.92},
        ],
        "key_facts": ["UTC", "TIMESTAMPTZ"],
        "contradictions": [],
        "expected_sections": 3,
    },
]


def _score_completeness(merged: str, key_facts: list[str]) -> float:
    """What fraction of key facts appear in the merged output?"""
    if not key_facts:
        return 1.0 if not merged else 0.5  # empty expected, non-empty is okay-ish
    merged_lower = merged.lower()
    found = sum(1 for f in key_facts if f.lower() in merged_lower)
    return found / len(key_facts)


def _score_contradiction_handling(
    merged: str,
    contradictions: list[tuple[str, str]],
    results: list[dict],
) -> float:
    """Check whether contradictions are acknowledged or resolved.

    A good aggregator either:
    - Mentions both sides (shows awareness)
    - Picks the higher-confidence side and notes the disagreement
    - Uses hedging language ("however", "but", "on the other hand")

    Baseline (concatenation) will include both sides but won't flag them,
    so it gets partial credit.
    """
    if not contradictions:
        return 1.0  # no contradictions to handle

    merged_lower = merged.lower()
    score = 0.0
    for side_a, side_b in contradictions:
        has_a = any(w in merged_lower for w in side_a.lower().split())
        has_b = any(w in merged_lower for w in side_b.lower().split())

        if has_a and has_b:
            # Both sides present - partial credit
            # Full credit if hedging language is used
            hedging = ["however", "but", "on the other hand", "conflicting",
                       "disagree", "contrast", "although", "while", "note that",
                       "caveat", "warning"]
            has_hedge = any(h in merged_lower for h in hedging)
            score += 1.0 if has_hedge else 0.5
        elif has_a or has_b:
            score += 0.25  # only one side mentioned
        # else: 0

    return score / len(contradictions)


def _score_coherence(merged: str, expected_sections: int) -> float:
    """Check structural coherence of the merged output."""
    if expected_sections == 0:
        return 1.0 if not merged else 0.5

    if not merged:
        return 0.0

    score = 0.0

    # 1. Has some structure (paragraphs/sections) - 0.4 weight
    paragraph_count = len([p for p in merged.split("\n\n") if p.strip()])
    if paragraph_count >= expected_sections:
        score += 0.4
    elif paragraph_count > 0:
        score += 0.4 * (paragraph_count / expected_sections)

    # 2. Not just raw dumps - has formatting/labels - 0.3 weight
    has_labels = "[" in merged or ":" in merged or "#" in merged or "- " in merged
    score += 0.3 if has_labels else 0.0

    # 3. Reasonable length - not too short, not absurdly long - 0.3 weight
    words = len(merged.split())
    if 10 <= words <= 2000:
        score += 0.3
    elif words > 0:
        score += 0.15

    return score


def run_benchmark() -> float:
    aggregator = ResultAggregator()

    completeness_scores: list[float] = []
    contradiction_scores: list[float] = []
    coherence_scores: list[float] = []

    for case in CASES:
        task = case["task"]
        results = case["results"]
        key_facts = case["key_facts"]
        contradictions = case["contradictions"]
        expected_sections = case["expected_sections"]

        try:
            merged = aggregator.aggregate(results, task)
        except Exception:
            merged = ""

        comp = _score_completeness(merged, key_facts)
        cont = _score_contradiction_handling(merged, contradictions, results)
        cohr = _score_coherence(merged, expected_sections)

        completeness_scores.append(comp)
        contradiction_scores.append(cont)
        coherence_scores.append(cohr)

    avg_completeness = sum(completeness_scores) / len(completeness_scores)
    avg_contradiction = sum(contradiction_scores) / len(contradiction_scores)
    avg_coherence = sum(coherence_scores) / len(coherence_scores)

    fitness = (
        0.4 * avg_completeness
        + 0.3 * avg_contradiction
        + 0.3 * avg_coherence
    )

    print(f"completeness:           {avg_completeness:.4f}")
    print(f"contradiction_handling: {avg_contradiction:.4f}")
    print(f"coherence:              {avg_coherence:.4f}")
    print(f"score: {fitness:.6f}")
    return fitness


if __name__ == "__main__":
    run_benchmark()
