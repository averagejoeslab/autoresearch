#!/usr/bin/env python3
"""
Benchmark: Task Decomposition

20 tasks with reference decompositions.  Measures:
  - step coverage   (are all necessary steps present?)
  - ordering        (are dependencies respected?)
  - granularity     (step count close to reference?)

Fitness = 0.4 * coverage + 0.3 * ordering + 0.3 * granularity
Prints  : score: X.XXXXXX
"""

from __future__ import annotations

import os
import sys

# ── allow imports from project root ──────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from harness import TaskDecomposer  # noqa: E402

# ─────────────────────────────────────────────────────────────────────
# Reference tasks.  Each has:
#   task            – natural-language task string
#   context         – dict passed to decompose()
#   ref_steps       – list of expected step actions (order matters)
#   ref_step_count  – ideal number of steps
# ─────────────────────────────────────────────────────────────────────
TASKS: list[dict] = [
    {
        "task": "Deploy a web application to production",
        "context": {"domain": "devops"},
        "ref_steps": ["setup_repo", "write_code", "add_tests", "configure_ci", "deploy"],
        "ref_step_count": 5,
    },
    {
        "task": "Write a research paper on climate change impacts",
        "context": {"domain": "academic"},
        "ref_steps": ["literature_review", "formulate_thesis", "gather_data", "analyze_data", "write_draft", "revise", "submit"],
        "ref_step_count": 7,
    },
    {
        "task": "Organize a team offsite event",
        "context": {"domain": "management"},
        "ref_steps": ["set_budget", "choose_venue", "plan_agenda", "send_invites", "arrange_logistics", "run_event", "collect_feedback"],
        "ref_step_count": 7,
    },
    {
        "task": "Migrate a database from MySQL to PostgreSQL",
        "context": {"domain": "engineering"},
        "ref_steps": ["audit_schema", "map_types", "export_data", "create_pg_schema", "import_data", "validate", "switch_connections"],
        "ref_step_count": 7,
    },
    {
        "task": "Launch a new product feature",
        "context": {"domain": "product"},
        "ref_steps": ["define_requirements", "design_ux", "implement", "test", "write_docs", "deploy", "announce"],
        "ref_step_count": 7,
    },
    {
        "task": "Set up a CI/CD pipeline",
        "context": {"domain": "devops"},
        "ref_steps": ["choose_tool", "configure_build", "add_tests", "configure_deploy", "verify_pipeline"],
        "ref_step_count": 5,
    },
    {
        "task": "Conduct a code review",
        "context": {"domain": "engineering"},
        "ref_steps": ["read_description", "check_tests", "review_code", "leave_comments", "approve_or_request_changes"],
        "ref_step_count": 5,
    },
    {
        "task": "Create a machine learning model for spam detection",
        "context": {"domain": "ml"},
        "ref_steps": ["collect_data", "preprocess", "feature_engineering", "train_model", "evaluate", "tune_hyperparams", "deploy_model"],
        "ref_step_count": 7,
    },
    {
        "task": "Plan a marketing campaign",
        "context": {"domain": "marketing"},
        "ref_steps": ["define_audience", "set_goals", "choose_channels", "create_content", "launch", "monitor", "analyze_results"],
        "ref_step_count": 7,
    },
    {
        "task": "Onboard a new employee",
        "context": {"domain": "hr"},
        "ref_steps": ["prepare_workspace", "setup_accounts", "assign_buddy", "orientation", "training", "first_project"],
        "ref_step_count": 6,
    },
    {
        "task": "Debug a production outage",
        "context": {"domain": "sre"},
        "ref_steps": ["acknowledge_incident", "assess_impact", "identify_cause", "apply_fix", "verify_recovery", "write_postmortem"],
        "ref_step_count": 6,
    },
    {
        "task": "Build a REST API",
        "context": {"domain": "backend"},
        "ref_steps": ["design_endpoints", "setup_project", "implement_routes", "add_auth", "write_tests", "document_api", "deploy"],
        "ref_step_count": 7,
    },
    {
        "task": "Redesign a company website",
        "context": {"domain": "design"},
        "ref_steps": ["audit_current_site", "gather_requirements", "create_wireframes", "design_mockups", "implement", "test", "launch"],
        "ref_step_count": 7,
    },
    {
        "task": "Prepare a quarterly financial report",
        "context": {"domain": "finance"},
        "ref_steps": ["gather_data", "reconcile_accounts", "compute_metrics", "draft_report", "review", "submit"],
        "ref_step_count": 6,
    },
    {
        "task": "Set up monitoring and alerting for a service",
        "context": {"domain": "sre"},
        "ref_steps": ["define_slis", "choose_tools", "instrument_code", "configure_dashboards", "set_alerts", "test_alerts"],
        "ref_step_count": 6,
    },
    {
        "task": "Write unit tests for a legacy codebase",
        "context": {"domain": "engineering"},
        "ref_steps": ["identify_modules", "setup_framework", "write_tests", "measure_coverage", "fix_failures"],
        "ref_step_count": 5,
    },
    {
        "task": "Implement user authentication with OAuth2",
        "context": {"domain": "security"},
        "ref_steps": ["register_provider", "configure_client", "implement_flow", "handle_tokens", "add_session_management", "test"],
        "ref_step_count": 6,
    },
    {
        "task": "Perform a security audit on a web application",
        "context": {"domain": "security"},
        "ref_steps": ["scope_audit", "scan_vulnerabilities", "manual_testing", "analyze_findings", "write_report", "remediate"],
        "ref_step_count": 6,
    },
    {
        "task": "Move a monolithic application to microservices",
        "context": {"domain": "architecture"},
        "ref_steps": ["identify_boundaries", "define_services", "setup_communication", "extract_service", "test_integration", "migrate_data", "decommission_monolith"],
        "ref_step_count": 7,
    },
    {
        "task": "Create a data pipeline for analytics",
        "context": {"domain": "data_engineering"},
        "ref_steps": ["identify_sources", "design_schema", "build_ingestion", "transform_data", "load_warehouse", "schedule", "validate"],
        "ref_step_count": 7,
    },
]


# ─────────────────────────────────────────────────────────────────────
# Scoring helpers
# ─────────────────────────────────────────────────────────────────────

def _normalise(action: str) -> str:
    """Lower-case, strip, collapse whitespace."""
    return " ".join(action.strip().lower().split())


def _keyword_overlap(produced_actions: list[str], ref_actions: list[str]) -> float:
    """Fraction of reference action *keywords* that appear in produced actions.

    Uses bag-of-words overlap rather than exact string match so that
    reasonable paraphrases still get partial credit.
    """
    ref_words: set[str] = set()
    for a in ref_actions:
        ref_words.update(_normalise(a).replace("_", " ").split())
    if not ref_words:
        return 1.0

    produced_words: set[str] = set()
    for a in produced_actions:
        produced_words.update(_normalise(a).replace("_", " ").split())
        # Also include description words for broader matching
    return len(ref_words & produced_words) / len(ref_words)


def _ordering_score(steps: list[dict]) -> float:
    """1.0 if every dependency index is < the step's own index, else penalise."""
    if not steps:
        return 0.0
    violations = 0
    total_deps = 0
    for i, step in enumerate(steps):
        for dep in step.get("depends_on", []):
            total_deps += 1
            if dep >= i:
                violations += 1
    if total_deps == 0:
        # No dependencies declared at all — partial credit
        return 0.5
    return 1.0 - violations / total_deps


def _granularity_score(produced_count: int, ref_count: int) -> float:
    """1.0 when counts match, decays as they diverge."""
    if ref_count == 0:
        return 1.0 if produced_count == 0 else 0.0
    ratio = produced_count / ref_count
    # Perfect at ratio=1, drops linearly toward 0 or +inf
    if ratio <= 0:
        return 0.0
    if ratio > 1:
        return max(0.0, 1.0 - (ratio - 1.0))
    return ratio  # ratio in (0, 1]


# ─────────────────────────────────────────────────────────────────────
# Main benchmark
# ─────────────────────────────────────────────────────────────────────

def run_benchmark() -> float:
    decomposer = TaskDecomposer()

    coverage_scores: list[float] = []
    ordering_scores: list[float] = []
    granularity_scores: list[float] = []

    for t in TASKS:
        steps = decomposer.decompose(t["task"], t.get("context"))

        # --- validate structure ---
        if not isinstance(steps, list) or not steps:
            coverage_scores.append(0.0)
            ordering_scores.append(0.0)
            granularity_scores.append(0.0)
            continue

        actions = [s.get("action", "") for s in steps]
        # Also fold in descriptions for keyword matching
        action_and_desc = [
            f'{s.get("action", "")} {s.get("description", "")}'
            for s in steps
        ]

        cov = _keyword_overlap(action_and_desc, t["ref_steps"])
        coverage_scores.append(cov)

        ordering_scores.append(_ordering_score(steps))

        gran = _granularity_score(len(steps), t["ref_step_count"])
        granularity_scores.append(gran)

    avg_cov = sum(coverage_scores) / len(coverage_scores)
    avg_ord = sum(ordering_scores) / len(ordering_scores)
    avg_gran = sum(granularity_scores) / len(granularity_scores)

    score = 0.4 * avg_cov + 0.3 * avg_ord + 0.3 * avg_gran

    print(f"  step_coverage : {avg_cov:.6f}")
    print(f"  ordering      : {avg_ord:.6f}")
    print(f"  granularity   : {avg_gran:.6f}")
    return score


if __name__ == "__main__":
    final = run_benchmark()
    print(f"score: {final:.6f}")
