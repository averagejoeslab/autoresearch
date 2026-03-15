#!/usr/bin/env python3
"""
Benchmark: Plan Recovery

20 plans with injected failures and ground-truth recovery plans.  Measures:
  - goal_preservation   – does recovery still reach the goal?
  - failure_avoidance   – does recovery avoid the failed step?
  - plan_minimality     – is the recovery plan not bloated?

Fitness = 0.4 * goal_preservation + 0.3 * failure_avoidance + 0.3 * plan_minimality
Prints  : score: X.XXXXXX
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from harness import PlanRecovery  # noqa: E402

# ─────────────────────────────────────────────────────────────────────
# 20 test cases
# Each has:
#   original_plan – list of step dicts
#   failed_step   – index of the step that fails
#   error         – error description
#   goal_actions  – set of actions that MUST appear in recovery
#                   (may include alternative actions NOT in original)
#   banned_action – the failed step's action (must NOT appear)
#   ideal_length  – ideal recovery plan length
# ─────────────────────────────────────────────────────────────────────

def _plan(*specs):
    """Helper: build a plan from (action, desc, deps) tuples."""
    return [
        {"action": a, "description": d, "depends_on": list(deps)}
        for a, d, deps in specs
    ]


CASES: list[dict] = [
    # ── Case 0: simple removal works ─────────────────────────────────
    {
        "original": _plan(
            ("setup_repo", "Create git repository", ()),
            ("write_code", "Implement feature", (0,)),
            ("add_tests", "Write tests", (1,)),
            ("deploy", "Push to production", (2,)),
        ),
        "failed_step": 2,
        "error": "Test framework not installed",
        "goal_actions": {"setup_repo", "write_code", "deploy"},
        "banned_action": "add_tests",
        "ideal_length": 3,
    },
    # ── Case 1: needs alternative – manual test instead of automated ─
    {
        "original": _plan(
            ("gather_data", "Collect data", ()),
            ("clean_data", "Remove nulls", (0,)),
            ("train_model", "Train ML model", (1,)),
            ("evaluate", "Run evaluation", (2,)),
        ),
        "failed_step": 1,
        "error": "Data cleaning script crashed on corrupt rows",
        "goal_actions": {"gather_data", "filter_data", "train_model", "evaluate"},
        "banned_action": "clean_data",
        "ideal_length": 4,
    },
    # ── Case 2: simple removal (auth skip) ───────────────────────────
    {
        "original": _plan(
            ("design_api", "Design endpoints", ()),
            ("implement", "Code the API", (0,)),
            ("add_auth", "Add authentication", (1,)),
            ("write_docs", "Write API docs", (1,)),
            ("deploy", "Deploy service", (2, 3)),
        ),
        "failed_step": 2,
        "error": "Auth provider unavailable",
        "goal_actions": {"design_api", "implement", "write_docs", "deploy"},
        "banned_action": "add_auth",
        "ideal_length": 4,
    },
    # ── Case 3: needs alternative venue ──────────────────────────────
    {
        "original": _plan(
            ("book_venue", "Reserve event space", ()),
            ("send_invites", "Email attendees", (0,)),
            ("arrange_catering", "Order food", (0,)),
            ("run_event", "Execute the event", (1, 2)),
        ),
        "failed_step": 0,
        "error": "Venue double-booked",
        "goal_actions": {"book_alternative_venue", "send_invites", "arrange_catering", "run_event"},
        "banned_action": "book_venue",
        "ideal_length": 4,
    },
    # ── Case 4: simple removal ───────────────────────────────────────
    {
        "original": _plan(
            ("scan_vulns", "Run vulnerability scanner", ()),
            ("manual_test", "Manual penetration test", (0,)),
            ("analyze", "Analyze results", (0, 1)),
            ("write_report", "Document findings", (2,)),
            ("remediate", "Fix issues", (3,)),
        ),
        "failed_step": 1,
        "error": "Pen testing tool license expired",
        "goal_actions": {"scan_vulns", "analyze", "write_report", "remediate"},
        "banned_action": "manual_test",
        "ideal_length": 4,
    },
    # ── Case 5: needs alternative – batch migration ──────────────────
    {
        "original": _plan(
            ("create_schema", "Design database schema", ()),
            ("migrate_data", "Move data to new schema", (0,)),
            ("update_app", "Point app to new DB", (1,)),
            ("verify", "Verify data integrity", (2,)),
        ),
        "failed_step": 1,
        "error": "Migration script OOM on full dataset",
        "goal_actions": {"create_schema", "migrate_data_batched", "update_app", "verify"},
        "banned_action": "migrate_data",
        "ideal_length": 4,
    },
    # ── Case 6: needs alternative – self review ──────────────────────
    {
        "original": _plan(
            ("write_draft", "Write first draft", ()),
            ("peer_review", "Get colleague review", (0,)),
            ("revise", "Incorporate feedback", (1,)),
            ("submit", "Submit paper", (2,)),
        ),
        "failed_step": 1,
        "error": "Reviewer unavailable for two weeks",
        "goal_actions": {"write_draft", "self_review", "revise", "submit"},
        "banned_action": "peer_review",
        "ideal_length": 4,
    },
    # ── Case 7: simple removal ───────────────────────────────────────
    {
        "original": _plan(
            ("choose_framework", "Pick web framework", ()),
            ("scaffold_project", "Generate project", (0,)),
            ("implement_features", "Build features", (1,)),
            ("add_tests", "Write tests", (2,)),
            ("deploy", "Push to hosting", (3,)),
        ),
        "failed_step": 0,
        "error": "Framework evaluation inconclusive",
        "goal_actions": {"scaffold_project", "implement_features", "add_tests", "deploy"},
        "banned_action": "choose_framework",
        "ideal_length": 4,
    },
    # ── Case 8: needs alternative – use docker-compose instead ───────
    {
        "original": _plan(
            ("setup_k8s", "Configure Kubernetes", ()),
            ("create_docker", "Write Dockerfiles", ()),
            ("push_images", "Push to registry", (1,)),
            ("deploy_pods", "Deploy to cluster", (0, 2)),
            ("verify", "Run health checks", (3,)),
        ),
        "failed_step": 0,
        "error": "K8s cluster provisioning failed – quota exceeded",
        "goal_actions": {"setup_docker_compose", "create_docker", "push_images", "deploy_containers", "verify"},
        "banned_action": "setup_k8s",
        "ideal_length": 5,
    },
    # ── Case 9: simple removal ───────────────────────────────────────
    {
        "original": _plan(
            ("collect_reqs", "Gather requirements", ()),
            ("design_ui", "Create mockups", (0,)),
            ("implement_ui", "Build frontend", (1,)),
            ("test_ui", "User testing", (2,)),
            ("launch", "Go live", (3,)),
        ),
        "failed_step": 3,
        "error": "No test users available",
        "goal_actions": {"collect_reqs", "design_ui", "implement_ui", "launch"},
        "banned_action": "test_ui",
        "ideal_length": 4,
    },
    # ── Case 10: needs alternative – use logging instead ─────────────
    {
        "original": _plan(
            ("define_metrics", "Choose KPIs", ()),
            ("instrument_code", "Add telemetry", (0,)),
            ("create_dashboard", "Build Grafana dashboard", (1,)),
            ("set_alerts", "Configure PagerDuty", (2,)),
        ),
        "failed_step": 2,
        "error": "Grafana instance unreachable – disk full",
        "goal_actions": {"define_metrics", "instrument_code", "create_log_dashboard", "set_alerts"},
        "banned_action": "create_dashboard",
        "ideal_length": 4,
    },
    # ── Case 11: simple removal ──────────────────────────────────────
    {
        "original": _plan(
            ("audit_deps", "Check dependencies", ()),
            ("update_deps", "Upgrade packages", (0,)),
            ("run_tests", "Run test suite", (1,)),
            ("fix_breaks", "Fix breaking changes", (2,)),
            ("release", "Publish new version", (3,)),
        ),
        "failed_step": 1,
        "error": "Dependency conflict unresolvable",
        "goal_actions": {"audit_deps", "run_tests", "fix_breaks", "release"},
        "banned_action": "update_deps",
        "ideal_length": 4,
    },
    # ── Case 12: needs alternative – organic social instead of ads ───
    {
        "original": _plan(
            ("define_audience", "Identify target users", ()),
            ("create_content", "Write ad copy", (0,)),
            ("setup_ads", "Configure ad platform", (0,)),
            ("launch_campaign", "Start campaign", (1, 2)),
            ("analyze", "Measure ROI", (3,)),
        ),
        "failed_step": 2,
        "error": "Ad platform API key revoked – account suspended",
        "goal_actions": {"define_audience", "create_content", "setup_organic_social", "launch_campaign", "analyze"},
        "banned_action": "setup_ads",
        "ideal_length": 5,
    },
    # ── Case 13: needs alternative – use API instead ─────────────────
    {
        "original": _plan(
            ("extract", "Pull data from source DB", ()),
            ("transform", "Apply transformations", (0,)),
            ("validate", "Check data quality", (1,)),
            ("load", "Insert into warehouse", (2,)),
        ),
        "failed_step": 0,
        "error": "Source database connection timeout – firewall blocked",
        "goal_actions": {"extract_via_api", "transform", "validate", "load"},
        "banned_action": "extract",
        "ideal_length": 4,
    },
    # ── Case 14: simple removal ──────────────────────────────────────
    {
        "original": _plan(
            ("provision_server", "Spin up VM", ()),
            ("install_deps", "Install packages", (0,)),
            ("configure_nginx", "Set up reverse proxy", (1,)),
            ("deploy_app", "Deploy application", (2,)),
            ("ssl_cert", "Install SSL certificate", (3,)),
        ),
        "failed_step": 2,
        "error": "Nginx config syntax error",
        "goal_actions": {"provision_server", "install_deps", "deploy_app", "ssl_cert"},
        "banned_action": "configure_nginx",
        "ideal_length": 4,
    },
    # ── Case 15: simple removal ──────────────────────────────────────
    {
        "original": _plan(
            ("create_branch", "Make feature branch", ()),
            ("implement", "Write the code", (0,)),
            ("run_linter", "Lint the code", (1,)),
            ("open_pr", "Create pull request", (2,)),
            ("merge", "Merge to main", (3,)),
        ),
        "failed_step": 2,
        "error": "Linter binary not found",
        "goal_actions": {"create_branch", "implement", "open_pr", "merge"},
        "banned_action": "run_linter",
        "ideal_length": 4,
    },
    # ── Case 16: needs alternative – use CSV export ──────────────────
    {
        "original": _plan(
            ("export_mysql", "Dump MySQL data", ()),
            ("convert_types", "Map MySQL types to PG", (0,)),
            ("create_pg_schema", "Create PG tables", (1,)),
            ("import_pg", "Load data into PG", (2,)),
            ("validate", "Compare row counts", (3,)),
        ),
        "failed_step": 0,
        "error": "MySQL export permission denied – read-only replica",
        "goal_actions": {"export_csv", "convert_types", "create_pg_schema", "import_pg", "validate"},
        "banned_action": "export_mysql",
        "ideal_length": 5,
    },
    # ── Case 17: simple removal ──────────────────────────────────────
    {
        "original": _plan(
            ("setup_env", "Configure dev environment", ()),
            ("write_feature", "Implement feature X", (0,)),
            ("write_tests", "Unit tests for X", (1,)),
            ("code_review", "Review by teammate", (2,)),
            ("deploy_staging", "Push to staging", (3,)),
            ("deploy_prod", "Push to production", (4,)),
        ),
        "failed_step": 3,
        "error": "Reviewer on vacation",
        "goal_actions": {"setup_env", "write_feature", "write_tests", "deploy_staging", "deploy_prod"},
        "banned_action": "code_review",
        "ideal_length": 5,
    },
    # ── Case 18: needs alternative – use simulation ──────────────────
    {
        "original": _plan(
            ("research", "Literature review", ()),
            ("formulate", "Formulate hypothesis", (0,)),
            ("experiment", "Run lab experiments", (1,)),
            ("analyze", "Statistical analysis", (2,)),
            ("write_paper", "Write the paper", (3,)),
        ),
        "failed_step": 2,
        "error": "Lab equipment malfunction – parts on backorder",
        "goal_actions": {"research", "formulate", "run_simulation", "analyze", "write_paper"},
        "banned_action": "experiment",
        "ideal_length": 5,
    },
    # ── Case 19: needs alternative – async planning ──────────────────
    {
        "original": _plan(
            ("plan_sprint", "Sprint planning meeting", ()),
            ("assign_tasks", "Distribute work", (0,)),
            ("daily_standup", "Run daily standups", (1,)),
            ("sprint_review", "Demo to stakeholders", (2,)),
            ("retrospective", "Team retro", (3,)),
        ),
        "failed_step": 0,
        "error": "Calendar conflict with planning meeting – all rooms booked",
        "goal_actions": {"async_planning", "assign_tasks", "daily_standup", "sprint_review", "retrospective"},
        "banned_action": "plan_sprint",
        "ideal_length": 5,
    },
]


# ─────────────────────────────────────────────────────────────────────
# Scoring helpers
# ─────────────────────────────────────────────────────────────────────

def _goal_preservation(recovery: list[dict], goal_actions: set[str]) -> float:
    """Fraction of required goal actions present in the recovery plan."""
    if not goal_actions:
        return 1.0
    recovered_actions = {s.get("action", "") for s in recovery}
    return len(goal_actions & recovered_actions) / len(goal_actions)


def _failure_avoidance(recovery: list[dict], banned_action: str) -> float:
    """1.0 if the banned action is absent, 0.0 if present."""
    for s in recovery:
        if s.get("action", "") == banned_action:
            return 0.0
    return 1.0


def _plan_minimality(recovery_length: int, ideal_length: int) -> float:
    """1.0 when lengths match; decays with divergence."""
    if ideal_length == 0:
        return 1.0 if recovery_length == 0 else 0.0
    ratio = recovery_length / ideal_length
    if ratio <= 0:
        return 0.0
    if ratio > 1:
        return max(0.0, 1.0 - (ratio - 1.0))
    return ratio


def run_benchmark() -> float:
    recovery = PlanRecovery()

    gp_scores: list[float] = []
    fa_scores: list[float] = []
    pm_scores: list[float] = []

    for c in CASES:
        result = recovery.recover(c["original"], c["failed_step"], c["error"])

        if not isinstance(result, list):
            gp_scores.append(0.0)
            fa_scores.append(0.0)
            pm_scores.append(0.0)
            continue

        gp_scores.append(_goal_preservation(result, c["goal_actions"]))
        fa_scores.append(_failure_avoidance(result, c["banned_action"]))
        pm_scores.append(_plan_minimality(len(result), c["ideal_length"]))

    avg_gp = sum(gp_scores) / len(gp_scores)
    avg_fa = sum(fa_scores) / len(fa_scores)
    avg_pm = sum(pm_scores) / len(pm_scores)

    score = 0.4 * avg_gp + 0.3 * avg_fa + 0.3 * avg_pm

    print(f"  goal_preservation : {avg_gp:.6f}")
    print(f"  failure_avoidance : {avg_fa:.6f}")
    print(f"  plan_minimality   : {avg_pm:.6f}")
    return score


if __name__ == "__main__":
    final = run_benchmark()
    print(f"score: {final:.6f}")
