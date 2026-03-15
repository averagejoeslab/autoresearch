#!/usr/bin/env python3
"""
Benchmark: Context Filter
===========================
20 tasks testing context selection for sub-agent subtasks.

Fitness = 0.4 * recall + 0.3 * precision + 0.3 * budget_efficiency

Run:  python benchmark.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from harness import ContextFilter  # noqa: E402

# ── Build a pool of 50 context items ────────────────────────────────

CONTEXT_POOL: list[dict] = [
    {"id": 0,  "content": "The user wants to build a web application using Flask",               "type": "instruction", "timestamp": 100},
    {"id": 1,  "content": "Python 3.11 is required for this project",                            "type": "fact",        "timestamp": 101},
    {"id": 2,  "content": "The database is PostgreSQL 15 running on port 5432",                  "type": "fact",        "timestamp": 102},
    {"id": 3,  "content": "Authentication should use JWT tokens",                                 "type": "instruction", "timestamp": 103},
    {"id": 4,  "content": "The frontend uses React 18 with TypeScript",                          "type": "fact",        "timestamp": 104},
    {"id": 5,  "content": "API responses should follow REST conventions",                         "type": "instruction", "timestamp": 105},
    {"id": 6,  "content": "Unit tests must use pytest framework",                                 "type": "instruction", "timestamp": 106},
    {"id": 7,  "content": "The deployment target is AWS ECS",                                     "type": "fact",        "timestamp": 107},
    {"id": 8,  "content": "Error logging goes to CloudWatch",                                     "type": "fact",        "timestamp": 108},
    {"id": 9,  "content": "The team uses git flow branching strategy",                            "type": "observation", "timestamp": 109},
    {"id": 10, "content": "Code must pass flake8 and mypy checks",                                "type": "instruction", "timestamp": 110},
    {"id": 11, "content": "The user's name is Alice and she works at TechCorp",                   "type": "fact",        "timestamp": 111},
    {"id": 12, "content": "Previous attempt to set up Redis caching failed",                      "type": "observation", "timestamp": 112},
    {"id": 13, "content": "The project deadline is end of Q2",                                    "type": "fact",        "timestamp": 113},
    {"id": 14, "content": "Performance requirement: API responses under 200ms",                   "type": "instruction", "timestamp": 114},
    {"id": 15, "content": "The application needs to support 1000 concurrent users",               "type": "instruction", "timestamp": 115},
    {"id": 16, "content": "Data must be encrypted at rest and in transit",                        "type": "instruction", "timestamp": 116},
    {"id": 17, "content": "CSS styling follows BEM naming convention",                            "type": "observation", "timestamp": 117},
    {"id": 18, "content": "The search feature uses Elasticsearch",                                "type": "fact",        "timestamp": 118},
    {"id": 19, "content": "File uploads go to S3 bucket named techcorp-uploads",                  "type": "fact",        "timestamp": 119},
    {"id": 20, "content": "Rate limiting is set to 100 requests per minute per user",             "type": "instruction", "timestamp": 120},
    {"id": 21, "content": "The API documentation is generated with Swagger/OpenAPI",              "type": "fact",        "timestamp": 121},
    {"id": 22, "content": "Background jobs run on Celery with RabbitMQ broker",                   "type": "fact",        "timestamp": 122},
    {"id": 23, "content": "The mobile app communicates via GraphQL endpoint",                     "type": "fact",        "timestamp": 123},
    {"id": 24, "content": "User sessions expire after 30 minutes of inactivity",                  "type": "instruction", "timestamp": 124},
    {"id": 25, "content": "The CI/CD pipeline uses GitHub Actions",                               "type": "fact",        "timestamp": 125},
    {"id": 26, "content": "Docker images are stored in ECR registry",                             "type": "fact",        "timestamp": 126},
    {"id": 27, "content": "The notification system uses SNS and SES for emails",                  "type": "fact",        "timestamp": 127},
    {"id": 28, "content": "Payment processing integrates with Stripe API",                        "type": "fact",        "timestamp": 128},
    {"id": 29, "content": "Monitoring dashboards are in Grafana with Prometheus metrics",         "type": "fact",        "timestamp": 129},
    {"id": 30, "content": "The admin panel is built with Flask-Admin",                            "type": "fact",        "timestamp": 130},
    {"id": 31, "content": "Data validation uses Pydantic models",                                 "type": "fact",        "timestamp": 131},
    {"id": 32, "content": "The test database is SQLite for speed",                                "type": "fact",        "timestamp": 132},
    {"id": 33, "content": "Environment variables are managed with python-dotenv",                 "type": "observation", "timestamp": 133},
    {"id": 34, "content": "CORS is configured to allow requests from techcorp.com",               "type": "instruction", "timestamp": 134},
    {"id": 35, "content": "The API versioning strategy is URL-based (/v1/, /v2/)",                "type": "instruction", "timestamp": 135},
    {"id": 36, "content": "Websocket connections use Flask-SocketIO",                             "type": "fact",        "timestamp": 136},
    {"id": 37, "content": "The recommendation engine uses collaborative filtering",              "type": "fact",        "timestamp": 137},
    {"id": 38, "content": "User avatars are resized to 128x128 before upload",                    "type": "instruction", "timestamp": 138},
    {"id": 39, "content": "The search index is rebuilt nightly via cron job",                      "type": "observation", "timestamp": 139},
    {"id": 40, "content": "All timestamps are stored in UTC",                                     "type": "instruction", "timestamp": 140},
    {"id": 41, "content": "The user requested dark mode support",                                 "type": "instruction", "timestamp": 141},
    {"id": 42, "content": "Database migrations use Alembic",                                      "type": "fact",        "timestamp": 142},
    {"id": 43, "content": "The project uses poetry for dependency management",                    "type": "observation", "timestamp": 143},
    {"id": 44, "content": "Caching layer uses Redis with 5 minute TTL",                           "type": "fact",        "timestamp": 144},
    {"id": 45, "content": "Health check endpoint is at /api/health",                              "type": "fact",        "timestamp": 145},
    {"id": 46, "content": "The export feature generates CSV and PDF reports",                     "type": "fact",        "timestamp": 146},
    {"id": 47, "content": "Access control uses role-based permissions (admin, editor, viewer)",   "type": "instruction", "timestamp": 147},
    {"id": 48, "content": "The audit log tracks all data modifications",                          "type": "instruction", "timestamp": 148},
    {"id": 49, "content": "Two-factor authentication is optional for now",                        "type": "instruction", "timestamp": 149},
]

# ── 20 subtasks with ground-truth relevant item IDs ─────────────────

CASES: list[dict] = [
    {
        "subtask": "Set up JWT authentication for the Flask API",
        "budget": 10,
        "relevant_ids": {0, 3, 5, 24, 47, 49, 16, 1},
    },
    {
        "subtask": "Write pytest unit tests for the user model",
        "budget": 10,
        "relevant_ids": {6, 1, 2, 31, 32, 10},
    },
    {
        "subtask": "Configure the PostgreSQL database connection",
        "budget": 10,
        "relevant_ids": {2, 1, 42, 33, 31},
    },
    {
        "subtask": "Deploy the application to AWS ECS",
        "budget": 10,
        "relevant_ids": {7, 8, 25, 26, 45, 29},
    },
    {
        "subtask": "Build the React frontend login page",
        "budget": 10,
        "relevant_ids": {4, 3, 17, 41, 34, 24},
    },
    {
        "subtask": "Set up the CI/CD pipeline with GitHub Actions",
        "budget": 10,
        "relevant_ids": {25, 10, 26, 7, 6, 9},
    },
    {
        "subtask": "Implement Stripe payment processing",
        "budget": 10,
        "relevant_ids": {28, 16, 5, 48, 14},
    },
    {
        "subtask": "Configure Redis caching for API responses",
        "budget": 10,
        "relevant_ids": {44, 12, 14, 15, 20},
    },
    {
        "subtask": "Build the Elasticsearch search feature",
        "budget": 10,
        "relevant_ids": {18, 39, 14, 15, 5},
    },
    {
        "subtask": "Set up monitoring with Grafana and Prometheus",
        "budget": 10,
        "relevant_ids": {29, 8, 45, 7, 14},
    },
    {
        "subtask": "Implement file upload to S3",
        "budget": 10,
        "relevant_ids": {19, 38, 7, 16, 5},
    },
    {
        "subtask": "Build the notification system with email support",
        "budget": 10,
        "relevant_ids": {27, 22, 16, 5},
    },
    {
        "subtask": "Create the admin panel",
        "budget": 10,
        "relevant_ids": {30, 47, 48, 0, 2},
    },
    {
        "subtask": "Set up database migrations with Alembic",
        "budget": 10,
        "relevant_ids": {42, 2, 1, 31, 33},
    },
    {
        "subtask": "Implement rate limiting for the API",
        "budget": 10,
        "relevant_ids": {20, 14, 15, 5, 44},
    },
    {
        "subtask": "Generate API documentation with Swagger",
        "budget": 10,
        "relevant_ids": {21, 5, 35, 0, 45},
    },
    {
        "subtask": "Configure websocket real-time features",
        "budget": 10,
        "relevant_ids": {36, 15, 0, 5, 14},
    },
    {
        "subtask": "Build the CSV and PDF export feature",
        "budget": 10,
        "relevant_ids": {46, 2, 31, 47, 5},
    },
    {
        "subtask": "Add dark mode to the frontend",
        "budget": 10,
        "relevant_ids": {41, 4, 17, 34},
    },
    {
        "subtask": "Implement role-based access control",
        "budget": 10,
        "relevant_ids": {47, 3, 24, 48, 49, 16},
    },
]


def run_benchmark() -> float:
    cf = ContextFilter()

    recall_scores: list[float] = []
    precision_scores: list[float] = []
    budget_scores: list[float] = []

    for case in CASES:
        subtask = case["subtask"]
        budget = case["budget"]
        relevant_ids = case["relevant_ids"]

        try:
            selected = cf.filter_for_subtask(CONTEXT_POOL, subtask, budget)
        except Exception:
            selected = []

        selected_ids = {item["id"] for item in selected}

        # Recall: what fraction of relevant items were selected?
        if relevant_ids:
            recall = len(selected_ids & relevant_ids) / len(relevant_ids)
        else:
            recall = 1.0

        # Precision: what fraction of selected items are relevant?
        if selected_ids:
            precision = len(selected_ids & relevant_ids) / len(selected_ids)
        else:
            precision = 0.0

        # Budget efficiency: did we use the budget well?
        if budget > 0:
            # Penalize both under-use and over-use
            usage_ratio = len(selected) / budget
            if usage_ratio > 1.0:
                budget_eff = max(0.0, 1.0 - (usage_ratio - 1.0))  # over budget
            else:
                # reward using the budget if there are enough relevant items
                ideal_usage = min(len(relevant_ids), budget)
                if ideal_usage > 0:
                    budget_eff = min(len(selected), ideal_usage) / ideal_usage
                else:
                    budget_eff = 1.0
        else:
            budget_eff = 1.0 if not selected else 0.0

        recall_scores.append(recall)
        precision_scores.append(precision)
        budget_scores.append(budget_eff)

    avg_recall = sum(recall_scores) / len(recall_scores)
    avg_precision = sum(precision_scores) / len(precision_scores)
    avg_budget = sum(budget_scores) / len(budget_scores)

    fitness = 0.4 * avg_recall + 0.3 * avg_precision + 0.3 * avg_budget

    print(f"recall:            {avg_recall:.4f}")
    print(f"precision:         {avg_precision:.4f}")
    print(f"budget_efficiency: {avg_budget:.4f}")
    print(f"score: {fitness:.6f}")
    return fitness


if __name__ == "__main__":
    run_benchmark()
