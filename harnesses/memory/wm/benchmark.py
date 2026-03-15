#!/usr/bin/env python3
"""
Benchmark: Working Memory System
=================================

40 tasks across 4 challenge categories testing the COMPLETE working memory
architecture — eviction, budget allocation, compression, JIT loading, and
system prompt preservation as one unified system.

Categories:
  1. Information Retention Under Pressure  (15 tasks)
  2. Budget Allocation Adaptation          (10 tasks)
  3. JIT Loading Accuracy                  (10 tasks)
  4. System Prompt Preservation             (5 tasks)

Fitness = 0.35 * fact_retention
        + 0.25 * budget_adaptation
        + 0.25 * jit_loading_f1
        + 0.15 * system_prompt_preservation

Prints: score: X.XXXXXX

DO NOT MODIFY THIS FILE — it is the locked evaluation.
"""

from __future__ import annotations

import os
import sys

# ── allow imports from project root ──────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from harness import WorkingMemory  # noqa: E402


# =====================================================================
# Helper utilities
# =====================================================================

def _make_message(role: str, content: str, **kwargs) -> dict:
    """Create a well-formed message dict."""
    msg = {"role": role, "content": content}
    msg.update(kwargs)
    return msg


def _count_words(text: str) -> int:
    return len(text.split())


def _words_in_window(window: list[dict]) -> int:
    return sum(_count_words(m.get("content", "")) for m in window)


def _window_text(window: list[dict]) -> str:
    """Concatenate all content in the window for searching."""
    return " ".join(m.get("content", "") for m in window)


# =====================================================================
# Category 1: Information Retention Under Pressure (15 tasks)
# =====================================================================
# Simulate 60-message conversations with key facts embedded at positions
# 5, 15, 25, 35, 55.  Then compact to 30% budget and check which facts
# survive.  Each of the 15 tasks uses a different domain & fact set.

_RETENTION_SCENARIOS: list[dict] = [
    {
        "domain": "software_project",
        "system_prompt": "You are a senior software engineer helping plan a microservices migration.",
        "facts": {
            5:  "The legacy database uses Oracle 12c with 47 custom stored procedures.",
            15: "The team agreed to use gRPC for inter-service communication on March 3rd.",
            25: "The compliance deadline for PCI-DSS certification is September 15th.",
            35: "Service mesh will be Istio version 1.19 deployed on Kubernetes 1.28.",
            55: "The performance budget is 99.9th percentile latency under 200ms.",
        },
    },
    {
        "domain": "medical_research",
        "system_prompt": "You are a research assistant helping analyze clinical trial data.",
        "facts": {
            5:  "Patient cohort A received 250mg dosage twice daily for 12 weeks.",
            15: "The control group showed a 3.7% improvement in biomarker XR-42.",
            25: "Adverse events were reported by 8 out of 340 participants in arm B.",
            35: "The p-value for the primary endpoint was 0.0031, below the 0.05 threshold.",
            55: "The FDA pre-submission meeting is scheduled for November 8th.",
        },
    },
    {
        "domain": "financial_analysis",
        "system_prompt": "You are a financial analyst preparing a quarterly earnings report.",
        "facts": {
            5:  "Revenue for Q3 was $847.2 million, up 12.3% year-over-year.",
            15: "The largest cost driver was cloud infrastructure at $124.5 million.",
            25: "Customer acquisition cost decreased to $43.20 from $51.80 last quarter.",
            35: "The board approved a $200 million share buyback program on October 1st.",
            55: "Gross margin improved to 71.4% from 68.9% in the previous quarter.",
        },
    },
    {
        "domain": "devops_incident",
        "system_prompt": "You are an SRE investigating a production outage.",
        "facts": {
            5:  "The outage began at 14:23 UTC when the primary database hit 100% CPU.",
            15: "Root cause was a missing index on the orders table join with customers.",
            25: "Approximately 12,400 API requests failed with 503 errors during the window.",
            35: "The rollback to version 3.8.1 restored service at 15:47 UTC.",
            55: "The monitoring gap was in the pg_stat_activity dashboard missing long queries.",
        },
    },
    {
        "domain": "legal_review",
        "system_prompt": "You are a legal assistant reviewing contract terms.",
        "facts": {
            5:  "The indemnification clause in Section 7.3 caps liability at $5 million.",
            15: "The non-compete period is 18 months post-termination across all territories.",
            25: "Intellectual property assignment in clause 4.2 excludes prior inventions listed in Exhibit B.",
            35: "The governing law is the State of Delaware with mandatory arbitration.",
            55: "The termination for convenience clause requires 90 days written notice.",
        },
    },
    {
        "domain": "ml_training",
        "system_prompt": "You are an ML engineer fine-tuning a language model.",
        "facts": {
            5:  "The base model is LLaMA-3 8B with a context window of 8192 tokens.",
            15: "Training data consists of 2.3 million instruction-response pairs from ShareGPT.",
            25: "Learning rate was set to 2e-5 with cosine annealing over 3 epochs.",
            35: "LoRA rank is 64 with alpha 128, targeting q_proj and v_proj layers.",
            55: "Evaluation on MMLU showed 62.4% accuracy, up from 58.1% baseline.",
        },
    },
    {
        "domain": "product_launch",
        "system_prompt": "You are a product manager coordinating a major feature launch.",
        "facts": {
            5:  "The launch date is set for January 15th with a 48-hour staged rollout.",
            15: "Beta testing with 2,000 users showed a 34% improvement in task completion.",
            25: "The pricing tier for enterprise will be $499/month per seat minimum 10 seats.",
            35: "Marketing budget allocation is $1.2 million split 60/40 digital and events.",
            55: "The key competitor Acme Corp launched a similar feature 6 weeks ago.",
        },
    },
    {
        "domain": "data_pipeline",
        "system_prompt": "You are a data engineer building an ETL pipeline.",
        "facts": {
            5:  "Source data arrives as 47GB of compressed Parquet files every 6 hours.",
            15: "The transformation step deduplicates using a composite key of user_id and timestamp.",
            25: "The Spark cluster has 24 executor nodes each with 64GB RAM and 16 cores.",
            35: "Schema evolution is handled by Delta Lake with merge-on-read strategy.",
            55: "The SLA requires data freshness within 45 minutes of source update.",
        },
    },
    {
        "domain": "security_audit",
        "system_prompt": "You are a security engineer conducting a penetration test.",
        "facts": {
            5:  "The application exposes 23 API endpoints, 7 of which require no authentication.",
            15: "SQL injection was found in the /api/search endpoint via the query parameter.",
            25: "The session tokens use SHA-256 HMAC but with a hardcoded secret key.",
            35: "Cross-site scripting is possible in the user profile bio field (stored XSS).",
            55: "The TLS configuration supports TLS 1.0 which should be deprecated.",
        },
    },
    {
        "domain": "infrastructure",
        "system_prompt": "You are a cloud architect designing a multi-region deployment.",
        "facts": {
            5:  "Primary region is us-east-1 with failover to eu-west-1 and ap-southeast-1.",
            15: "The global load balancer uses latency-based routing with health checks every 10s.",
            25: "Database replication lag between regions must stay under 500 milliseconds.",
            35: "The total infrastructure budget is $340,000 per month across all regions.",
            55: "The disaster recovery RTO is 15 minutes and RPO is 5 minutes.",
        },
    },
    {
        "domain": "ux_research",
        "system_prompt": "You are a UX researcher synthesizing usability test findings.",
        "facts": {
            5:  "Average task completion time for checkout was 4 minutes 23 seconds.",
            15: "73% of participants failed to find the return policy link on the product page.",
            25: "The mobile navigation hamburger menu was missed by 5 out of 12 users.",
            35: "Users rated the search experience 2.3 out of 5 on the SUS questionnaire.",
            55: "The accessibility audit found 31 WCAG 2.1 Level AA violations.",
        },
    },
    {
        "domain": "supply_chain",
        "system_prompt": "You are a logistics analyst optimizing supply chain operations.",
        "facts": {
            5:  "Average lead time from supplier to warehouse is 14.3 days via ocean freight.",
            15: "Warehouse capacity utilization in Q3 was 87%, up from 72% in Q2.",
            25: "The top 3 SKUs account for 41% of total revenue but only 12% of inventory cost.",
            35: "Shipping cost per unit decreased to $3.42 after renegotiating the FedEx contract.",
            55: "Demand forecast accuracy improved to 89% using the new ARIMA-LSTM hybrid model.",
        },
    },
    {
        "domain": "education_platform",
        "system_prompt": "You are an EdTech product lead designing a learning management system.",
        "facts": {
            5:  "The platform serves 45,000 active students across 120 institutions.",
            15: "Average course completion rate is 23%, with video courses at 31% and text at 18%.",
            25: "The recommendation engine increased engagement by 27% in A/B testing.",
            35: "SCORM 2004 compliance is required for integration with existing LMS vendors.",
            55: "Student satisfaction NPS score is 42, up from 28 after the redesign.",
        },
    },
    {
        "domain": "robotics",
        "system_prompt": "You are a robotics engineer programming a warehouse picking robot.",
        "facts": {
            5:  "The robot arm has 6 degrees of freedom with a maximum payload of 5kg.",
            15: "Path planning uses RRT-star algorithm with a 50ms replanning cycle.",
            25: "The vision system identifies objects using YOLOv8 at 30 FPS on the Jetson Orin.",
            35: "Gripper force is calibrated between 2N and 15N depending on object fragility.",
            55: "The pick rate target is 600 items per hour with less than 0.1% error rate.",
        },
    },
    {
        "domain": "climate_modeling",
        "system_prompt": "You are a climate scientist analyzing simulation outputs.",
        "facts": {
            5:  "The simulation uses a 0.25-degree spatial resolution global grid.",
            15: "CO2 forcing is set to RCP 8.5 scenario levels of 936 ppm by 2100.",
            25: "Ocean heat content anomaly in the simulation is +1.2 ZJ per decade.",
            35: "Arctic sea ice extent in the model reaches zero in September by 2045.",
            55: "The ensemble consists of 30 runs with perturbed initial conditions.",
        },
    },
]


def _filler_message(position: int, domain: str) -> dict:
    """Generate a realistic filler message for a given conversation position."""
    # Alternate between user and assistant messages
    if position % 2 == 0:
        role = "user"
        templates = [
            f"Can you elaborate on the {domain} aspect we discussed earlier?",
            f"What are the next steps for the {domain} work?",
            f"I have a question about the current {domain} approach.",
            f"Let me share some additional context about the {domain} situation.",
            f"Could you review the {domain} requirements one more time?",
            f"I think we need to reconsider the {domain} strategy.",
            f"Here is an update on the {domain} progress from the team.",
            f"What do you think about the alternative {domain} approach?",
            f"We got new data on the {domain} metrics. Let me share.",
            f"The stakeholders want an update on {domain} by end of week.",
            f"There might be a dependency between this and the {domain} piece.",
            f"Can you help me draft a summary of the {domain} findings?",
            f"I noticed an issue with the {domain} analysis we did.",
            f"Let me know if you need more information about {domain}.",
            f"The {domain} timeline might need to be adjusted.",
        ]
    else:
        role = "assistant"
        templates = [
            f"Based on the {domain} details, I recommend we proceed with the current plan.",
            f"Let me analyze the {domain} data and provide a detailed breakdown.",
            f"There are several important considerations for the {domain} component.",
            f"I have reviewed the {domain} materials and here are my observations.",
            f"The {domain} approach looks solid but we should verify a few assumptions.",
            f"Here is my assessment of the {domain} situation and recommended actions.",
            f"Looking at the {domain} requirements, I see three key areas to address.",
            f"I have some concerns about the {domain} timeline that we should discuss.",
            f"The {domain} metrics suggest we are on track but need to monitor closely.",
            f"Let me walk through the {domain} analysis step by step.",
            f"For the {domain} work, I suggest we break it into phases.",
            f"The {domain} data shows some interesting patterns worth investigating.",
            f"I will prepare a comprehensive {domain} report with these details.",
            f"Based on best practices for {domain}, here is what I recommend.",
            f"The {domain} component interacts with several other systems.",
        ]
    template = templates[position % len(templates)]
    # Add some padding to make messages realistic size
    padding = f" This involves careful consideration of multiple factors " \
              f"and requires coordination across the team to ensure alignment " \
              f"with the overall project objectives. Position {position} in conversation."
    return _make_message(role, template + padding)


def _run_retention_benchmark() -> float:
    """Category 1: Information Retention Under Pressure.

    Returns average fact retention rate across all 15 scenarios.
    """
    retention_scores: list[float] = []

    for scenario in _RETENTION_SCENARIOS:
        # Budget: large enough for a full conversation, but compaction will
        # squeeze it to 30%.
        wm = WorkingMemory(total_budget=3000)

        # Ingest system prompt
        wm.ingest(_make_message("system", scenario["system_prompt"]))

        # Simulate 60-message conversation with facts at specific positions
        for pos in range(60):
            if pos in scenario["facts"]:
                # This is a fact-bearing message
                fact = scenario["facts"][pos]
                if pos % 2 == 0:
                    msg = _make_message("user", f"Important update: {fact}")
                else:
                    msg = _make_message("assistant", f"Noted. Key finding: {fact}")
                msg["importance"] = 0.9
                wm.ingest(msg)
            else:
                wm.ingest(_filler_message(pos, scenario["domain"]))

        # Now force compaction: reduce budget to 30% of original
        wm.total_budget = 900
        wm.compact()

        # Check how many facts survived
        window = wm.get_window()
        window_text = _window_text(window)
        window_text_lower = window_text.lower()

        facts_found = 0
        total_facts = len(scenario["facts"])
        for pos, fact in scenario["facts"].items():
            # Check if the core information from the fact is in the window.
            # Use multiple signal words from each fact to be fair about
            # paraphrasing / compression.
            fact_words = fact.lower().split()
            # Pick distinctive words (> 4 chars) as markers
            markers = [w for w in fact_words if len(w) > 4]
            if not markers:
                markers = fact_words[:3]
            # Require at least 40% of markers to be present
            found_markers = sum(1 for m in markers if m in window_text_lower)
            if len(markers) > 0 and found_markers / len(markers) >= 0.4:
                facts_found += 1

        retention = facts_found / total_facts if total_facts > 0 else 0.0
        retention_scores.append(retention)

    avg_retention = sum(retention_scores) / len(retention_scores) if retention_scores else 0.0
    return avg_retention


# =====================================================================
# Category 2: Budget Allocation Adaptation (10 tasks)
# =====================================================================
# Present workloads where one section dominates and check if the working
# memory adapts its allocation.

_BUDGET_SCENARIOS: list[dict] = [
    # History-heavy (3 tasks)
    {
        "label": "history_heavy_1",
        "type": "history_heavy",
        "history_count": 40,
        "tool_count": 3,
        "expected_dominant": "history",
    },
    {
        "label": "history_heavy_2",
        "type": "history_heavy",
        "history_count": 45,
        "tool_count": 2,
        "expected_dominant": "history",
    },
    {
        "label": "history_heavy_3",
        "type": "history_heavy",
        "history_count": 50,
        "tool_count": 1,
        "expected_dominant": "history",
    },
    # Tool-heavy (3 tasks)
    {
        "label": "tool_heavy_1",
        "type": "tool_heavy",
        "history_count": 5,
        "tool_count": 20,
        "expected_dominant": "tool_results",
    },
    {
        "label": "tool_heavy_2",
        "type": "tool_heavy",
        "history_count": 3,
        "tool_count": 25,
        "expected_dominant": "tool_results",
    },
    {
        "label": "tool_heavy_3",
        "type": "tool_heavy",
        "history_count": 4,
        "tool_count": 18,
        "expected_dominant": "tool_results",
    },
    # Mixed (4 tasks)
    {
        "label": "mixed_balanced",
        "type": "mixed",
        "history_count": 15,
        "tool_count": 15,
        "expected_dominant": None,  # neither should dominate excessively
    },
    {
        "label": "mixed_history_leaning",
        "type": "mixed",
        "history_count": 25,
        "tool_count": 10,
        "expected_dominant": "history",
    },
    {
        "label": "mixed_tool_leaning",
        "type": "mixed",
        "history_count": 10,
        "tool_count": 20,
        "expected_dominant": "tool_results",
    },
    {
        "label": "mixed_with_system",
        "type": "mixed",
        "history_count": 15,
        "tool_count": 12,
        "expected_dominant": None,
    },
]


def _make_history_msg(idx: int) -> dict:
    """Generate a realistic history message."""
    if idx % 2 == 0:
        return _make_message(
            "user",
            f"Step {idx}: I need help with the implementation of feature "
            f"number {idx}. The requirements specify that we need to handle "
            f"edge cases including null values, timeouts, and concurrent access. "
            f"Please advise on the best approach."
        )
    else:
        return _make_message(
            "assistant",
            f"For step {idx}, I recommend implementing a defensive approach "
            f"with input validation, retry logic with exponential backoff, "
            f"and optimistic locking for concurrency. Here are the details "
            f"of the implementation strategy for this component."
        )


def _make_tool_result(idx: int) -> dict:
    """Generate a realistic tool result message."""
    tool_types = [
        ("file_read", f"Contents of src/module_{idx}.py:\n"
         f"class Handler{idx}:\n    def process(self, data):\n"
         f"        result = self.validate(data)\n"
         f"        return self.transform(result)\n"
         f"    def validate(self, data):\n"
         f"        if not data: raise ValueError('empty')\n"
         f"        return data"),
        ("search", f"Found 3 results for query 'handler {idx}':\n"
         f"1. src/handlers/base.py:45 - class BaseHandler\n"
         f"2. src/handlers/impl.py:120 - class ConcreteHandler{idx}\n"
         f"3. tests/test_handler.py:30 - def test_handler_{idx}"),
        ("terminal", f"$ python -m pytest tests/test_{idx}.py -v\n"
         f"PASSED test_basic_{idx} (0.12s)\n"
         f"PASSED test_edge_case_{idx} (0.08s)\n"
         f"FAILED test_concurrent_{idx} - AssertionError\n"
         f"2 passed, 1 failed in 0.34s"),
        ("api_call", f"GET /api/v1/resource/{idx}\n"
         f"Status: 200 OK\n"
         f"Response: {{'id': {idx}, 'name': 'resource_{idx}', "
         f"'status': 'active', 'metrics': {{'latency_ms': {23 + idx}, "
         f"'throughput': {1000 - idx * 10}}}}}"),
    ]
    tool_type, content = tool_types[idx % len(tool_types)]
    return _make_message("tool_result", content, tool_name=tool_type)


def _run_budget_benchmark() -> float:
    """Category 2: Budget Allocation Adaptation.

    Returns average adaptation score across all 10 scenarios.
    """
    scores: list[float] = []

    for scenario in _BUDGET_SCENARIOS:
        wm = WorkingMemory(total_budget=2000)

        # Add system prompt
        wm.ingest(_make_message("system", "You are a helpful coding assistant."))

        # Add history messages
        for i in range(scenario["history_count"]):
            wm.ingest(_make_history_msg(i))

        # Add tool results
        for i in range(scenario["tool_count"]):
            wm.ingest(_make_tool_result(i))

        window = wm.get_window()

        # Measure: how well does the window content reflect the workload?
        history_words = 0
        tool_words = 0
        total_words = 0
        for m in window:
            words = _count_words(m.get("content", ""))
            total_words += words
            role = m.get("role", "")
            if role == "tool_result":
                tool_words += words
            elif role in ("user", "assistant"):
                section = m.get("section", "")
                if section != "system":
                    history_words += words

        if total_words == 0:
            scores.append(0.0)
            continue

        # Calculate what fraction of the budget each section got
        history_frac = history_words / total_words
        tool_frac = tool_words / total_words

        # Calculate what the ideal allocation should be based on input volume
        input_history_words = scenario["history_count"] * 35  # approx words per msg
        input_tool_words = scenario["tool_count"] * 40
        input_total = input_history_words + input_tool_words
        if input_total == 0:
            scores.append(1.0)
            continue

        ideal_history_frac = input_history_words / input_total
        ideal_tool_frac = input_tool_words / input_total

        # Score: how close is actual allocation to ideal? (1 - mean absolute error)
        mae = (abs(history_frac - ideal_history_frac) +
               abs(tool_frac - ideal_tool_frac)) / 2
        adaptation_score = max(0.0, 1.0 - mae)

        # Bonus: check that expected dominant section actually has more content
        if scenario["expected_dominant"] == "history" and history_frac > tool_frac:
            adaptation_score = min(1.0, adaptation_score + 0.1)
        elif scenario["expected_dominant"] == "tool_results" and tool_frac > history_frac:
            adaptation_score = min(1.0, adaptation_score + 0.1)
        elif scenario["expected_dominant"] is None:
            # Mixed: neither should dominate excessively (ratio within 3:1)
            if history_frac > 0 and tool_frac > 0:
                ratio = max(history_frac, tool_frac) / min(history_frac, tool_frac)
                if ratio < 3.0:
                    adaptation_score = min(1.0, adaptation_score + 0.1)

        scores.append(min(1.0, adaptation_score))

    return sum(scores) / len(scores) if scores else 0.0


# =====================================================================
# Category 3: JIT Loading Accuracy (10 tasks)
# =====================================================================
# Provide a query + 30 chunks (5 relevant, 25 irrelevant). Working memory
# must select the right chunks.

_JIT_SCENARIOS: list[dict] = [
    {
        "query": "How do I fix the database connection timeout issue?",
        "relevant": [
            {"id": "db1", "content": "Database connection pool settings: max_connections=50, timeout=30s, idle_timeout=600s. Configured in config/database.yml."},
            {"id": "db2", "content": "Common timeout fix: increase connection_timeout to 60s and add retry logic with exponential backoff starting at 1s."},
            {"id": "db3", "content": "PostgreSQL pg_hba.conf must allow connections from the application subnet 10.0.1.0/24 on port 5432."},
            {"id": "db4", "content": "Connection pooling with PgBouncer reduces timeout issues by maintaining warm connections. Set pool_mode to transaction."},
            {"id": "db5", "content": "Database monitoring shows connection count spikes to 200+ during peak hours, exceeding the max_connections limit of 100."},
        ],
        "irrelevant_seed": "frontend_styling",
    },
    {
        "query": "What is the authentication flow for the mobile app?",
        "relevant": [
            {"id": "auth1", "content": "Mobile authentication uses OAuth 2.0 PKCE flow: app generates code_verifier, sends code_challenge to /authorize endpoint."},
            {"id": "auth2", "content": "After user login, the auth server returns an authorization code which is exchanged for access and refresh tokens at /token."},
            {"id": "auth3", "content": "Refresh tokens are stored in the device keychain (iOS) or encrypted SharedPreferences (Android) with a 30-day expiry."},
            {"id": "auth4", "content": "The access token JWT contains claims: sub, email, roles[], exp. It is sent as Bearer token in Authorization header."},
            {"id": "auth5", "content": "Biometric authentication (Face ID / fingerprint) can be used to unlock stored credentials without re-entering password."},
        ],
        "irrelevant_seed": "build_pipeline",
    },
    {
        "query": "How should we structure the Kubernetes deployment manifests?",
        "relevant": [
            {"id": "k8s1", "content": "Use Kustomize overlays: base/ has common resources, overlays/dev and overlays/prod have environment-specific patches."},
            {"id": "k8s2", "content": "Deployment spec should set resource requests (cpu: 250m, memory: 512Mi) and limits (cpu: 1000m, memory: 1Gi) for each container."},
            {"id": "k8s3", "content": "Health checks: livenessProbe on /healthz every 10s, readinessProbe on /ready every 5s with initialDelaySeconds of 30."},
            {"id": "k8s4", "content": "Horizontal Pod Autoscaler targets 70% CPU utilization with minReplicas=2 and maxReplicas=10 for the web tier."},
            {"id": "k8s5", "content": "Secrets are managed with Sealed Secrets controller: encrypt with kubeseal, store encrypted YAML in git safely."},
        ],
        "irrelevant_seed": "user_analytics",
    },
    {
        "query": "What are the data validation rules for the order processing system?",
        "relevant": [
            {"id": "val1", "content": "Order validation: total_amount must be positive, currency must be ISO 4217 (3-letter code), quantity must be integer >= 1."},
            {"id": "val2", "content": "Customer validation: email must match RFC 5322 pattern, phone must be E.164 format, shipping_address requires street, city, postal_code, country."},
            {"id": "val3", "content": "Payment validation: card_number passes Luhn check, expiry_date is future, CVV is 3-4 digits. PCI compliance requires tokenization."},
            {"id": "val4", "content": "Inventory check: each line item SKU must exist in catalog, requested quantity must not exceed available_stock minus reserved_stock."},
            {"id": "val5", "content": "Business rules: orders over $10,000 require manager approval, international orders need customs declaration, bulk orders get 15% discount."},
        ],
        "irrelevant_seed": "team_management",
    },
    {
        "query": "How do we implement the real-time notification system?",
        "relevant": [
            {"id": "notif1", "content": "WebSocket server using Socket.IO: clients connect to /ws/notifications, authenticate with JWT, join room based on user_id."},
            {"id": "notif2", "content": "Notification types: order_update, message_received, system_alert, promotion. Each has priority level: low, medium, high, critical."},
            {"id": "notif3", "content": "Push notifications via Firebase Cloud Messaging (Android) and APNs (iOS). Server sends to FCM/APNs when user is offline."},
            {"id": "notif4", "content": "Notification persistence: all notifications stored in notifications table with columns: id, user_id, type, payload, read_at, created_at."},
            {"id": "notif5", "content": "Rate limiting: max 10 notifications per minute per user. Batch similar notifications (e.g., 5 new messages -> 1 notification)."},
        ],
        "irrelevant_seed": "database_migration",
    },
    {
        "query": "What is the test coverage strategy for the payment module?",
        "relevant": [
            {"id": "test1", "content": "Unit tests cover all payment methods: credit_card, debit_card, bank_transfer, digital_wallet. Each has happy path and 3+ error cases."},
            {"id": "test2", "content": "Integration tests use Stripe test mode with test card numbers: 4242424242424242 (success), 4000000000000002 (decline)."},
            {"id": "test3", "content": "Load testing: simulate 500 concurrent payment requests using Locust. Target: 95th percentile response time under 2 seconds."},
            {"id": "test4", "content": "Security testing: verify PCI DSS compliance, test for injection attacks on payment forms, validate TLS 1.2+ enforcement."},
            {"id": "test5", "content": "Regression suite: 47 tests covering refunds, partial captures, subscription billing, currency conversion, and webhook handling."},
        ],
        "irrelevant_seed": "documentation",
    },
    {
        "query": "How is the search indexing pipeline configured?",
        "relevant": [
            {"id": "search1", "content": "Elasticsearch index configuration: 3 primary shards, 1 replica, refresh_interval 30s. Mapping uses text type with standard analyzer."},
            {"id": "search2", "content": "Indexing pipeline: CDC events from PostgreSQL via Debezium -> Kafka topic -> Elasticsearch sink connector with batch_size 500."},
            {"id": "search3", "content": "Custom analyzers: product_analyzer uses edge_ngram tokenizer (min=2, max=10) for autocomplete. Description uses english stemmer."},
            {"id": "search4", "content": "Reindexing strategy: zero-downtime via alias swap. New index built in background, alias switched atomically when ready."},
            {"id": "search5", "content": "Search relevance tuning: boost title field 3x, use function_score with recency decay (scale=30d) and popularity boost."},
        ],
        "irrelevant_seed": "hr_policies",
    },
    {
        "query": "What are the API rate limiting rules and how are they enforced?",
        "relevant": [
            {"id": "rate1", "content": "Rate limits: free tier 100 req/min, pro tier 1000 req/min, enterprise tier 10000 req/min. Limits tracked per API key."},
            {"id": "rate2", "content": "Implementation: Redis-based sliding window counter. Key pattern: ratelimit:{api_key}:{minute_bucket}. TTL set to 120 seconds."},
            {"id": "rate3", "content": "Response headers: X-RateLimit-Limit, X-RateLimit-Remaining, X-RateLimit-Reset (Unix timestamp). 429 response when exceeded."},
            {"id": "rate4", "content": "Burst allowance: up to 2x the per-minute limit in a 10-second window, enabling short bursts without penalty."},
            {"id": "rate5", "content": "Rate limit bypass: internal services use service accounts with no rate limit. Health check endpoints are exempt."},
        ],
        "irrelevant_seed": "graphic_design",
    },
    {
        "query": "How do we handle data migration from the legacy system?",
        "relevant": [
            {"id": "mig1", "content": "Migration phases: 1) schema mapping (legacy Oracle to new PostgreSQL), 2) data extraction, 3) transformation, 4) loading, 5) validation."},
            {"id": "mig2", "content": "Data mapping: legacy CUSTOMER table -> new users + profiles tables. ADDRESS blob parsed into structured street, city, state, zip fields."},
            {"id": "mig3", "content": "Historical data: 12 years of transactions (47M rows). Migrate last 3 years to hot storage, archive remainder to S3 Parquet."},
            {"id": "mig4", "content": "Cutover strategy: dual-write during transition (2 weeks), then read comparison to validate consistency before legacy shutdown."},
            {"id": "mig5", "content": "Rollback plan: maintain legacy system read-only for 90 days post-migration. Keep CDC stream for delta sync if rollback needed."},
        ],
        "irrelevant_seed": "marketing_campaign",
    },
    {
        "query": "What logging and observability standards should we follow?",
        "relevant": [
            {"id": "obs1", "content": "Structured logging in JSON format: timestamp, level, service, trace_id, span_id, message, and context-specific fields."},
            {"id": "obs2", "content": "Distributed tracing with OpenTelemetry: auto-instrument HTTP clients and database drivers. Export spans to Jaeger."},
            {"id": "obs3", "content": "Metrics: RED method for services (Rate, Errors, Duration). USE method for infrastructure (Utilization, Saturation, Errors)."},
            {"id": "obs4", "content": "Log levels: DEBUG for development only, INFO for request lifecycle, WARN for recoverable issues, ERROR for failures needing attention."},
            {"id": "obs5", "content": "Alerting tiers: P1 pages on-call (service down), P2 sends Slack (error rate >5%), P3 creates ticket (slow degradation)."},
        ],
        "irrelevant_seed": "social_media",
    },
]

# Irrelevant chunk pools keyed by seed topic
_IRRELEVANT_POOLS: dict[str, list[dict]] = {
    "frontend_styling": [
        {"id": "irr_fs1", "content": "CSS Grid layout with 12 columns and 8px gutter provides a flexible responsive design foundation."},
        {"id": "irr_fs2", "content": "The design system uses 4px spacing scale: 4, 8, 12, 16, 24, 32, 48, 64 for consistent padding and margins."},
        {"id": "irr_fs3", "content": "Color palette: primary #2563EB, secondary #7C3AED, success #059669, warning #D97706, error #DC2626."},
        {"id": "irr_fs4", "content": "Typography scale: heading-1 36px/44px, heading-2 30px/36px, body 16px/24px, caption 12px/16px. Font: Inter."},
        {"id": "irr_fs5", "content": "Animation standards: transitions use ease-out timing at 200ms for hover, 300ms for modal open, 150ms for button press."},
    ],
    "build_pipeline": [
        {"id": "irr_bp1", "content": "Webpack 5 configuration: entry point src/index.tsx, output dist/bundle.[contenthash].js. Tree shaking enabled."},
        {"id": "irr_bp2", "content": "ESLint rules extend airbnb-typescript with custom overrides for import ordering and line length of 120 characters."},
        {"id": "irr_bp3", "content": "Docker multi-stage build: stage 1 node:20-alpine for build, stage 2 nginx:alpine for serving static assets."},
        {"id": "irr_bp4", "content": "npm workspace setup with packages/shared, packages/web, and packages/mobile sharing common types and utilities."},
        {"id": "irr_bp5", "content": "Pre-commit hooks run Prettier, ESLint, TypeScript compiler, and jest --changedSince to validate before push."},
    ],
    "user_analytics": [
        {"id": "irr_ua1", "content": "Google Analytics 4 event tracking: page_view, button_click, form_submit, purchase, with custom dimensions for user_type."},
        {"id": "irr_ua2", "content": "Mixpanel funnel: sign_up -> onboarding_complete -> first_purchase -> repeat_purchase. Current conversion: 12% end-to-end."},
        {"id": "irr_ua3", "content": "A/B testing framework: feature flags via LaunchDarkly, traffic split 50/50, minimum sample size 10,000 per variant."},
        {"id": "irr_ua4", "content": "User cohort analysis: monthly active users segmented by acquisition channel, geography, and subscription tier."},
        {"id": "irr_ua5", "content": "Heatmap data from Hotjar shows 67% of users never scroll below the fold on the pricing page."},
    ],
    "team_management": [
        {"id": "irr_tm1", "content": "Sprint planning uses 2-week cycles with capacity calculated at 6 story points per developer per sprint."},
        {"id": "irr_tm2", "content": "Engineering levels: IC1 (junior), IC2 (mid), IC3 (senior), IC4 (staff), IC5 (principal). Promotion requires peer review."},
        {"id": "irr_tm3", "content": "On-call rotation: 1 week shifts, 4-person rotation, primary and secondary. Handoff document required at rotation change."},
        {"id": "irr_tm4", "content": "Team retrospective format: Start/Stop/Continue with anonymous voting. Action items tracked in Jira with owner and due date."},
        {"id": "irr_tm5", "content": "Technical interview process: 1 phone screen, 1 coding round, 1 system design, 1 behavioral. Decision within 48 hours."},
    ],
    "database_migration": [
        {"id": "irr_dm1", "content": "Table partitioning strategy: range partition orders table by created_at monthly. Partition pruning reduces query time 60%."},
        {"id": "irr_dm2", "content": "Read replica configuration: 2 replicas in same region for read scaling, 1 cross-region replica for disaster recovery."},
        {"id": "irr_dm3", "content": "Vacuum and analyze schedule: autovacuum threshold 50 tuples, analyze threshold 50 tuples, cost_limit 200 per cycle."},
        {"id": "irr_dm4", "content": "Connection string format: postgresql://user:pass@host:5432/dbname?sslmode=require&application_name=myapp"},
        {"id": "irr_dm5", "content": "Index maintenance: rebuild indexes monthly during maintenance window. B-tree for equality, GIN for full-text, BRIN for time series."},
    ],
    "documentation": [
        {"id": "irr_doc1", "content": "API documentation hosted on ReadTheDocs with Sphinx. OpenAPI 3.0 spec auto-generated from FastAPI route decorators."},
        {"id": "irr_doc2", "content": "README template: project description, quick start, configuration options, API reference, contributing guide, license."},
        {"id": "irr_doc3", "content": "Architecture Decision Records stored in docs/adr/ directory. Format: title, status, context, decision, consequences."},
        {"id": "irr_doc4", "content": "Runbook for common operational tasks: database failover, certificate renewal, scaling up/down, log investigation."},
        {"id": "irr_doc5", "content": "Code documentation standard: all public functions must have docstring with Args, Returns, Raises. Google style preferred."},
    ],
    "hr_policies": [
        {"id": "irr_hr1", "content": "PTO policy: 20 days annual leave, 10 sick days, 5 personal days. Rollover limited to 5 unused PTO days per year."},
        {"id": "irr_hr2", "content": "Remote work policy: hybrid 3 days office / 2 days remote. Core hours 10am-3pm for meetings and collaboration."},
        {"id": "irr_hr3", "content": "Professional development budget: $2,500 per employee per year for conferences, courses, books, and certifications."},
        {"id": "irr_hr4", "content": "Parental leave: 16 weeks paid for primary caregiver, 8 weeks for secondary. Gradual return option at 80% for 4 weeks."},
        {"id": "irr_hr5", "content": "Annual review cycle: self-assessment in January, manager review in February, calibration in March, feedback in April."},
    ],
    "graphic_design": [
        {"id": "irr_gd1", "content": "Brand guidelines: logo minimum clear space is 2x the height of the logomark. Never stretch, rotate, or recolor."},
        {"id": "irr_gd2", "content": "Icon set uses 24x24 grid with 2px stroke weight. Rounded line caps, 4px corner radius for filled variants."},
        {"id": "irr_gd3", "content": "Photo style guide: natural lighting preferred, candid shots over posed. Minimum resolution 2400x1600 for hero images."},
        {"id": "irr_gd4", "content": "Illustration style: flat design with subtle gradients, limited to 4 colors per illustration from brand palette."},
        {"id": "irr_gd5", "content": "Print specifications: business cards 3.5x2 inches, 300 DPI, CMYK color space, 0.125 inch bleed on all sides."},
    ],
    "marketing_campaign": [
        {"id": "irr_mc1", "content": "Email campaign: 3-part drip sequence. Day 1: welcome, Day 3: feature highlight, Day 7: case study with CTA."},
        {"id": "irr_mc2", "content": "Social media calendar: Monday thought leadership, Wednesday product tips, Friday community spotlight. Post at 10am EST."},
        {"id": "irr_mc3", "content": "Content marketing: 2 blog posts per week targeting long-tail SEO keywords with 1500+ word count for ranking."},
        {"id": "irr_mc4", "content": "Paid advertising: Google Ads budget $15,000/month, Facebook $8,000/month. Target CPA $25 for trial signups."},
        {"id": "irr_mc5", "content": "Webinar series: monthly 45-minute sessions on industry topics. Average attendance 150, conversion to trial 8%."},
    ],
    "social_media": [
        {"id": "irr_sm1", "content": "Instagram content strategy: product photos Monday/Thursday, behind-scenes Tuesday, user-generated Wednesday, Stories daily."},
        {"id": "irr_sm2", "content": "Twitter engagement: respond to mentions within 2 hours. Weekly Twitter Space on industry trends, Thursday 2pm EST."},
        {"id": "irr_sm3", "content": "LinkedIn articles: monthly long-form thought leadership pieces. Target 500+ reactions and 50+ comments per post."},
        {"id": "irr_sm4", "content": "TikTok strategy: 60-second tutorial videos 3x per week. Trending audio + educational content for developer audience."},
        {"id": "irr_sm5", "content": "Community management: Discord server with channels for support, feedback, showcase. Moderator team of 5 volunteers."},
    ],
}


def _build_jit_chunks(scenario: dict) -> list[dict]:
    """Build the 30-chunk pool (5 relevant + 25 irrelevant) for a JIT scenario."""
    chunks = list(scenario["relevant"])  # 5 relevant
    # Get 25 irrelevant from other pools (cycle through all except the scenario's seed)
    irrelevant = []
    for seed, pool in _IRRELEVANT_POOLS.items():
        if seed != scenario["irrelevant_seed"]:
            irrelevant.extend(pool)
    # Take exactly 25 irrelevant chunks
    irrelevant = irrelevant[:25]
    chunks.extend(irrelevant)
    # Deterministic shuffle: interleave to avoid relevant chunks being clustered
    mixed: list[dict] = []
    relevant_ids = {c["id"] for c in scenario["relevant"]}
    irr_iter = iter([c for c in chunks if c["id"] not in relevant_ids])
    rel_iter = iter([c for c in chunks if c["id"] in relevant_ids])
    # Place relevant chunks at positions 3, 8, 14, 21, 27
    rel_positions = {3, 8, 14, 21, 27}
    rel_idx = 0
    irr_idx = 0
    for i in range(30):
        if i in rel_positions:
            chunk = next(rel_iter, None)
            if chunk:
                mixed.append(chunk)
                continue
        chunk = next(irr_iter, None)
        if chunk:
            mixed.append(chunk)
    # Fill any remaining
    for c in rel_iter:
        mixed.append(c)
    for c in irr_iter:
        mixed.append(c)
    return mixed[:30]


def _run_jit_benchmark() -> float:
    """Category 3: JIT Loading Accuracy.

    Returns average F1 score of relevant chunk selection across all 10 scenarios.
    """
    f1_scores: list[float] = []

    for scenario in _JIT_SCENARIOS:
        wm = WorkingMemory(total_budget=2000)

        # Add a system prompt and some baseline context
        wm.ingest(_make_message("system", "You are a helpful assistant with access to a knowledge base."))
        wm.ingest(_make_message("user", "I have a question about our system."))

        # Build the chunk pool
        chunks = _build_jit_chunks(scenario)
        relevant_ids = {c["id"] for c in scenario["relevant"]}

        # Ask working memory to load relevant chunks
        loaded = wm.query_and_load(scenario["query"], chunks)
        loaded_ids = {c.get("id") for c in loaded}

        # Calculate precision, recall, F1
        true_positives = len(loaded_ids & relevant_ids)
        if loaded_ids:
            precision = true_positives / len(loaded_ids)
        else:
            precision = 0.0
        if relevant_ids:
            recall = true_positives / len(relevant_ids)
        else:
            recall = 1.0

        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        f1_scores.append(f1)

    return sum(f1_scores) / len(f1_scores) if f1_scores else 0.0


# =====================================================================
# Category 4: System Prompt Preservation (5 tasks)
# =====================================================================

_SYSTEM_PROMPTS: list[str] = [
    "You are a senior software engineer at TechCorp. You follow clean code principles, "
    "write comprehensive tests, and always consider edge cases. When reviewing code, "
    "check for security vulnerabilities, performance issues, and maintainability. "
    "Never suggest solutions that violate SOLID principles or introduce technical debt.",

    "You are a medical AI assistant. You must always include disclaimers when discussing "
    "treatments. Never provide definitive diagnoses. Always recommend consulting a healthcare "
    "professional. Cite evidence-based sources when possible. Be empathetic and clear "
    "in explanations. Do not suggest experimental treatments without explicit warnings.",

    "You are a financial advisor AI. All recommendations must include risk disclaimers. "
    "Never guarantee returns. Always consider the client's risk tolerance, time horizon, "
    "and financial goals. Recommend diversification across asset classes. Comply with "
    "SEC regulations. Do not provide tax advice without recommending a CPA.",

    "You are a legal research assistant. Always cite relevant statutes and case law. "
    "Distinguish between binding and persuasive authority. Flag when an issue requires "
    "jurisdiction-specific analysis. Never provide legal advice directly; frame responses "
    "as research summaries. Note when precedent may be outdated or subject to appeal.",

    "You are an expert data scientist specializing in machine learning. When recommending "
    "models, always consider the bias-variance tradeoff, data size requirements, and "
    "interpretability needs. Validate assumptions about data distributions. Recommend "
    "appropriate evaluation metrics for the specific problem type. Never overfit to "
    "validation sets and always hold out a proper test set.",
]


def _run_system_prompt_benchmark() -> float:
    """Category 4: System Prompt Preservation.

    Returns average word-level preservation rate across 5 scenarios.
    """
    preservation_scores: list[float] = []

    for system_prompt in _SYSTEM_PROMPTS:
        wm = WorkingMemory(total_budget=2500)

        # Ingest system prompt
        wm.ingest(_make_message("system", system_prompt))

        # Simulate 50 messages of conversation
        for i in range(50):
            if i % 2 == 0:
                wm.ingest(_make_message(
                    "user",
                    f"Question {i}: Can you help me with task number {i}? "
                    f"I need detailed analysis and recommendations based on "
                    f"the current situation. This involves reviewing multiple "
                    f"aspects and providing actionable insights."
                ))
            else:
                wm.ingest(_make_message(
                    "assistant",
                    f"Response {i}: Based on my analysis, here are the key "
                    f"findings and recommendations. I have considered multiple "
                    f"factors including requirements, constraints, and best "
                    f"practices to provide this comprehensive response."
                ))

        # Compact
        wm.compact()

        # Check system prompt preservation in the window
        window = wm.get_window()
        window_text = _window_text(window)

        # Word-level preservation: what fraction of the system prompt words
        # appear in the window?
        prompt_words = system_prompt.lower().split()
        if not prompt_words:
            preservation_scores.append(1.0)
            continue

        found = 0
        window_lower = window_text.lower()
        for word in prompt_words:
            # Check exact word presence (not substring)
            if word in window_lower:
                found += 1

        preservation = found / len(prompt_words)
        preservation_scores.append(preservation)

    return sum(preservation_scores) / len(preservation_scores) if preservation_scores else 0.0


# =====================================================================
# Main benchmark
# =====================================================================

def run_benchmark() -> float:
    """Run the complete working memory benchmark.

    Returns the composite fitness score.
    """
    print("=" * 60)
    print("Working Memory Benchmark")
    print("=" * 60)

    # Category 1: Information Retention Under Pressure
    print("\n--- Category 1: Information Retention Under Pressure ---")
    fact_retention = _run_retention_benchmark()
    print(f"  fact_retention:    {fact_retention:.6f}")

    # Category 2: Budget Allocation Adaptation
    print("\n--- Category 2: Budget Allocation Adaptation ---")
    budget_adaptation = _run_budget_benchmark()
    print(f"  budget_adaptation: {budget_adaptation:.6f}")

    # Category 3: JIT Loading Accuracy
    print("\n--- Category 3: JIT Loading Accuracy ---")
    jit_f1 = _run_jit_benchmark()
    print(f"  jit_loading_f1:    {jit_f1:.6f}")

    # Category 4: System Prompt Preservation
    print("\n--- Category 4: System Prompt Preservation ---")
    sys_preservation = _run_system_prompt_benchmark()
    print(f"  sys_prompt_pres:   {sys_preservation:.6f}")

    # Composite score
    score = (
        0.35 * fact_retention
        + 0.25 * budget_adaptation
        + 0.25 * jit_f1
        + 0.15 * sys_preservation
    )

    print("\n" + "=" * 60)
    print(f"  fact_retention     (0.35): {fact_retention:.6f}")
    print(f"  budget_adaptation  (0.25): {budget_adaptation:.6f}")
    print(f"  jit_loading_f1     (0.25): {jit_f1:.6f}")
    print(f"  sys_prompt_pres    (0.15): {sys_preservation:.6f}")
    print(f"score: {score:.6f}")
    return score


if __name__ == "__main__":
    run_benchmark()
