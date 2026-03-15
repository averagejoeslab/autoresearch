#!/usr/bin/env python3
"""
Benchmark: Long-Term Memory
==============================
Evaluates the complete LongTermMemory system across 4 challenge categories:

  1. Storage & Retrieval        (15 tasks)  — exact, semantic, cross-category
  2. Episodic→Semantic Consolidation (10 tasks)  — generalization + specificity
  3. Retrieval Under Scale      (5 tasks)   — recall stability at 50/100/200/500
  4. Knowledge Integration      (10 tasks)  — multi-hop fact chaining

Fitness = 0.30 * retrieval_f1
        + 0.25 * consolidation_score
        + 0.20 * multi_hop_accuracy
        + 0.15 * scale_stability
        + 0.10 * (1 - avg_latency_normalized)

Run:  python benchmark.py
"""

from __future__ import annotations

import os
import sys
import time

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from harness import LongTermMemory  # noqa: E402


# ====================================================================
# Shared helpers
# ====================================================================

def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    import re
    text = re.sub(r"[^\w\s]", " ", text.lower())
    return " ".join(text.split())


def _words(text: str) -> set[str]:
    return set(_normalize(text).split())


def _result_texts(results: list[dict]) -> list[str]:
    return [r.get("content", "") for r in results]


def _any_result_contains(results: list[dict], keywords: list[str]) -> bool:
    """True if ANY result's content contains ALL keywords (case-insensitive)."""
    kw_lower = [k.lower() for k in keywords]
    for r in results:
        content = r.get("content", "").lower()
        if all(k in content for k in kw_lower):
            return True
    return False


def _result_contains_any(results: list[dict], keyword_sets: list[list[str]]) -> bool:
    """True if any result matches any one of the keyword sets."""
    for kw_set in keyword_sets:
        if _any_result_contains(results, kw_set):
            return True
    return False


def _recall_at_k(results: list[dict], expected_keywords: list[list[str]], k: int = 5) -> float:
    """Fraction of expected items found in top-k results.

    expected_keywords: list of keyword-lists.  An expected item is "found"
    if any result contains ALL of its keywords.
    """
    results = results[:k]
    found = 0
    for kw_set in expected_keywords:
        if _any_result_contains(results, kw_set):
            found += 1
    return found / max(len(expected_keywords), 1)


def _precision_at_k(results: list[dict], expected_keywords: list[list[str]], k: int = 5) -> float:
    """Fraction of returned results that match at least one expected item."""
    results = results[:k]
    if not results:
        return 0.0
    matched = 0
    for r in results:
        content = r.get("content", "").lower()
        for kw_set in expected_keywords:
            if all(k.lower() in content for k in kw_set):
                matched += 1
                break
    return matched / len(results)


# ====================================================================
# CATEGORY 1: Storage and Retrieval (15 tasks)
# ====================================================================

# 100 facts across 10 categories
FACTS: list[dict] = [
    # --- Geography (10) ---
    {"content": "The capital of France is Paris", "category": "geography"},
    {"content": "The capital of Japan is Tokyo", "category": "geography"},
    {"content": "The capital of Australia is Canberra, not Sydney", "category": "geography"},
    {"content": "Brazil is the largest country in South America by area", "category": "geography"},
    {"content": "The Nile is the longest river in Africa", "category": "geography"},
    {"content": "Mount Everest is located on the border of Nepal and Tibet", "category": "geography"},
    {"content": "Iceland is a volcanic island in the North Atlantic Ocean", "category": "geography"},
    {"content": "The Amazon rainforest spans nine countries in South America", "category": "geography"},
    {"content": "Switzerland is a landlocked country bordered by France, Germany, Austria, Liechtenstein, and Italy", "category": "geography"},
    {"content": "The Sahara Desert is the largest hot desert in the world, located in northern Africa", "category": "geography"},

    # --- Science (10) ---
    {"content": "Water boils at 100 degrees Celsius at standard atmospheric pressure", "category": "science"},
    {"content": "The speed of light in a vacuum is approximately 299,792 kilometers per second", "category": "science"},
    {"content": "DNA stands for deoxyribonucleic acid and carries genetic information", "category": "science"},
    {"content": "Photosynthesis converts carbon dioxide and water into glucose and oxygen using sunlight", "category": "science"},
    {"content": "The human body has 206 bones in the adult skeleton", "category": "science"},
    {"content": "Gravity on Earth accelerates objects at approximately 9.8 meters per second squared", "category": "science"},
    {"content": "Mitochondria are often called the powerhouse of the cell because they generate ATP", "category": "science"},
    {"content": "The periodic table organizes chemical elements by atomic number", "category": "science"},
    {"content": "Einstein's equation E equals mc squared relates energy to mass", "category": "science"},
    {"content": "Antibiotics treat bacterial infections but are ineffective against viruses", "category": "science"},

    # --- Programming (10) ---
    {"content": "Python was created by Guido van Rossum and first released in 1991", "category": "programming"},
    {"content": "JavaScript is the primary language for web browser scripting", "category": "programming"},
    {"content": "Git is a distributed version control system created by Linus Torvalds", "category": "programming"},
    {"content": "SQL stands for Structured Query Language and is used to manage relational databases", "category": "programming"},
    {"content": "The Rust programming language emphasizes memory safety without garbage collection", "category": "programming"},
    {"content": "Docker containers package applications with their dependencies for consistent deployment", "category": "programming"},
    {"content": "REST APIs use HTTP methods like GET, POST, PUT, and DELETE for web services", "category": "programming"},
    {"content": "Machine learning models learn patterns from data rather than being explicitly programmed", "category": "programming"},
    {"content": "TCP/IP is the fundamental communication protocol suite of the internet", "category": "programming"},
    {"content": "Linux is an open-source operating system kernel first released by Linus Torvalds in 1991", "category": "programming"},

    # --- History (10) ---
    {"content": "World War II ended in 1945 with the surrender of Japan", "category": "history"},
    {"content": "The Declaration of Independence was signed in 1776 in Philadelphia", "category": "history"},
    {"content": "The Roman Empire fell in 476 AD when the last Western Roman Emperor was deposed", "category": "history"},
    {"content": "The printing press was invented by Johannes Gutenberg around 1440", "category": "history"},
    {"content": "The French Revolution began in 1789 with the storming of the Bastille", "category": "history"},
    {"content": "The Berlin Wall fell on November 9, 1989, reunifying East and West Germany", "category": "history"},
    {"content": "The Renaissance was a cultural movement that began in Italy in the 14th century", "category": "history"},
    {"content": "Neil Armstrong became the first person to walk on the Moon on July 20, 1969", "category": "history"},
    {"content": "The Industrial Revolution began in Britain in the late 18th century", "category": "history"},
    {"content": "The Magna Carta was signed in 1215 and limited the power of the English king", "category": "history"},

    # --- Music (10) ---
    {"content": "Beethoven composed nine symphonies and became deaf later in life", "category": "music"},
    {"content": "The Beatles were a British rock band formed in Liverpool in 1960", "category": "music"},
    {"content": "Jazz originated in New Orleans in the early 20th century from African American communities", "category": "music"},
    {"content": "Mozart was a prolific composer from Salzburg who wrote over 600 works", "category": "music"},
    {"content": "The piano has 88 keys spanning just over seven octaves", "category": "music"},
    {"content": "Hip hop music emerged in the Bronx, New York City in the 1970s", "category": "music"},
    {"content": "Bach's Well-Tempered Clavier is considered a foundational work for keyboard music", "category": "music"},
    {"content": "A standard guitar has six strings tuned to E, A, D, G, B, and E", "category": "music"},
    {"content": "Vinyl records store audio as grooves in a spiral track on a flat disc", "category": "music"},
    {"content": "The blues originated in the Deep South of the United States in the 1860s", "category": "music"},

    # --- Food (10) ---
    {"content": "Sushi originated in Japan as a method of preserving fish in fermented rice", "category": "food"},
    {"content": "The avocado is technically a berry that originated in south-central Mexico", "category": "food"},
    {"content": "Sourdough bread uses a natural fermentation process with wild yeast and lactobacilli", "category": "food"},
    {"content": "Chocolate is made from cacao beans which are native to Central and South America", "category": "food"},
    {"content": "Parmesan cheese must be aged for at least 12 months by Italian law", "category": "food"},
    {"content": "Olive oil has been produced in the Mediterranean region for over 6000 years", "category": "food"},
    {"content": "Kimchi is a traditional Korean dish of fermented vegetables, usually napa cabbage", "category": "food"},
    {"content": "Coffee beans are actually the seeds of the coffee cherry fruit", "category": "food"},
    {"content": "Pasta was popularized in Italy but similar noodles existed independently in China", "category": "food"},
    {"content": "Vanilla is the second most expensive spice in the world after saffron", "category": "food"},

    # --- Sports (10) ---
    {"content": "The modern Olympic Games were first held in Athens, Greece in 1896", "category": "sports"},
    {"content": "A regulation basketball court is 94 feet long and 50 feet wide", "category": "sports"},
    {"content": "The FIFA World Cup is held every four years and is the most watched sporting event", "category": "sports"},
    {"content": "Tennis scoring uses the sequence love, 15, 30, 40 for points within a game", "category": "sports"},
    {"content": "The marathon distance of 26.2 miles was standardized at the 1908 London Olympics", "category": "sports"},
    {"content": "Cricket is the second most popular sport globally with over 2.5 billion fans", "category": "sports"},
    {"content": "Ice hockey was invented in Canada in the 19th century and uses a vulcanized rubber puck", "category": "sports"},
    {"content": "The Tour de France is an annual bicycle race covering approximately 3500 kilometers", "category": "sports"},
    {"content": "Usain Bolt set the 100m world record of 9.58 seconds in Berlin in 2009", "category": "sports"},
    {"content": "A standard soccer match consists of two 45-minute halves", "category": "sports"},

    # --- Literature (10) ---
    {"content": "Shakespeare wrote approximately 37 plays including Hamlet, Macbeth, and Romeo and Juliet", "category": "literature"},
    {"content": "The novel 1984 by George Orwell describes a dystopian totalitarian surveillance state", "category": "literature"},
    {"content": "Homer's Iliad and Odyssey are among the oldest works of Western literature", "category": "literature"},
    {"content": "Gabriel Garcia Marquez wrote One Hundred Years of Solitude pioneering magical realism", "category": "literature"},
    {"content": "The Gutenberg Bible printed around 1455 was the first major book printed in Europe", "category": "literature"},
    {"content": "Jane Austen wrote Pride and Prejudice exploring themes of class and marriage in Regency England", "category": "literature"},
    {"content": "Franz Kafka's The Metamorphosis begins with the protagonist waking up as a giant insect", "category": "literature"},
    {"content": "Leo Tolstoy's War and Peace is set during Napoleon's invasion of Russia", "category": "literature"},
    {"content": "Toni Morrison won the Nobel Prize in Literature in 1993 for her body of work", "category": "literature"},
    {"content": "The Epic of Gilgamesh from ancient Mesopotamia is one of the earliest known literary works", "category": "literature"},

    # --- Technology (10) ---
    {"content": "The first iPhone was released by Apple in June 2007 and revolutionized smartphones", "category": "technology"},
    {"content": "ARPANET, the precursor to the internet, sent its first message in 1969", "category": "technology"},
    {"content": "Transistors replaced vacuum tubes in computers leading to smaller and faster machines", "category": "technology"},
    {"content": "GPS satellites orbit Earth at approximately 20,200 kilometers altitude", "category": "technology"},
    {"content": "Bluetooth technology is named after Harald Bluetooth, a 10th-century Danish king", "category": "technology"},
    {"content": "The World Wide Web was invented by Tim Berners-Lee at CERN in 1989", "category": "technology"},
    {"content": "Moore's Law predicted that transistor density on chips doubles approximately every two years", "category": "technology"},
    {"content": "Quantum computers use qubits that can exist in superposition of states", "category": "technology"},
    {"content": "The first programmable computer, the Z3, was built by Konrad Zuse in 1941", "category": "technology"},
    {"content": "5G networks promise speeds up to 20 gigabits per second and low latency", "category": "technology"},

    # --- Mathematics (10) ---
    {"content": "Pi is approximately 3.14159 and represents the ratio of a circle's circumference to its diameter", "category": "mathematics"},
    {"content": "The Pythagorean theorem states that a squared plus b squared equals c squared for right triangles", "category": "mathematics"},
    {"content": "Zero was first used as a number by ancient Indian mathematicians around the 5th century", "category": "mathematics"},
    {"content": "Euler's identity combines five fundamental constants: e to the i pi plus one equals zero", "category": "mathematics"},
    {"content": "The Fibonacci sequence starts 0, 1, 1, 2, 3, 5, 8 where each number is the sum of the two before it", "category": "mathematics"},
    {"content": "Prime numbers are natural numbers greater than 1 that are only divisible by 1 and themselves", "category": "mathematics"},
    {"content": "Calculus was independently developed by Isaac Newton and Gottfried Wilhelm Leibniz", "category": "mathematics"},
    {"content": "The Riemann Hypothesis about the distribution of prime numbers remains unproven", "category": "mathematics"},
    {"content": "A Mobius strip is a surface with only one side and one boundary curve", "category": "mathematics"},
    {"content": "Boolean algebra uses true and false values and forms the basis of digital circuit design", "category": "mathematics"},
]

assert len(FACTS) == 100, f"Expected 100 facts, got {len(FACTS)}"

# --- Retrieval tasks ---

EXACT_QUERIES: list[dict] = [
    {
        "query": "What is the capital of France?",
        "expected": [["capital", "France", "Paris"]],
    },
    {
        "query": "What is the speed of light?",
        "expected": [["speed", "light", "299"]],
    },
    {
        "query": "Who created Python?",
        "expected": [["Python", "Guido"]],
    },
    {
        "query": "When did World War II end?",
        "expected": [["World War II", "1945"]],
    },
    {
        "query": "How many keys does a piano have?",
        "expected": [["piano", "88"]],
    },
]

SEMANTIC_QUERIES: list[dict] = [
    {
        "query": "Which country has Paris as its capital?",
        "expected": [["capital", "France", "Paris"]],
    },
    {
        "query": "What molecule carries genetic information in living organisms?",
        "expected": [["DNA", "genetic"]],
    },
    {
        "query": "Which version control system did the Linux creator make?",
        "expected": [["Git", "Linus Torvalds"]],
    },
    {
        "query": "What ancient document limited the English monarchy's power?",
        "expected": [["Magna Carta", "1215"]],
    },
    {
        "query": "What Korean fermented vegetable dish uses cabbage?",
        "expected": [["Kimchi", "Korean", "fermented"]],
    },
]

CROSS_CATEGORY_QUERIES: list[dict] = [
    {
        "query": "Name something invented or created by Linus Torvalds",
        "expected": [["Git", "Linus Torvalds"], ["Linux", "Linus Torvalds"]],
    },
    {
        "query": "What originated in the late 18th century in Britain?",
        "expected": [["Industrial Revolution", "Britain"]],
    },
    {
        "query": "What was first made or popularized in Italy?",
        "expected": [["Pasta", "Italy"], ["Renaissance", "Italy"]],
    },
    {
        "query": "What is named after a historical Danish figure?",
        "expected": [["Bluetooth", "Danish"]],
    },
    {
        "query": "What involves fermentation in food preparation?",
        "expected": [["Sourdough", "fermentation"], ["Kimchi", "fermented"], ["Sushi", "fermented"]],
    },
]


# ====================================================================
# CATEGORY 2: Episodic → Semantic Consolidation (10 tasks)
# ====================================================================

EPISODIC_MEMORIES: list[dict] = [
    # --- Coffee orders (6 episodes: 4 latte, 1 cappuccino, 1 tea) ---
    {"content": "Monday morning: user ordered a vanilla latte from the cafe", "category": "coffee_orders"},
    {"content": "Tuesday: user got their usual latte with oat milk", "category": "coffee_orders"},
    {"content": "Wednesday: user ordered a latte, extra hot, no sugar", "category": "coffee_orders"},
    {"content": "Thursday: user tried a cappuccino instead of their usual", "category": "coffee_orders"},
    {"content": "Friday: user ordered a large latte to go", "category": "coffee_orders"},
    {"content": "Saturday: user ordered green tea because they wanted something lighter", "category": "coffee_orders"},

    # --- Meeting preferences (6 episodes: 4 morning, 1 afternoon, 1 async) ---
    {"content": "User scheduled the design review for 9 AM, said they prefer morning meetings", "category": "meeting_preferences"},
    {"content": "User moved the standup to 10 AM, mentioned mornings work best for focus", "category": "meeting_preferences"},
    {"content": "User booked the sprint planning at 9:30 AM on Monday", "category": "meeting_preferences"},
    {"content": "User had to take a 3 PM meeting due to timezone constraints with the London team", "category": "meeting_preferences"},
    {"content": "User blocked 8 AM to 11 AM for meetings on their calendar", "category": "meeting_preferences"},
    {"content": "User asked to replace the Friday sync with an async Slack update", "category": "meeting_preferences"},

    # --- Coding habits (6 episodes: 4 Python, 1 TypeScript, 1 Rust) ---
    {"content": "User wrote a data processing script in Python using pandas", "category": "coding_habits"},
    {"content": "User built a REST API prototype in Python with FastAPI", "category": "coding_habits"},
    {"content": "User created a command-line tool in Python with click library", "category": "coding_habits"},
    {"content": "User implemented the frontend component in TypeScript with React", "category": "coding_habits"},
    {"content": "User wrote a Python script to automate the deployment pipeline", "category": "coding_habits"},
    {"content": "User experimented with Rust for a performance-critical module", "category": "coding_habits"},

    # --- Communication style (6 episodes: 4 concise/bullet, 1 detailed, 1 visual) ---
    {"content": "User sent a status update as three bullet points, very concise", "category": "communication_style"},
    {"content": "User wrote the project summary using short bullet-point format", "category": "communication_style"},
    {"content": "User responded to the question with a brief two-sentence answer", "category": "communication_style"},
    {"content": "User wrote a detailed three-page design document for the architecture review", "category": "communication_style"},
    {"content": "User preferred bullet points for the weekly report format", "category": "communication_style"},
    {"content": "User created a visual diagram instead of writing text for the system overview", "category": "communication_style"},

    # --- Work schedule (6 episodes: 4 early-start, 1 late-night, 1 weekend) ---
    {"content": "User started working at 7 AM and finished by 4 PM", "category": "work_schedule"},
    {"content": "User logged in at 6:30 AM to finish the report before standup", "category": "work_schedule"},
    {"content": "User began coding at 7:15 AM, said early mornings are most productive", "category": "work_schedule"},
    {"content": "User stayed late until 11 PM to fix a critical production bug", "category": "work_schedule"},
    {"content": "User started the day at 7 AM and took a long lunch break", "category": "work_schedule"},
    {"content": "User worked on Saturday to meet the release deadline", "category": "work_schedule"},
]

assert len(EPISODIC_MEMORIES) == 30, f"Expected 30 episodic memories, got {len(EPISODIC_MEMORIES)}"

# Generalization queries: expect the majority pattern
GENERALIZATION_QUERIES: list[dict] = [
    {
        "question": "What does the user typically order for coffee?",
        "expected_keywords": ["latte"],
        "category_hint": "coffee_orders",
    },
    {
        "question": "When does the user prefer to have meetings?",
        "expected_keywords": ["morning"],
        "category_hint": "meeting_preferences",
    },
    {
        "question": "What programming language does the user primarily use?",
        "expected_keywords": ["python"],
        "category_hint": "coding_habits",
    },
    {
        "question": "How does the user prefer to communicate information?",
        "expected_keywords": ["bullet"],
        "category_hint": "communication_style",
    },
    {
        "question": "When does the user typically start their work day?",
        "expected_keywords": ["7", "early"],
        "category_hint": "work_schedule",
    },
]

# Specificity queries: expect the minority event to be preserved
SPECIFICITY_QUERIES: list[dict] = [
    {
        "question": "Has the user ever ordered tea?",
        "expected_keywords": ["tea"],
        "category_hint": "coffee_orders",
    },
    {
        "question": "Has the user ever preferred async communication over meetings?",
        "expected_keywords": ["async"],
        "category_hint": "meeting_preferences",
    },
    {
        "question": "Has the user ever written code in Rust?",
        "expected_keywords": ["rust"],
        "category_hint": "coding_habits",
    },
    {
        "question": "Has the user ever written a detailed long-form document?",
        "expected_keywords": ["detailed", "page"],
        "category_hint": "communication_style",
    },
    {
        "question": "Has the user ever worked late at night?",
        "expected_keywords": ["11 pm", "late"],
        "category_hint": "work_schedule",
    },
]


# ====================================================================
# CATEGORY 3: Retrieval Under Scale (5 tasks)
# ====================================================================

def _generate_scale_memories(n: int) -> list[dict]:
    """Generate n memories across diverse topics for scale testing.

    Uses deterministic generation (no randomness) so results are reproducible.
    """
    categories = ["tech_news", "personal_notes", "work_tasks", "learning", "health"]
    templates = [
        # tech_news
        [
            "Released version {v}.0 of the data processing framework with improved performance",
            "New vulnerability CVE-2024-{v}001 discovered in popular web library",
            "Cloud provider announced {v}x faster GPU instances for machine learning",
            "Open-source project reached {v}0,000 GitHub stars milestone",
            "Browser update {v}.0 brings improved JavaScript engine performance",
        ],
        # personal_notes
        [
            "Had a productive brainstorming session about project {v} architecture",
            "Interesting article about cognitive biases in decision making, variant {v}",
            "Idea for project {v}: build an automated testing framework for APIs",
            "Reflection: need to improve time management, iteration {v}",
            "Learned about distributed systems consensus protocol version {v}",
        ],
        # work_tasks
        [
            "Completed code review for pull request #{v}42 on the authentication module",
            "Deployed hotfix {v}.1 to production for the payment service",
            "Wrote technical spec {v} for the new caching layer design",
            "Fixed bug #{v}99 in the search indexing pipeline",
            "Migrated database schema to version {v} with zero downtime",
        ],
        # learning
        [
            "Studied chapter {v} of the distributed systems textbook on replication",
            "Completed exercise {v} on dynamic programming algorithms",
            "Watched lecture {v} on transformer architecture and attention mechanisms",
            "Practiced system design interview question {v}: design a URL shortener",
            "Read paper {v} on reinforcement learning from human feedback",
        ],
        # health
        [
            "Morning run: {v} kilometers in 30 minutes, felt energetic",
            "Slept {v} hours last night, need to maintain consistent schedule",
            "Tried new recipe {v}: Mediterranean salad with quinoa",
            "Meditation session {v}: 15 minutes of focused breathing",
            "Walked {v}0,000 steps today exploring the neighborhood",
        ],
    ]

    memories: list[dict] = []
    for i in range(n):
        cat_idx = i % len(categories)
        tmpl_idx = (i // len(categories)) % len(templates[cat_idx])
        v = (i // (len(categories) * 5)) + 1
        content = templates[cat_idx][tmpl_idx].format(v=v)
        memories.append({"content": content, "category": categories[cat_idx]})
    return memories


# Scale test queries — things that should be findable at any scale
SCALE_QUERIES: list[dict] = [
    {
        "query": "What code reviews were completed?",
        "expected_keywords": [["code review", "pull request"]],
    },
    {
        "query": "Tell me about distributed systems learning",
        "expected_keywords": [["distributed systems"]],
    },
    {
        "query": "What exercise or meditation was done?",
        "expected_keywords": [["meditation"], ["exercise"]],
    },
    {
        "query": "What was deployed to production?",
        "expected_keywords": [["deployed", "production"]],
    },
    {
        "query": "What papers or lectures were studied?",
        "expected_keywords": [["lecture"], ["paper"]],
    },
]


# ====================================================================
# CATEGORY 4: Knowledge Integration — Multi-hop (10 tasks)
# ====================================================================

# Fact chains: each chain is a list of facts plus a multi-hop question
MULTI_HOP_CHAINS: list[dict] = [
    {
        "facts": [
            {"content": "Python was created by Guido van Rossum", "category": "people"},
            {"content": "Guido van Rossum joined Microsoft in November 2020", "category": "people"},
            {"content": "Microsoft is headquartered in Redmond, Washington", "category": "companies"},
        ],
        "question": "Where does the creator of Python work?",
        "expected_keywords": ["Microsoft"],
        "hops": 2,
    },
    {
        "facts": [
            {"content": "The Eiffel Tower is located in Paris", "category": "landmarks"},
            {"content": "Paris is the capital of France", "category": "geography"},
            {"content": "France is a founding member of the European Union", "category": "politics"},
        ],
        "question": "Is the country where the Eiffel Tower is located a member of the EU?",
        "expected_keywords": ["France", "European Union"],
        "hops": 3,
    },
    {
        "facts": [
            {"content": "TensorFlow was developed by Google Brain team", "category": "tech"},
            {"content": "Google Brain was founded by Andrew Ng and Jeff Dean", "category": "tech"},
        ],
        "question": "Who founded the team that developed TensorFlow?",
        "expected_keywords": ["Andrew Ng", "Jeff Dean"],
        "hops": 2,
    },
    {
        "facts": [
            {"content": "The Great Wall of China was primarily built during the Ming Dynasty", "category": "history"},
            {"content": "The Ming Dynasty ruled China from 1368 to 1644", "category": "history"},
            {"content": "The Ming capital was Beijing, then called Peking", "category": "history"},
        ],
        "question": "What dynasty built the Great Wall and when did they rule?",
        "expected_keywords": ["Ming", "1368"],
        "hops": 2,
    },
    {
        "facts": [
            {"content": "CRISPR gene editing was developed by Jennifer Doudna and Emmanuelle Charpentier", "category": "science"},
            {"content": "Jennifer Doudna is a professor at UC Berkeley", "category": "people"},
        ],
        "question": "Where does one of the developers of CRISPR teach?",
        "expected_keywords": ["Berkeley"],
        "hops": 2,
    },
    {
        "facts": [
            {"content": "SpaceX was founded by Elon Musk in 2002", "category": "companies"},
            {"content": "SpaceX developed the Falcon 9 reusable rocket", "category": "tech"},
            {"content": "Falcon 9 first successfully landed a booster stage in December 2015", "category": "tech"},
        ],
        "question": "When did the rocket made by Elon Musk's space company first land successfully?",
        "expected_keywords": ["2015"],
        "hops": 3,
    },
    {
        "facts": [
            {"content": "The Nobel Prize in Physics 2024 was awarded for work on neural networks", "category": "science"},
            {"content": "Geoffrey Hinton received the 2024 Nobel Prize in Physics", "category": "people"},
            {"content": "Geoffrey Hinton is known as one of the godfathers of deep learning", "category": "people"},
        ],
        "question": "What is the 2024 Physics Nobel laureate known as?",
        "expected_keywords": ["godfather", "deep learning"],
        "hops": 2,
    },
    {
        "facts": [
            {"content": "Kubernetes was originally designed by Google engineers", "category": "tech"},
            {"content": "Kubernetes is now maintained by the Cloud Native Computing Foundation", "category": "tech"},
            {"content": "The Cloud Native Computing Foundation is part of the Linux Foundation", "category": "organizations"},
        ],
        "question": "What foundation oversees the organization that maintains Kubernetes?",
        "expected_keywords": ["Linux Foundation"],
        "hops": 3,
    },
    {
        "facts": [
            {"content": "The Mona Lisa was painted by Leonardo da Vinci", "category": "art"},
            {"content": "The Mona Lisa is displayed in the Louvre Museum", "category": "art"},
            {"content": "The Louvre Museum is located in Paris, France", "category": "landmarks"},
        ],
        "question": "In what city can you see the painting by Leonardo da Vinci?",
        "expected_keywords": ["Paris"],
        "hops": 3,
    },
    {
        "facts": [
            {"content": "TypeScript was created by Anders Hejlsberg at Microsoft", "category": "programming"},
            {"content": "Anders Hejlsberg also designed the C# programming language", "category": "programming"},
        ],
        "question": "What other language did the creator of TypeScript design?",
        "expected_keywords": ["C#"],
        "hops": 2,
    },
]

assert len(MULTI_HOP_CHAINS) == 10, f"Expected 10 multi-hop chains, got {len(MULTI_HOP_CHAINS)}"


# ====================================================================
# Benchmark runners for each category
# ====================================================================

def _run_retrieval_benchmark(ltm: LongTermMemory) -> float:
    """Category 1: Storage and Retrieval. Returns F1 score (avg of R@5 and P@5)."""
    # Store all 100 facts
    for fact in FACTS:
        ltm.store(fact["content"], category=fact["category"])

    all_scores: list[float] = []

    # 5 exact-match queries
    for q in EXACT_QUERIES:
        results = ltm.retrieve(q["query"], k=5)
        recall = _recall_at_k(results, q["expected"], k=5)
        precision = _precision_at_k(results, q["expected"], k=5)
        f1 = 2 * recall * precision / max(recall + precision, 1e-9)
        all_scores.append(f1)

    # 5 semantic queries
    for q in SEMANTIC_QUERIES:
        results = ltm.retrieve(q["query"], k=5)
        recall = _recall_at_k(results, q["expected"], k=5)
        precision = _precision_at_k(results, q["expected"], k=5)
        f1 = 2 * recall * precision / max(recall + precision, 1e-9)
        all_scores.append(f1)

    # 5 cross-category queries
    for q in CROSS_CATEGORY_QUERIES:
        results = ltm.retrieve(q["query"], k=5)
        recall = _recall_at_k(results, q["expected"], k=5)
        precision = _precision_at_k(results, q["expected"], k=5)
        f1 = 2 * recall * precision / max(recall + precision, 1e-9)
        all_scores.append(f1)

    return sum(all_scores) / max(len(all_scores), 1)


def _run_consolidation_benchmark(ltm: LongTermMemory) -> float:
    """Category 2: Episodic to Semantic Consolidation."""
    # Store episodic memories
    for mem in EPISODIC_MEMORIES:
        ltm.store(mem["content"], category=mem["category"])

    # Trigger consolidation
    ltm.consolidate()

    generalization_scores: list[float] = []
    specificity_scores: list[float] = []

    # Generalization queries — use query_knowledge
    for q in GENERALIZATION_QUERIES:
        answer = ltm.query_knowledge(q["question"])
        answer_lower = answer.lower()
        # Check if expected keywords appear in the answer
        found = any(kw.lower() in answer_lower for kw in q["expected_keywords"])
        generalization_scores.append(1.0 if found else 0.0)

    # Specificity queries — minority events should be retrievable
    for q in SPECIFICITY_QUERIES:
        # Use both query_knowledge and retrieve to give the system every chance
        answer = ltm.query_knowledge(q["question"])
        results = ltm.retrieve(q["question"], k=5)
        answer_lower = answer.lower()
        result_texts = " ".join(r.get("content", "").lower() for r in results)
        combined = answer_lower + " " + result_texts
        found = any(kw.lower() in combined for kw in q["expected_keywords"])
        specificity_scores.append(1.0 if found else 0.0)

    gen_avg = sum(generalization_scores) / max(len(generalization_scores), 1)
    spec_avg = sum(specificity_scores) / max(len(specificity_scores), 1)

    # Weight: 60% generalization (the hard part), 40% specificity
    return 0.6 * gen_avg + 0.4 * spec_avg


def _run_scale_benchmark() -> float:
    """Category 3: Retrieval Under Scale.

    Tests retrieval quality at 50, 100, 200, and 500 memories.
    Returns stability score = 1 - coefficient_of_variation of recall across scales.
    """
    scale_points = [50, 100, 200, 500]
    scale_memories = _generate_scale_memories(500)

    recalls_by_scale: list[float] = []

    for n in scale_points:
        ltm = LongTermMemory()
        for mem in scale_memories[:n]:
            ltm.store(mem["content"], category=mem["category"])

        # Measure recall across all scale queries
        query_recalls: list[float] = []
        for q in SCALE_QUERIES:
            results = ltm.retrieve(q["query"], k=5)
            recall = _recall_at_k(results, q["expected_keywords"], k=5)
            query_recalls.append(recall)

        avg_recall = sum(query_recalls) / max(len(query_recalls), 1)
        recalls_by_scale.append(avg_recall)

    # Stability = 1 - coefficient of variation (stddev / mean)
    if not recalls_by_scale:
        return 0.0
    mean_recall = sum(recalls_by_scale) / len(recalls_by_scale)
    if mean_recall < 1e-9:
        # If recall is zero everywhere, that's stable but useless — score 0
        return 0.0
    variance = sum((r - mean_recall) ** 2 for r in recalls_by_scale) / len(recalls_by_scale)
    stddev = variance ** 0.5
    cv = stddev / mean_recall

    # Combine: base recall quality + stability
    stability = max(0.0, 1.0 - cv)

    # Blend: 50% mean recall (must actually retrieve things) + 50% stability
    return 0.5 * mean_recall + 0.5 * stability


def _run_multihop_benchmark() -> float:
    """Category 4: Knowledge Integration — multi-hop reasoning."""
    ltm = LongTermMemory()

    # Store all facts from all chains
    for chain in MULTI_HOP_CHAINS:
        for fact in chain["facts"]:
            ltm.store(fact["content"], category=fact["category"])

    # Run consolidation to let the system build connections
    ltm.consolidate()

    scores: list[float] = []

    for chain in MULTI_HOP_CHAINS:
        # Use query_knowledge for the integrated answer
        answer = ltm.query_knowledge(chain["question"])

        # Also try retrieve as fallback
        results = ltm.retrieve(chain["question"], k=10)
        result_texts = " ".join(r.get("content", "").lower() for r in results)
        combined = answer.lower() + " " + result_texts

        # Check if expected keywords are found
        found = all(kw.lower() in combined for kw in chain["expected_keywords"])
        scores.append(1.0 if found else 0.0)

    return sum(scores) / max(len(scores), 1)


# ====================================================================
# Main benchmark
# ====================================================================

def run_benchmark() -> float:
    """Run the complete LTM benchmark and print the fitness score."""

    # ── Category 1: Storage & Retrieval ──────────────────────────────
    ltm1 = LongTermMemory()
    t0 = time.time()
    retrieval_f1 = _run_retrieval_benchmark(ltm1)
    t_retrieval = time.time() - t0

    # ── Category 2: Consolidation ────────────────────────────────────
    ltm2 = LongTermMemory()
    t0 = time.time()
    consolidation_score = _run_consolidation_benchmark(ltm2)
    t_consolidation = time.time() - t0

    # ── Category 3: Scale stability ──────────────────────────────────
    t0 = time.time()
    scale_stability = _run_scale_benchmark()
    t_scale = time.time() - t0

    # ── Category 4: Multi-hop ────────────────────────────────────────
    t0 = time.time()
    multi_hop_accuracy = _run_multihop_benchmark()
    t_multihop = time.time() - t0

    # ── Latency component ────────────────────────────────────────────
    total_time = t_retrieval + t_consolidation + t_scale + t_multihop
    # Normalize: 0 at <= 0.5s, 1 at >= 10s
    latency_normalized = min(1.0, max(0.0, (total_time - 0.5) / 9.5))

    # ── Composite fitness ────────────────────────────────────────────
    fitness = (
        0.30 * retrieval_f1
        + 0.25 * consolidation_score
        + 0.20 * multi_hop_accuracy
        + 0.15 * scale_stability
        + 0.10 * (1.0 - latency_normalized)
    )

    # ── Report ───────────────────────────────────────────────────────
    print(f"  retrieval_f1:       {retrieval_f1:.6f}  (weight 0.30)")
    print(f"  consolidation:      {consolidation_score:.6f}  (weight 0.25)")
    print(f"  multi_hop:          {multi_hop_accuracy:.6f}  (weight 0.20)")
    print(f"  scale_stability:    {scale_stability:.6f}  (weight 0.15)")
    print(f"  latency_norm:       {latency_normalized:.6f}  (weight 0.10, inverted)")
    print(f"  total_time:         {total_time:.3f}s")
    print(f"score: {fitness:.6f}")

    return fitness


if __name__ == "__main__":
    run_benchmark()
