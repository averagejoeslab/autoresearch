#!/usr/bin/env python3
"""
Benchmark: Memory Retrieval
============================
Evaluates MemoryStore on 30 retrieval tasks across three categories:
  - Exact-match queries (10)
  - Keyword-overlap / semantic-similarity queries (10)
  - Metadata-filtered queries (10)

Stores 100 facts, then measures recall@5, precision@5, and latency.

Fitness = 0.5 * recall@5 + 0.3 * precision@5 + 0.2 * (1 - latency_norm)

Run:  python benchmark.py
"""

from __future__ import annotations

import os
import sys
import time

# ---------------------------------------------------------------------------
# Path setup so we can import from contracts/
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from harness import MemoryStore  # noqa: E402

# ====================================================================
# Inline dataset: 100 facts
# ====================================================================
FACTS: list[dict] = [
    {"key": "capital_france", "value": "The capital of France is Paris", "metadata": {"topic": "geography", "importance": 0.9, "region": "europe"}},
    {"key": "capital_germany", "value": "The capital of Germany is Berlin", "metadata": {"topic": "geography", "importance": 0.9, "region": "europe"}},
    {"key": "capital_japan", "value": "The capital of Japan is Tokyo", "metadata": {"topic": "geography", "importance": 0.9, "region": "asia"}},
    {"key": "capital_brazil", "value": "The capital of Brazil is Brasilia", "metadata": {"topic": "geography", "importance": 0.8, "region": "south_america"}},
    {"key": "capital_australia", "value": "The capital of Australia is Canberra", "metadata": {"topic": "geography", "importance": 0.8, "region": "oceania"}},
    {"key": "capital_canada", "value": "The capital of Canada is Ottawa", "metadata": {"topic": "geography", "importance": 0.8, "region": "north_america"}},
    {"key": "capital_india", "value": "The capital of India is New Delhi", "metadata": {"topic": "geography", "importance": 0.9, "region": "asia"}},
    {"key": "capital_italy", "value": "The capital of Italy is Rome", "metadata": {"topic": "geography", "importance": 0.8, "region": "europe"}},
    {"key": "capital_mexico", "value": "The capital of Mexico is Mexico City", "metadata": {"topic": "geography", "importance": 0.8, "region": "north_america"}},
    {"key": "capital_egypt", "value": "The capital of Egypt is Cairo", "metadata": {"topic": "geography", "importance": 0.8, "region": "africa"}},
    {"key": "planet_mercury", "value": "Mercury is the closest planet to the Sun", "metadata": {"topic": "astronomy", "importance": 0.7, "region": "space"}},
    {"key": "planet_venus", "value": "Venus is the hottest planet in the solar system", "metadata": {"topic": "astronomy", "importance": 0.7, "region": "space"}},
    {"key": "planet_mars", "value": "Mars is known as the Red Planet", "metadata": {"topic": "astronomy", "importance": 0.8, "region": "space"}},
    {"key": "planet_jupiter", "value": "Jupiter is the largest planet in the solar system", "metadata": {"topic": "astronomy", "importance": 0.7, "region": "space"}},
    {"key": "planet_saturn", "value": "Saturn is famous for its rings", "metadata": {"topic": "astronomy", "importance": 0.7, "region": "space"}},
    {"key": "element_hydrogen", "value": "Hydrogen is the lightest element with atomic number 1", "metadata": {"topic": "chemistry", "importance": 0.9, "region": "science"}},
    {"key": "element_helium", "value": "Helium is the second lightest element used in balloons", "metadata": {"topic": "chemistry", "importance": 0.7, "region": "science"}},
    {"key": "element_oxygen", "value": "Oxygen is essential for breathing and has atomic number 8", "metadata": {"topic": "chemistry", "importance": 0.9, "region": "science"}},
    {"key": "element_carbon", "value": "Carbon is the basis of organic chemistry", "metadata": {"topic": "chemistry", "importance": 0.9, "region": "science"}},
    {"key": "element_gold", "value": "Gold is a precious metal with atomic number 79", "metadata": {"topic": "chemistry", "importance": 0.7, "region": "science"}},
    {"key": "lang_python", "value": "Python is a popular programming language created by Guido van Rossum", "metadata": {"topic": "programming", "importance": 0.9, "region": "tech"}},
    {"key": "lang_javascript", "value": "JavaScript is the language of the web and runs in browsers", "metadata": {"topic": "programming", "importance": 0.9, "region": "tech"}},
    {"key": "lang_rust", "value": "Rust is a systems programming language focused on safety", "metadata": {"topic": "programming", "importance": 0.8, "region": "tech"}},
    {"key": "lang_java", "value": "Java is a widely used object-oriented programming language", "metadata": {"topic": "programming", "importance": 0.8, "region": "tech"}},
    {"key": "lang_go", "value": "Go is a programming language created at Google for concurrency", "metadata": {"topic": "programming", "importance": 0.7, "region": "tech"}},
    {"key": "animal_elephant", "value": "The elephant is the largest living land animal", "metadata": {"topic": "biology", "importance": 0.7, "region": "nature"}},
    {"key": "animal_whale", "value": "The blue whale is the largest animal ever known", "metadata": {"topic": "biology", "importance": 0.8, "region": "nature"}},
    {"key": "animal_cheetah", "value": "The cheetah is the fastest land animal reaching 70 mph", "metadata": {"topic": "biology", "importance": 0.7, "region": "nature"}},
    {"key": "animal_dolphin", "value": "Dolphins are intelligent marine mammals known for echolocation", "metadata": {"topic": "biology", "importance": 0.7, "region": "nature"}},
    {"key": "animal_eagle", "value": "Eagles are birds of prey with excellent vision", "metadata": {"topic": "biology", "importance": 0.6, "region": "nature"}},
    {"key": "math_pi", "value": "Pi is approximately 3.14159 and represents the ratio of circumference to diameter", "metadata": {"topic": "math", "importance": 0.9, "region": "science"}},
    {"key": "math_euler", "value": "Euler number e is approximately 2.71828 and is the base of natural logarithms", "metadata": {"topic": "math", "importance": 0.8, "region": "science"}},
    {"key": "math_pythagorean", "value": "The Pythagorean theorem states a squared plus b squared equals c squared", "metadata": {"topic": "math", "importance": 0.9, "region": "science"}},
    {"key": "math_prime", "value": "A prime number is only divisible by 1 and itself", "metadata": {"topic": "math", "importance": 0.8, "region": "science"}},
    {"key": "math_fibonacci", "value": "The Fibonacci sequence starts 0 1 1 2 3 5 8 13 where each number is the sum of the previous two", "metadata": {"topic": "math", "importance": 0.7, "region": "science"}},
    {"key": "history_rome", "value": "The Roman Empire fell in 476 AD", "metadata": {"topic": "history", "importance": 0.8, "region": "europe"}},
    {"key": "history_moon", "value": "The first moon landing was in 1969 by Apollo 11", "metadata": {"topic": "history", "importance": 0.9, "region": "space"}},
    {"key": "history_internet", "value": "The internet originated from ARPANET in 1969", "metadata": {"topic": "history", "importance": 0.8, "region": "tech"}},
    {"key": "history_printing", "value": "The printing press was invented by Gutenberg around 1440", "metadata": {"topic": "history", "importance": 0.8, "region": "europe"}},
    {"key": "history_electricity", "value": "Benjamin Franklin demonstrated electricity with a kite in 1752", "metadata": {"topic": "history", "importance": 0.7, "region": "north_america"}},
    {"key": "food_sushi", "value": "Sushi is a Japanese dish made with vinegared rice and seafood", "metadata": {"topic": "food", "importance": 0.6, "region": "asia"}},
    {"key": "food_pizza", "value": "Pizza originated in Naples Italy with tomato sauce and mozzarella", "metadata": {"topic": "food", "importance": 0.6, "region": "europe"}},
    {"key": "food_taco", "value": "Tacos are a traditional Mexican dish with a folded tortilla", "metadata": {"topic": "food", "importance": 0.6, "region": "north_america"}},
    {"key": "food_curry", "value": "Curry is a spiced dish popular in India and Southeast Asia", "metadata": {"topic": "food", "importance": 0.6, "region": "asia"}},
    {"key": "food_pasta", "value": "Pasta is a staple of Italian cuisine made from wheat flour", "metadata": {"topic": "food", "importance": 0.6, "region": "europe"}},
    {"key": "music_beethoven", "value": "Beethoven composed nine symphonies and was deaf in later life", "metadata": {"topic": "music", "importance": 0.8, "region": "europe"}},
    {"key": "music_mozart", "value": "Mozart was a prolific composer from Austria who lived only 35 years", "metadata": {"topic": "music", "importance": 0.8, "region": "europe"}},
    {"key": "music_jazz", "value": "Jazz music originated in New Orleans in the early 20th century", "metadata": {"topic": "music", "importance": 0.7, "region": "north_america"}},
    {"key": "music_rock", "value": "Rock and roll emerged in the 1950s combining blues and country", "metadata": {"topic": "music", "importance": 0.7, "region": "north_america"}},
    {"key": "music_hiphop", "value": "Hip hop originated in the Bronx New York in the 1970s", "metadata": {"topic": "music", "importance": 0.7, "region": "north_america"}},
    {"key": "sport_soccer", "value": "Soccer or football is the most popular sport in the world", "metadata": {"topic": "sports", "importance": 0.8, "region": "global"}},
    {"key": "sport_basketball", "value": "Basketball was invented by James Naismith in 1891", "metadata": {"topic": "sports", "importance": 0.7, "region": "north_america"}},
    {"key": "sport_tennis", "value": "Tennis Grand Slams are the Australian Open French Open Wimbledon and US Open", "metadata": {"topic": "sports", "importance": 0.7, "region": "global"}},
    {"key": "sport_cricket", "value": "Cricket is hugely popular in India England and Australia", "metadata": {"topic": "sports", "importance": 0.7, "region": "global"}},
    {"key": "sport_swimming", "value": "Swimming has been an Olympic sport since 1896", "metadata": {"topic": "sports", "importance": 0.6, "region": "global"}},
    {"key": "ocean_pacific", "value": "The Pacific Ocean is the largest and deepest ocean on Earth", "metadata": {"topic": "geography", "importance": 0.8, "region": "global"}},
    {"key": "ocean_atlantic", "value": "The Atlantic Ocean separates the Americas from Europe and Africa", "metadata": {"topic": "geography", "importance": 0.7, "region": "global"}},
    {"key": "mountain_everest", "value": "Mount Everest is the tallest mountain at 8849 meters", "metadata": {"topic": "geography", "importance": 0.9, "region": "asia"}},
    {"key": "river_nile", "value": "The Nile is one of the longest rivers flowing through northeast Africa", "metadata": {"topic": "geography", "importance": 0.7, "region": "africa"}},
    {"key": "river_amazon", "value": "The Amazon River carries more water than any other river", "metadata": {"topic": "geography", "importance": 0.7, "region": "south_america"}},
    {"key": "body_heart", "value": "The human heart beats about 100000 times per day", "metadata": {"topic": "biology", "importance": 0.8, "region": "science"}},
    {"key": "body_brain", "value": "The human brain contains about 86 billion neurons", "metadata": {"topic": "biology", "importance": 0.8, "region": "science"}},
    {"key": "body_bone", "value": "An adult human has 206 bones in their body", "metadata": {"topic": "biology", "importance": 0.7, "region": "science"}},
    {"key": "body_blood", "value": "Blood carries oxygen and nutrients throughout the body", "metadata": {"topic": "biology", "importance": 0.7, "region": "science"}},
    {"key": "body_lung", "value": "The lungs exchange oxygen and carbon dioxide during breathing", "metadata": {"topic": "biology", "importance": 0.7, "region": "science"}},
    {"key": "physics_gravity", "value": "Gravity is the force of attraction between masses described by Newton", "metadata": {"topic": "physics", "importance": 0.9, "region": "science"}},
    {"key": "physics_light", "value": "Light travels at approximately 300000 kilometers per second", "metadata": {"topic": "physics", "importance": 0.9, "region": "science"}},
    {"key": "physics_relativity", "value": "Einstein theory of relativity relates energy and mass via E equals mc squared", "metadata": {"topic": "physics", "importance": 0.9, "region": "science"}},
    {"key": "physics_quantum", "value": "Quantum mechanics describes behavior of particles at atomic scale", "metadata": {"topic": "physics", "importance": 0.8, "region": "science"}},
    {"key": "physics_entropy", "value": "Entropy measures the disorder of a system and always increases", "metadata": {"topic": "physics", "importance": 0.7, "region": "science"}},
    {"key": "inventor_edison", "value": "Thomas Edison invented the practical incandescent light bulb", "metadata": {"topic": "history", "importance": 0.8, "region": "north_america"}},
    {"key": "inventor_tesla", "value": "Nikola Tesla pioneered alternating current electricity systems", "metadata": {"topic": "history", "importance": 0.8, "region": "north_america"}},
    {"key": "inventor_wright", "value": "The Wright brothers achieved the first powered flight in 1903", "metadata": {"topic": "history", "importance": 0.8, "region": "north_america"}},
    {"key": "inventor_bell", "value": "Alexander Graham Bell invented the telephone in 1876", "metadata": {"topic": "history", "importance": 0.7, "region": "north_america"}},
    {"key": "inventor_curie", "value": "Marie Curie discovered radioactivity and won two Nobel Prizes", "metadata": {"topic": "history", "importance": 0.8, "region": "europe"}},
    {"key": "country_pop_china", "value": "China has the second largest population in the world", "metadata": {"topic": "demographics", "importance": 0.8, "region": "asia"}},
    {"key": "country_pop_india", "value": "India has the largest population in the world", "metadata": {"topic": "demographics", "importance": 0.8, "region": "asia"}},
    {"key": "country_pop_usa", "value": "The United States has the third largest population globally", "metadata": {"topic": "demographics", "importance": 0.7, "region": "north_america"}},
    {"key": "country_area_russia", "value": "Russia is the largest country by area spanning Europe and Asia", "metadata": {"topic": "geography", "importance": 0.8, "region": "europe"}},
    {"key": "country_area_canada", "value": "Canada is the second largest country by area", "metadata": {"topic": "geography", "importance": 0.7, "region": "north_america"}},
    {"key": "tech_ai", "value": "Artificial intelligence simulates human intelligence in machines", "metadata": {"topic": "technology", "importance": 0.9, "region": "tech"}},
    {"key": "tech_blockchain", "value": "Blockchain is a decentralized ledger technology behind cryptocurrencies", "metadata": {"topic": "technology", "importance": 0.7, "region": "tech"}},
    {"key": "tech_cloud", "value": "Cloud computing delivers computing services over the internet", "metadata": {"topic": "technology", "importance": 0.8, "region": "tech"}},
    {"key": "tech_5g", "value": "5G is the fifth generation of mobile network technology", "metadata": {"topic": "technology", "importance": 0.7, "region": "tech"}},
    {"key": "tech_iot", "value": "Internet of Things connects everyday devices to the internet", "metadata": {"topic": "technology", "importance": 0.7, "region": "tech"}},
    {"key": "lit_shakespeare", "value": "Shakespeare wrote plays like Hamlet Romeo and Juliet and Macbeth", "metadata": {"topic": "literature", "importance": 0.9, "region": "europe"}},
    {"key": "lit_tolstoy", "value": "Tolstoy wrote War and Peace one of the greatest novels", "metadata": {"topic": "literature", "importance": 0.8, "region": "europe"}},
    {"key": "lit_twain", "value": "Mark Twain wrote The Adventures of Huckleberry Finn", "metadata": {"topic": "literature", "importance": 0.7, "region": "north_america"}},
    {"key": "lit_orwell", "value": "George Orwell wrote 1984 a dystopian novel about totalitarianism", "metadata": {"topic": "literature", "importance": 0.8, "region": "europe"}},
    {"key": "lit_austen", "value": "Jane Austen wrote Pride and Prejudice about social manners", "metadata": {"topic": "literature", "importance": 0.7, "region": "europe"}},
    {"key": "eco_gdp", "value": "GDP measures the total economic output of a country", "metadata": {"topic": "economics", "importance": 0.8, "region": "global"}},
    {"key": "eco_inflation", "value": "Inflation is the rate at which prices increase over time", "metadata": {"topic": "economics", "importance": 0.8, "region": "global"}},
    {"key": "eco_supply", "value": "Supply and demand determine prices in a market economy", "metadata": {"topic": "economics", "importance": 0.8, "region": "global"}},
    {"key": "eco_trade", "value": "International trade allows countries to exchange goods and services", "metadata": {"topic": "economics", "importance": 0.7, "region": "global"}},
    {"key": "eco_currency", "value": "Currency exchange rates fluctuate based on economic conditions", "metadata": {"topic": "economics", "importance": 0.7, "region": "global"}},
    {"key": "art_davinci", "value": "Leonardo da Vinci painted the Mona Lisa and The Last Supper", "metadata": {"topic": "art", "importance": 0.9, "region": "europe"}},
    {"key": "art_vangogh", "value": "Van Gogh painted Starry Night and cut off part of his ear", "metadata": {"topic": "art", "importance": 0.8, "region": "europe"}},
    {"key": "art_picasso", "value": "Pablo Picasso co-founded Cubism and painted Guernica", "metadata": {"topic": "art", "importance": 0.8, "region": "europe"}},
    {"key": "art_monet", "value": "Claude Monet was a founder of French Impressionist painting", "metadata": {"topic": "art", "importance": 0.7, "region": "europe"}},
    {"key": "art_michelangelo", "value": "Michelangelo sculpted David and painted the Sistine Chapel ceiling", "metadata": {"topic": "art", "importance": 0.8, "region": "europe"}},
]

assert len(FACTS) == 100, f"Expected 100 facts, got {len(FACTS)}"

# ====================================================================
# 30 benchmark tasks
# ====================================================================
# Each task: query, expected_keys (ground truth relevant keys), filters (optional)
TASKS: list[dict] = [
    # --- Exact-match queries (1-10) ---
    {"query": "capital of France Paris", "expected_keys": ["capital_france"], "filters": None},
    {"query": "capital of Germany Berlin", "expected_keys": ["capital_germany"], "filters": None},
    {"query": "capital of Japan Tokyo", "expected_keys": ["capital_japan"], "filters": None},
    {"query": "Hydrogen lightest element atomic number 1", "expected_keys": ["element_hydrogen"], "filters": None},
    {"query": "Python programming language Guido van Rossum", "expected_keys": ["lang_python"], "filters": None},
    {"query": "Mount Everest tallest mountain 8849 meters", "expected_keys": ["mountain_everest"], "filters": None},
    {"query": "Einstein theory relativity E equals mc squared", "expected_keys": ["physics_relativity"], "filters": None},
    {"query": "Shakespeare Hamlet Romeo Juliet Macbeth", "expected_keys": ["lit_shakespeare"], "filters": None},
    {"query": "Leonardo da Vinci Mona Lisa Last Supper", "expected_keys": ["art_davinci"], "filters": None},
    {"query": "blue whale largest animal ever known", "expected_keys": ["animal_whale"], "filters": None},

    # --- Keyword-overlap / semantic queries (11-20) ---
    {"query": "fastest animal on land", "expected_keys": ["animal_cheetah"], "filters": None},
    {"query": "what planet has rings", "expected_keys": ["planet_saturn"], "filters": None},
    {"query": "oldest river in Africa", "expected_keys": ["river_nile"], "filters": None},
    {"query": "programming language for web browsers", "expected_keys": ["lang_javascript"], "filters": None},
    {"query": "invention of telephone communication", "expected_keys": ["inventor_bell"], "filters": None},
    {"query": "largest ocean deepest water", "expected_keys": ["ocean_pacific"], "filters": None},
    {"query": "first powered flight airplane", "expected_keys": ["inventor_wright"], "filters": None},
    {"query": "human heart beats day", "expected_keys": ["body_heart"], "filters": None},
    {"query": "country largest population world", "expected_keys": ["country_pop_india", "country_pop_china"], "filters": None},
    {"query": "famous dystopian novel totalitarianism", "expected_keys": ["lit_orwell"], "filters": None},

    # --- Metadata-filtered queries (21-30) ---
    {"query": "capital city", "expected_keys": ["capital_france", "capital_germany", "capital_japan", "capital_brazil", "capital_australia", "capital_canada", "capital_india", "capital_italy", "capital_mexico", "capital_egypt"], "filters": {"topic": "geography"}},
    {"query": "planet solar system", "expected_keys": ["planet_mercury", "planet_venus", "planet_mars", "planet_jupiter", "planet_saturn"], "filters": {"topic": "astronomy"}},
    {"query": "programming language", "expected_keys": ["lang_python", "lang_javascript", "lang_rust", "lang_java", "lang_go"], "filters": {"topic": "programming"}},
    {"query": "element chemistry", "expected_keys": ["element_hydrogen", "element_helium", "element_oxygen", "element_carbon", "element_gold"], "filters": {"topic": "chemistry"}},
    {"query": "composer music symphony", "expected_keys": ["music_beethoven", "music_mozart"], "filters": {"topic": "music"}},
    {"query": "sport popular invented", "expected_keys": ["sport_soccer", "sport_basketball", "sport_tennis", "sport_cricket", "sport_swimming"], "filters": {"topic": "sports"}},
    {"query": "painting art famous", "expected_keys": ["art_davinci", "art_vangogh", "art_picasso", "art_monet", "art_michelangelo"], "filters": {"topic": "art"}},
    {"query": "novel literature wrote", "expected_keys": ["lit_shakespeare", "lit_tolstoy", "lit_twain", "lit_orwell", "lit_austen"], "filters": {"topic": "literature"}},
    {"query": "invention history pioneer", "expected_keys": ["inventor_edison", "inventor_tesla", "inventor_wright", "inventor_bell", "inventor_curie", "history_rome", "history_moon", "history_internet", "history_printing", "history_electricity"], "filters": {"topic": "history"}},
    {"query": "technology innovation", "expected_keys": ["tech_ai", "tech_blockchain", "tech_cloud", "tech_5g", "tech_iot"], "filters": {"topic": "technology"}},
]

assert len(TASKS) == 30, f"Expected 30 tasks, got {len(TASKS)}"

# ====================================================================
# Benchmark runner
# ====================================================================


def run_benchmark() -> float:
    store = MemoryStore()

    # Load all facts
    for fact in FACTS:
        store.store(fact["key"], fact["value"], fact["metadata"])

    assert store.size() == 100, f"Store size mismatch: {store.size()}"

    k = 5
    total_recall = 0.0
    total_precision = 0.0
    total_latency = 0.0
    max_latency = 0.0

    for task in TASKS:
        query = task["query"]
        expected = set(task["expected_keys"])
        filters = task.get("filters")

        t0 = time.perf_counter()
        results = store.retrieve(query, k=k, filters=filters)
        elapsed = time.perf_counter() - t0

        total_latency += elapsed
        max_latency = max(max_latency, elapsed)

        retrieved_keys = {r["key"] for r in results}

        # Recall@5: fraction of expected items found in top-5
        if expected:
            recall = len(retrieved_keys & expected) / len(expected)
        else:
            recall = 1.0

        # Precision@5: fraction of returned items that are relevant
        if retrieved_keys:
            precision = len(retrieved_keys & expected) / len(retrieved_keys)
        else:
            precision = 0.0

        total_recall += recall
        total_precision += precision

    n = len(TASKS)
    avg_recall = total_recall / n
    avg_precision = total_precision / n

    # Normalise latency: 0 is best (< 1ms per query), 1 is worst (> 100ms per query)
    avg_latency_ms = (total_latency / n) * 1000
    latency_norm = min(avg_latency_ms / 100.0, 1.0)

    fitness = 0.5 * avg_recall + 0.3 * avg_precision + 0.2 * (1.0 - latency_norm)

    print(f"recall@{k}:          {avg_recall:.6f}")
    print(f"precision@{k}:       {avg_precision:.6f}")
    print(f"avg_latency_ms:      {avg_latency_ms:.4f}")
    print(f"latency_normalized:  {latency_norm:.6f}")
    print(f"score: {fitness:.6f}")

    return fitness


if __name__ == "__main__":
    run_benchmark()
