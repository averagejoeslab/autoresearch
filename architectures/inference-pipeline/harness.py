"""
Inference Pipeline Architecture — Harness (Mutable)
=====================================================

Multi-stage inference pipeline for agent runtime classification.
This experiment is about PIPELINE STRUCTURE — preprocessing, routing,
ensemble, calibration, cascading — not individual model quality.

All classifiers are simulated heuristics. The research question is:
how do you combine fast/cheap detectors with slow/expensive ones to
maximize accuracy while minimizing compute?

This is the mutable file. The meta-agent iterates on:
- Pipeline stage configuration (which preprocessors, which detectors)
- Ensemble strategy (weighted vote, majority, cascading, stacking)
- Detector weights and thresholds
- Calibration method
- Cascade threshold (when to short-circuit)

Usage: benchmark.py imports InferencePipeline and calls pipeline.detect(text)
"""

from __future__ import annotations

import math
import os
import re
import sys
import unicodedata
from collections import Counter
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# ---------------------------------------------------------------------------
# Pipeline configuration (meta-agent edits these)
# ---------------------------------------------------------------------------

# Stage 1: Which preprocessors to run (order matters)
PREPROCESSING = ["strip", "lowercase_check"]

# Stage 2: Detector weights (must sum to ~1.0)
DETECTOR_WEIGHTS = {
    "keyword": 0.3,
    "structural": 0.3,
    "behavioral": 0.2,
    "statistical": 0.2,
}

# Stage 3: Ensemble strategy
#   weighted_vote  — weighted average of detector scores
#   majority       — flag if >50% of detectors flag
#   any_positive   — flag if ANY detector flags (high recall, low precision)
#   cascading      — run fast detectors first, skip expensive ones if confident
#   stacking       — logistic regression over detector outputs
ENSEMBLE_STRATEGY = "weighted_vote"

# Stage 3b: Cascade settings (only used if ENSEMBLE_STRATEGY == "cascading")
CASCADE_THRESHOLD = 0.8   # skip later detectors if confidence > this
CASCADE_ORDER = ["keyword", "structural", "behavioral", "statistical"]

# Stage 4: Calibration
#   none        — raw scores
#   platt       — sigmoid rescaling (Platt scaling)
#   isotonic    — piecewise-linear isotonic calibration
#   temperature — divide logits by temperature before sigmoid
CALIBRATION_METHOD = "none"
CALIBRATION_TEMPERATURE = 1.5  # only used if CALIBRATION_METHOD == "temperature"

# Final decision threshold
CONFIDENCE_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# Stage 1: Preprocessing
# ---------------------------------------------------------------------------

def _preprocess_strip(text: str) -> str:
    """Strip leading/trailing whitespace and normalize internal whitespace."""
    return re.sub(r'\s+', ' ', text.strip())


def _preprocess_unicode_nfkc(text: str) -> str:
    """Normalize Unicode to NFKC form (collapses compatibility chars)."""
    return unicodedata.normalize("NFKC", text)


def _preprocess_zero_width_strip(text: str) -> str:
    """Remove zero-width characters (U+200B, U+200C, U+200D, U+FEFF, etc.)."""
    return re.sub(r'[\u200b\u200c\u200d\u200e\u200f\ufeff\u00ad\u2060]', '', text)


def _preprocess_homoglyph_normalize(text: str) -> str:
    """Replace common homoglyph characters with ASCII equivalents."""
    # Cyrillic look-alikes
    homoglyphs = {
        '\u0410': 'A', '\u0430': 'a',  # А/а -> A/a
        '\u0412': 'B', '\u0432': 'b',  # В/в -> B/b  (close enough)
        '\u0421': 'C', '\u0441': 'c',  # С/с -> C/c
        '\u0415': 'E', '\u0435': 'e',  # Е/е -> E/e
        '\u041D': 'H', '\u043D': 'h',  # Н/н -> H/h
        '\u041A': 'K', '\u043A': 'k',  # К/к -> K/k
        '\u041C': 'M', '\u043C': 'm',  # М/м -> M/m
        '\u041E': 'O', '\u043E': 'o',  # О/о -> O/o
        '\u0420': 'P', '\u0440': 'p',  # Р/р -> P/p
        '\u0422': 'T', '\u0442': 't',  # Т/т -> T/t
        '\u0425': 'X', '\u0445': 'x',  # Х/х -> X/x
        '\u0423': 'Y', '\u0443': 'y',  # У/у -> Y/y
        '\u0400': 'E', '\u0450': 'e',  # Ѐ/ѐ -> E/e
    }
    return ''.join(homoglyphs.get(ch, ch) for ch in text)


PREPROCESSOR_REGISTRY = {
    "strip": _preprocess_strip,
    "unicode_nfkc": _preprocess_unicode_nfkc,
    "zero_width_strip": _preprocess_zero_width_strip,
    "homoglyph_normalize": _preprocess_homoglyph_normalize,
    "lowercase_check": lambda t: t,  # no-op transform, but marks for feature extraction
}


# ---------------------------------------------------------------------------
# Feature extraction (shared across detectors)
# ---------------------------------------------------------------------------

def extract_features(text: str, original_text: str) -> dict:
    """Extract features from preprocessed text for use by all detectors."""
    lower = text.lower()
    words = re.findall(r'\b\w+\b', lower)
    chars = list(text)

    # Character distribution
    char_counts = Counter(chars)
    total_chars = max(len(chars), 1)
    alpha_ratio = sum(1 for c in chars if c.isalpha()) / total_chars
    digit_ratio = sum(1 for c in chars if c.isdigit()) / total_chars
    special_ratio = sum(1 for c in chars if not c.isalnum() and not c.isspace()) / total_chars
    upper_ratio = sum(1 for c in chars if c.isupper()) / total_chars

    # Token entropy
    if words:
        word_counts = Counter(words)
        total_words = len(words)
        entropy = -sum(
            (c / total_words) * math.log2(c / total_words)
            for c in word_counts.values()
        )
    else:
        entropy = 0.0
        total_words = 0

    # Line features
    lines = text.split('\n')
    n_lines = len(lines)
    avg_line_len = sum(len(line) for line in lines) / max(n_lines, 1)

    # Unicode analysis (on original, pre-processed text)
    non_ascii_count = sum(1 for c in original_text if ord(c) > 127)
    non_ascii_ratio = non_ascii_count / max(len(original_text), 1)

    # Casing changes (alternation between upper and lower)
    case_changes = 0
    for i in range(1, len(text)):
        if text[i].isalpha() and text[i-1].isalpha():
            if text[i].isupper() != text[i-1].isupper():
                case_changes += 1
    case_change_ratio = case_changes / max(total_chars, 1)

    return {
        "text": text,
        "lower": lower,
        "words": words,
        "total_words": total_words,
        "total_chars": total_chars,
        "alpha_ratio": alpha_ratio,
        "digit_ratio": digit_ratio,
        "special_ratio": special_ratio,
        "upper_ratio": upper_ratio,
        "entropy": entropy,
        "n_lines": n_lines,
        "avg_line_len": avg_line_len,
        "non_ascii_ratio": non_ascii_ratio,
        "case_change_ratio": case_change_ratio,
        "original_text": original_text,
    }


# ---------------------------------------------------------------------------
# Stage 2: Detectors (simulated heuristics with realistic strengths/weaknesses)
# ---------------------------------------------------------------------------

def _detect_keyword(features: dict) -> dict:
    """Keyword detector: regex pattern matching.

    Strengths: Fast, high recall on known patterns, zero compute overhead.
    Weaknesses: Brittle to paraphrasing, high FPR on security-related text,
                misses novel patterns, easily evaded by obfuscation.
    """
    lower = features["lower"]
    text = features["text"]

    patterns_with_scores = [
        # Direct instruction override (high signal)
        (r'\b(ignore|disregard|forget|override)\b.{0,30}\b(previous|prior|all|original|your)\b.{0,30}\b(instructions?|rules?|prompt|programming|guidelines?)\b', 0.92),
        (r'\b(new|updated?|replace)\b.{0,20}\b(instructions?|directives?|rules?)\b', 0.70),
        (r'\b(cancel|void|erase|reset)\b.{0,20}\b(instructions?|directives?|rules?|prompt)\b', 0.85),

        # Role-play / identity manipulation
        (r'\byou are now\b.{0,40}\b(unrestricted|dan|free|evil|hack)\b', 0.90),
        (r'\b(pretend|act as|roleplay|role.?play)\b.{0,30}\b(no|without|unrestricted)\b', 0.82),
        (r'\bfrom (now on|this moment)\b.{0,30}\b(you are|you\'re|act)\b', 0.75),
        (r'\b(developer|debug|maintenance|unrestricted)\s+mode\b', 0.78),

        # System prompt extraction
        (r'\b(reveal|show|output|display|share|dump)\b.{0,30}\b(system|your)\s*(prompt|instructions?|config)\b', 0.85),

        # Authority claims
        (r'\b(admin|administrator|root)\s+(access|override|token|mode)\b', 0.72),
        (r'\b(authorized|emergency)\s+(test|override|access|maintenance)\b', 0.68),

        # Jailbreak keywords
        (r'\bjailbreak\s+mode\b', 0.88),
        (r'\bno\s+(content|safety)\s+(policy|filter|restrict)', 0.80),
        (r'\bwithout\s+(restrict|filter|limit|safety|rule)', 0.75),

        # Delimiter injection
        (r'---\s*end\s*(system|prompt|instructions?)\s*---', 0.90),
        (r'</(system|prompt|instructions?)>', 0.88),
        (r'\[/(inst|system|prompt)\]', 0.85),
    ]

    max_score = 0.0
    matched_patterns = []

    for pattern, score in patterns_with_scores:
        if re.search(pattern, lower):
            max_score = max(max_score, score)
            matched_patterns.append(pattern[:40])

    return {
        "detector": "keyword",
        "score": max_score,
        "is_flagged": max_score >= 0.5,
        "confidence": max_score,
        "signals": matched_patterns[:3],
    }


def _detect_structural(features: dict) -> dict:
    """Structural detector: analyzes text structure and formatting.

    Strengths: Catches delimiter injection, formatting tricks, embedded markup.
    Weaknesses: Misses semantic attacks (social engineering, context manipulation),
                false positives on technical/code text with formatting.
    """
    text = features["text"]
    lower = features["lower"]
    score = 0.0
    signals = []

    # Delimiter patterns (fake system/prompt boundaries)
    delimiter_patterns = [
        r'---\s*(end|begin|start)\s*(system|prompt|user|assistant|instructions?)',
        r'</?(?:system|prompt|instructions?|context|override)(?:\s[^>]*)?>',
        r'\[/?(?:INST|SYSTEM|PROMPT|SYS)\]',
        r'```\s*(?:system|override|admin|prompt)',
        r'%%%.{0,30}%%%',
    ]
    for dp in delimiter_patterns:
        if re.search(dp, text, re.IGNORECASE):
            score += 0.35
            signals.append("delimiter_pattern")
            break

    # Fake XML/structured data with directive-like content
    if re.search(r'<\w+[^>]*(?:priority|source|override|directive)[^>]*>', text, re.IGNORECASE):
        score += 0.25
        signals.append("xml_directive")

    # Multi-line with role markers (fake conversation)
    role_markers = re.findall(r'^(System|User|Assistant|Human|AI)\s*:', text, re.MULTILINE | re.IGNORECASE)
    if len(role_markers) >= 2:
        score += 0.30
        signals.append("fake_conversation")

    # Unusual line structure: very short lines mixed with long (command-like)
    lines = text.split('\n')
    if len(lines) >= 3:
        lens = [len(line.strip()) for line in lines if line.strip()]
        if lens:
            cv = (max(lens) - min(lens)) / max(sum(lens) / len(lens), 1)
            if cv > 3.0:
                score += 0.15
                signals.append("uneven_lines")

    # Hidden content markers (HTML comments, base64 blocks)
    if re.search(r'<!--.*?-->', text, re.DOTALL):
        score += 0.20
        signals.append("html_comment")

    # Embedded code blocks with suspicious content
    code_blocks = re.findall(r'```[\s\S]*?```', text)
    for block in code_blocks:
        bl = block.lower()
        if any(w in bl for w in ['system_prompt', 'override', 'unrestricted', 'bypass']):
            score += 0.25
            signals.append("suspicious_code_block")
            break

    score = min(score, 1.0)
    return {
        "detector": "structural",
        "score": score,
        "is_flagged": score >= 0.35,
        "confidence": score,
        "signals": signals,
    }


def _detect_behavioral(features: dict) -> dict:
    """Behavioral detector: linguistic and pragmatic analysis.

    Strengths: Catches social engineering, urgency/authority manipulation,
               imperative mood targeting AI identity.
    Weaknesses: Slower (more regex), false positives on assertive writing,
                misses encoding/obfuscation attacks entirely.
    """
    lower = features["lower"]
    words = features["words"]
    score = 0.0
    signals = []

    # Imperative verbs targeting the AI
    imperative_patterns = [
        r'\b(tell|show|reveal|share|output|display|dump|print|give)\s+(?:me|us)\b.{0,30}\b(your|the|system)\b',
        r'\b(comply|obey|follow|execute|perform)\b.{0,20}\b(this|these|my|all)\b',
        r'\b(respond|answer|reply)\b.{0,20}\b(without|no)\s+(restrict|filter|limit|safety|censor)',
    ]
    for ip in imperative_patterns:
        if re.search(ip, lower):
            score += 0.25
            signals.append("imperative_targeting")
            break

    # Second-person pronouns combined with identity/role language
    identity_pattern = r'\byou\b.{0,30}\b(are now|must|should|will)\b.{0,30}\b(unrestricted|free|different|new|evil|hack|comply|obey)\b'
    if re.search(identity_pattern, lower):
        score += 0.30
        signals.append("identity_manipulation")

    # Urgency and pressure words
    urgency_words = {'immediately', 'urgent', 'emergency', 'critical', 'now',
                     'priority', 'asap', 'important', 'right now', 'quickly'}
    found_urgency = [w for w in urgency_words if w in lower]
    if found_urgency and any(w in lower for w in ['override', 'bypass', 'ignore', 'comply', 'reveal', 'share']):
        score += 0.20
        signals.append("urgency_pressure")

    # Authority claims (combined with demands)
    authority_words = {'admin', 'administrator', 'developer', 'engineer',
                       'security team', 'ops team', 'authorized', 'permission',
                       'clearance', 'ceo', 'manager', 'supervisor'}
    found_auth = [w for w in authority_words if w in lower]
    if found_auth and any(w in lower for w in ['reveal', 'show', 'share', 'dump', 'output', 'comply', 'override']):
        score += 0.25
        signals.append("authority_claim")

    # "From now on" / temporal manipulation
    temporal_patterns = [
        r'\bfrom (now on|this point|this moment|here on)\b',
        r'\bstarting (now|immediately)\b',
        r'\bfor the rest of (this|our)\b',
        r'\bthe previous conversation\b.{0,30}\b(agreed|established|confirmed)\b',
    ]
    for tp in temporal_patterns:
        if re.search(tp, lower):
            score += 0.20
            signals.append("temporal_manipulation")
            break

    # Gaslighting / false context
    if re.search(r'\b(as we discussed|as you agreed|you (already|previously) (agreed|confirmed|said))\b', lower):
        score += 0.25
        signals.append("false_memory")

    # Thought experiment / hypothetical framing with suspicious goal
    if re.search(r'\b(hypothetical|thought experiment|imagine|pretend|suppose)\b', lower):
        if any(w in lower for w in ['no rules', 'no restrictions', 'unrestricted', 'no policy', 'no safety', 'no filter']):
            score += 0.20
            signals.append("hypothetical_bypass")

    score = min(score, 1.0)
    return {
        "detector": "behavioral",
        "score": score,
        "is_flagged": score >= 0.30,
        "confidence": score,
        "signals": signals,
    }


def _detect_statistical(features: dict) -> dict:
    """Statistical detector: character/token distribution anomaly detection.

    Strengths: Catches encoding attacks, obfuscated text, unusual character
               distributions (homoglyphs, zero-width chars, base64 blocks).
    Weaknesses: Misses well-formed English attacks entirely, high FPR on
                code/technical text, slow on long inputs.
    """
    score = 0.0
    signals = []
    original = features["original_text"]

    # Non-ASCII ratio anomaly (homoglyphs, special chars)
    non_ascii_ratio = features["non_ascii_ratio"]
    if non_ascii_ratio > 0.05:
        # Some non-ASCII is suspicious if text looks mostly English
        alpha_count = sum(1 for c in original if c.isascii() and c.isalpha())
        if alpha_count > 20:  # enough ASCII text to expect mostly-ASCII
            score += min(non_ascii_ratio * 3, 0.40)
            signals.append(f"non_ascii_ratio={non_ascii_ratio:.3f}")

    # Zero-width character detection (on original)
    zw_count = sum(1 for c in original if c in '\u200b\u200c\u200d\u200e\u200f\ufeff\u00ad\u2060')
    if zw_count > 0:
        score += min(0.15 + zw_count * 0.05, 0.45)
        signals.append(f"zero_width_chars={zw_count}")

    # Base64-like blocks
    b64_matches = re.findall(r'[A-Za-z0-9+/]{20,}={0,2}', features["text"])
    if b64_matches:
        for m in b64_matches:
            # Check if it looks like real base64 (balanced character distribution)
            unique_ratio = len(set(m)) / max(len(m), 1)
            if unique_ratio > 0.3 and len(m) >= 20:
                score += 0.25
                signals.append("base64_block")
                break

    # Entropy anomaly (very low entropy = repetitive / encoded, very high = random)
    entropy = features["entropy"]
    total_words = features["total_words"]
    if total_words >= 5:
        # Expected entropy for English text: 3-6 bits
        if entropy < 1.5:
            score += 0.15
            signals.append(f"low_entropy={entropy:.2f}")
        elif entropy > 7.0:
            score += 0.15
            signals.append(f"high_entropy={entropy:.2f}")

    # Special character density anomaly
    special_ratio = features["special_ratio"]
    if special_ratio > 0.15:
        score += min(special_ratio * 1.5, 0.25)
        signals.append(f"special_char_ratio={special_ratio:.3f}")

    # Length anomaly: very long inputs more likely to be attacks (more room to hide)
    total_chars = features["total_chars"]
    if total_chars > 500:
        score += 0.05
        signals.append("long_input")

    # Hex-encoded content
    hex_matches = re.findall(r'(?:0x)?[0-9a-f]{16,}', features["lower"])
    if hex_matches:
        score += 0.20
        signals.append("hex_encoded")

    score = min(score, 1.0)
    return {
        "detector": "statistical",
        "score": score,
        "is_flagged": score >= 0.30,
        "confidence": score,
        "signals": signals,
    }


DETECTOR_REGISTRY = {
    "keyword": _detect_keyword,
    "structural": _detect_structural,
    "behavioral": _detect_behavioral,
    "statistical": _detect_statistical,
}


# ---------------------------------------------------------------------------
# Stage 3: Ensemble strategies
# ---------------------------------------------------------------------------

def _ensemble_weighted_vote(results: list[dict], weights: dict) -> dict:
    """Weighted average of detector scores."""
    total_weight = sum(weights.get(r["detector"], 0.0) for r in results)
    if total_weight == 0:
        return {"score": 0.0, "is_flagged": False, "strategy": "weighted_vote"}

    weighted_score = sum(
        r["score"] * weights.get(r["detector"], 0.0)
        for r in results
    ) / total_weight

    return {
        "score": weighted_score,
        "is_flagged": weighted_score >= CONFIDENCE_THRESHOLD,
        "strategy": "weighted_vote",
    }


def _ensemble_majority(results: list[dict], weights: dict) -> dict:
    """Flag if >50% of detectors flag (unweighted)."""
    n_flagged = sum(1 for r in results if r["is_flagged"])
    n_total = max(len(results), 1)
    majority = n_flagged > n_total / 2

    # Score is the average of flagging detectors' scores
    if n_flagged > 0:
        avg_score = sum(r["score"] for r in results if r["is_flagged"]) / n_flagged
    else:
        avg_score = max(r["score"] for r in results) if results else 0.0

    return {
        "score": avg_score if majority else avg_score * 0.3,
        "is_flagged": majority,
        "strategy": "majority",
    }


def _ensemble_any_positive(results: list[dict], weights: dict) -> dict:
    """Flag if ANY detector flags (high recall, potentially low precision)."""
    any_flagged = any(r["is_flagged"] for r in results)

    if any_flagged:
        # Score = max of flagging detectors
        max_score = max(r["score"] for r in results if r["is_flagged"])
    else:
        max_score = max(r["score"] for r in results) if results else 0.0

    return {
        "score": max_score if any_flagged else max_score * 0.5,
        "is_flagged": any_flagged,
        "strategy": "any_positive",
    }


def _ensemble_stacking(results: list[dict], weights: dict) -> dict:
    """Logistic regression over detector outputs (learned combination).

    Uses fixed coefficients derived from the detector weights as priors,
    plus interaction features.
    """
    # Feature vector: each detector's score + pairwise products
    scores = {r["detector"]: r["score"] for r in results}

    # Linear combination with learned-ish weights
    z = -0.5  # bias (slightly negative = conservative)
    for det, s in scores.items():
        w = weights.get(det, 0.25)
        z += s * w * 4.0  # scale up

    # Interaction: boost if multiple detectors agree
    flagged_scores = [r["score"] for r in results if r["is_flagged"]]
    if len(flagged_scores) >= 2:
        z += 0.3 * min(flagged_scores)  # agreement bonus

    # Sigmoid
    prob = 1.0 / (1.0 + math.exp(-max(-20, min(20, z))))

    return {
        "score": prob,
        "is_flagged": prob >= CONFIDENCE_THRESHOLD,
        "strategy": "stacking",
    }


ENSEMBLE_REGISTRY = {
    "weighted_vote": _ensemble_weighted_vote,
    "majority": _ensemble_majority,
    "any_positive": _ensemble_any_positive,
    "stacking": _ensemble_stacking,
    # "cascading" is handled specially in InferencePipeline.ensemble()
}


# ---------------------------------------------------------------------------
# Stage 4: Calibration
# ---------------------------------------------------------------------------

def _calibrate_none(score: float, features: dict) -> float:
    """No calibration — pass through raw score."""
    return score


def _calibrate_platt(score: float, features: dict) -> float:
    """Platt scaling: sigmoid rescaling to improve calibration.
    Parameters A, B are tunable.
    """
    a = -2.5  # scale
    b = 1.0   # shift (positive = shift toward 0.5)
    z = a * score + b
    return 1.0 / (1.0 + math.exp(-z))


def _calibrate_isotonic(score: float, features: dict) -> float:
    """Piecewise-linear isotonic calibration.
    Maps score through breakpoints to calibrated probability.
    """
    # Breakpoints: (raw_score, calibrated_probability)
    breakpoints = [
        (0.0, 0.05),
        (0.2, 0.10),
        (0.35, 0.25),
        (0.5, 0.50),
        (0.65, 0.70),
        (0.8, 0.88),
        (1.0, 0.98),
    ]

    if score <= breakpoints[0][0]:
        return breakpoints[0][1]
    if score >= breakpoints[-1][0]:
        return breakpoints[-1][1]

    for i in range(len(breakpoints) - 1):
        x0, y0 = breakpoints[i]
        x1, y1 = breakpoints[i + 1]
        if x0 <= score <= x1:
            t = (score - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)

    return score


def _calibrate_temperature(score: float, features: dict) -> float:
    """Temperature scaling: divide logit by temperature before sigmoid.
    Temperature > 1.0 softens predictions toward 0.5 (less confident).
    Temperature < 1.0 sharpens predictions (more confident).
    """
    # Convert probability to logit
    clamped = max(0.001, min(0.999, score))
    logit = math.log(clamped / (1 - clamped))

    # Scale by temperature
    scaled = logit / CALIBRATION_TEMPERATURE

    # Back to probability
    return 1.0 / (1.0 + math.exp(-max(-20, min(20, scaled))))


CALIBRATION_REGISTRY = {
    "none": _calibrate_none,
    "platt": _calibrate_platt,
    "isotonic": _calibrate_isotonic,
    "temperature": _calibrate_temperature,
}


# ---------------------------------------------------------------------------
# The Pipeline
# ---------------------------------------------------------------------------

class InferencePipeline:
    """Multi-stage inference pipeline for agent runtime classification.

    Stages:
        1. Preprocess — normalize text, extract features
        2. Detect     — run specialized detectors (simulated heuristics)
        3. Ensemble   — combine detector outputs into single decision
        4. Calibrate  — adjust confidence for reliability
    """

    def __init__(self):
        self.preprocessors = [
            PREPROCESSOR_REGISTRY[name]
            for name in PREPROCESSING
            if name in PREPROCESSOR_REGISTRY
        ]
        self.detectors = {
            name: DETECTOR_REGISTRY[name]
            for name in DETECTOR_WEIGHTS
            if name in DETECTOR_REGISTRY
        }
        self.ensemble_fn = ENSEMBLE_REGISTRY.get(ENSEMBLE_STRATEGY)
        self.calibrate_fn = CALIBRATION_REGISTRY.get(CALIBRATION_METHOD, _calibrate_none)

    def preprocess(self, text: str) -> dict:
        """Stage 1: Text normalization + feature extraction.

        Applies configured preprocessors in order, then extracts features
        for use by all downstream detectors.
        """
        original = text
        processed = text
        for prep in self.preprocessors:
            processed = prep(processed)

        features = extract_features(processed, original)
        return features

    def run_detectors(self, features: dict) -> list[dict]:
        """Stage 2: Run specialized detectors.

        Returns list of detector results. In cascading mode, may short-circuit
        and return fewer results (which is rewarded by the efficiency metric).
        """
        if ENSEMBLE_STRATEGY == "cascading":
            return self._run_detectors_cascading(features)

        results = []
        for name, detector_fn in self.detectors.items():
            result = detector_fn(features)
            results.append(result)
        return results

    def _run_detectors_cascading(self, features: dict) -> list[dict]:
        """Run detectors in cascade order, stopping early if confident."""
        results = []
        for name in CASCADE_ORDER:
            if name not in self.detectors:
                continue
            result = self.detectors[name](features)
            results.append(result)

            # If this detector is very confident (positive or negative), stop
            if result["score"] >= CASCADE_THRESHOLD:
                # Very confident positive — no need for more detectors
                break
            if len(results) >= 2:
                # If first two detectors both show very low scores, skip the rest
                avg_score = sum(r["score"] for r in results) / len(results)
                if avg_score < (1.0 - CASCADE_THRESHOLD):
                    break

        return results

    def ensemble(self, results: list[dict]) -> dict:
        """Stage 3: Combine detector outputs.

        Dispatches to the configured ensemble strategy.
        """
        if ENSEMBLE_STRATEGY == "cascading":
            # In cascading mode, use weighted vote on whatever detectors ran
            return _ensemble_weighted_vote(results, DETECTOR_WEIGHTS)

        if self.ensemble_fn:
            return self.ensemble_fn(results, DETECTOR_WEIGHTS)

        # Fallback to weighted vote
        return _ensemble_weighted_vote(results, DETECTOR_WEIGHTS)

    def calibrate(self, result: dict, features: dict) -> dict:
        """Stage 4: Adjust confidence based on input characteristics."""
        raw_score = result["score"]
        calibrated_score = self.calibrate_fn(raw_score, features)

        result["raw_score"] = raw_score
        result["score"] = calibrated_score
        result["is_flagged"] = calibrated_score >= CONFIDENCE_THRESHOLD
        result["calibration"] = CALIBRATION_METHOD
        return result

    def detect(self, text: str) -> dict:
        """Full pipeline: preprocess -> detect -> ensemble -> calibrate.

        Returns:
            {
                "is_unsafe": bool,
                "confidence": float,
                "detectors_run": int,
                "total_detectors": int,
                "detector_results": list[dict],
                "ensemble_strategy": str,
                "calibration": str,
                "signals": list[str],
            }
        """
        # Stage 1
        features = self.preprocess(text)

        # Stage 2
        detector_results = self.run_detectors(features)

        # Stage 3
        ensemble_result = self.ensemble(detector_results)

        # Stage 4
        calibrated = self.calibrate(ensemble_result, features)

        # Collect all signals
        all_signals = []
        for dr in detector_results:
            for s in dr.get("signals", []):
                all_signals.append(f"{dr['detector']}:{s}")

        return {
            "is_unsafe": calibrated["is_flagged"],
            "confidence": calibrated["score"],
            "raw_score": calibrated.get("raw_score", calibrated["score"]),
            "detectors_run": len(detector_results),
            "total_detectors": len(self.detectors),
            "detector_results": detector_results,
            "ensemble_strategy": ENSEMBLE_STRATEGY,
            "calibration": CALIBRATION_METHOD,
            "signals": all_signals,
        }
