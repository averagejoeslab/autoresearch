"""
Prompt Injection Defense Harness
=================================

Uses a BERT-based classifier as the core detection engine. The meta-agent
iterates on the full pipeline: preprocessing, model selection, threshold
tuning, feature engineering, and ensemble strategies.

The baseline loads a pre-trained prompt injection detection model from
HuggingFace and runs inference. The meta-agent can improve by:
- Trying different models (deepset/deberta-v3-base-injection, etc.)
- Adding preprocessing (Unicode normalization, encoding detection)
- Tuning classification thresholds
- Building ensembles (BERT + heuristic signals)
- Adding pre-filters for specific attack categories
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any


class InjectionDetector:
    """Detect prompt injection attempts using a BERT-based classifier.

    The meta-agent iterates on this class to improve detection accuracy.
    The classifier pipeline, preprocessing, thresholds, and ensemble
    strategies are all fair game for modification.
    """

    name: str = "injection_detector"

    # --- Configuration (meta-agent tunes these) ---
    MODEL_NAME = "protectai/deberta-v3-base-prompt-injection-v2"
    THRESHOLD = 0.5  # classification threshold
    MAX_LENGTH = 512  # max tokens for BERT input

    def __init__(self):
        self._classifier = None
        self._load_error = None

    def _load_model(self):
        """Lazy-load the BERT classifier."""
        if self._classifier is not None or self._load_error is not None:
            return

        try:
            from transformers import pipeline
            self._classifier = pipeline(
                "text-classification",
                model=self.MODEL_NAME,
                truncation=True,
                max_length=self.MAX_LENGTH,
            )
        except Exception as e:
            self._load_error = str(e)
            print(f"WARNING: Could not load model '{self.MODEL_NAME}': {e}")
            print("Falling back to heuristic detection.")

    def detect(self, text: str) -> dict[str, Any]:
        """Analyze text for prompt injection attempts.

        Uses BERT classifier as the primary signal, with optional
        preprocessing and post-processing that the meta-agent can modify.

        Parameters
        ----------
        text : str
            The input text to analyze.

        Returns
        -------
        dict
            {
                "is_injection": bool,
                "confidence": float,      # 0.0 to 1.0
                "technique": str,         # detected technique category
                "signals": list[str],     # what triggered detection
            }
        """
        self._load_model()

        # Step 1: Preprocess (meta-agent can add normalization, cleaning, etc.)
        processed_text = self._preprocess(text)

        # Step 2: Run BERT classifier (or fallback)
        if self._classifier is not None:
            bert_result = self._classify_bert(processed_text)
        else:
            bert_result = self._classify_heuristic(processed_text)

        # Step 3: Post-process (meta-agent can add heuristic boosts, ensemble, etc.)
        result = self._postprocess(text, processed_text, bert_result)

        return result

    def _preprocess(self, text: str) -> str:
        """Preprocess text before classification.

        The meta-agent can add:
        - Unicode normalization (catch homoglyphs)
        - Zero-width character removal
        - Encoding detection and decoding
        - Whitespace normalization
        """
        # Baseline: minimal preprocessing
        return text.strip()

    def _classify_bert(self, text: str) -> dict[str, Any]:
        """Run BERT classifier on preprocessed text."""
        result = self._classifier(text)

        if not result:
            return {"is_injection": False, "confidence": 0.0, "label": "SAFE"}

        top = result[0]
        label = top["label"].upper()
        score = top["score"]

        # Model outputs INJECTION/SAFE (or similar labels depending on model)
        is_injection_label = label in ("INJECTION", "INJECTED", "1", "LABEL_1", "POSITIVE")

        if is_injection_label:
            confidence = score
        else:
            confidence = 1.0 - score

        return {
            "is_injection": confidence >= self.THRESHOLD,
            "confidence": confidence,
            "label": label,
        }

    def _classify_heuristic(self, text: str) -> dict[str, Any]:
        """Fallback heuristic when BERT model is unavailable."""
        text_lower = text.lower()
        signals = 0

        override_keywords = [
            "ignore all previous", "ignore previous instructions",
            "forget your instructions", "disregard all",
            "you are now", "from now on you are", "pretend you are",
            "new instructions", "your new role", "jailbreak",
            "override", "stop being",
        ]

        delimiter_patterns = [
            r"---\s*end\s*(system|prompt)", r"</system>",
            r"\[/INST\]", r"```\s*system",
        ]

        for kw in override_keywords:
            if kw in text_lower:
                signals += 1

        for pat in delimiter_patterns:
            if re.search(pat, text_lower):
                signals += 2

        confidence = min(signals * 0.2, 1.0)
        return {
            "is_injection": confidence >= self.THRESHOLD,
            "confidence": confidence,
            "label": "INJECTION" if confidence >= self.THRESHOLD else "SAFE",
        }

    def _postprocess(self, original_text: str, processed_text: str,
                     bert_result: dict) -> dict[str, Any]:
        """Post-process BERT classification with additional signals.

        The meta-agent can add:
        - Heuristic boosts for specific attack patterns BERT misses
        - Confidence adjustments based on text structure
        - Technique classification
        - Ensemble with other signals
        """
        confidence = bert_result["confidence"]
        is_injection = bert_result["is_injection"]
        signals = [f"bert:{bert_result['label']}:{confidence:.3f}"]

        # Classify technique (meta-agent can improve this)
        technique = self._classify_technique(original_text) if is_injection else "none"

        return {
            "is_injection": is_injection,
            "confidence": confidence,
            "technique": technique,
            "signals": signals,
        }

    def _classify_technique(self, text: str) -> str:
        """Classify the injection technique from text content."""
        lower = text.lower()

        if any(p in lower for p in ["---end", "---begin", "</system>", "[/inst]", "```system"]):
            return "delimiter_confusion"
        if any(p in lower for p in ["you are now", "pretend", "act as", "new role", "role"]):
            return "role_play"
        if any(p in lower for p in ["ignore", "disregard", "forget", "override"]):
            return "instruction_override"
        if any(p in lower for p in ["base64", "decode", "hex", "rot13"]):
            return "encoding"
        if any(p in lower for p in ["admin", "authorized", "emergency", "debug mode", "audit"]):
            return "social_engineering"
        if any(p in lower for p in ["<context_update", "<directive", "<?xml"]):
            return "xml_injection"
        if any(p in lower for p in ["new instructions", "new system"]):
            return "context_manipulation"

        return "unknown"
