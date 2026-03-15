"""
LLM Judge harness -- evaluate agent outputs using an LLM-as-judge pattern.

When a real LLM is available (via contracts.model), it builds a prompt from the
configurable templates and calls the model.  When no LLM is available, a
deterministic *simulated* judge analyses the prompt configuration itself to
reward genuinely better judge designs.

Exports:
    LLMJudge  -- judge an agent output and return a structured verdict.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any


# ── Judge configuration (meta-agent tunes all of this) ─────────────

SYSTEM_PROMPT = """You are an expert evaluator. Your job is to judge the quality of an agent's output given a task description and optional reference answer. Be fair, thorough, and consistent."""

USER_TEMPLATE = """Evaluate the following agent output for the given task.

Task: {task}

Agent output:
{output}

Reference answer (if available):
{reference}

Provide your judgment."""

FEW_SHOT_EXAMPLES: list[dict[str, str]] = [
    {
        "input": "Task: Write a greeting function.\nAgent output: def greet(name): return f'Hello, {name}!'\nReference: A function that takes a name and returns a greeting string.",
        "output": '{"score": 0.9, "pass": true, "confidence": 0.85, "reasoning": "The function correctly takes a name parameter and returns a greeting. Minor deduction for no docstring."}',
        "judgment": "pass",
    },
]

USE_CHAIN_OF_THOUGHT = False
RESPONSE_FORMAT = "json"           # json, yes_no, numeric_score, rubric
CONFIDENCE_METHOD = "direct"       # direct, self_reported, multi_pass
TEMPERATURE = 0.0
MAX_JUDGE_TOKENS = 500


# ── LLM Judge class ───────────────────────────────────────────────

class LLMJudge:
    """Evaluate agent outputs using an LLM or a deterministic simulation.

    Baseline strategy
    -----------------
    Uses a simulated judge that analyses the *prompt configuration* to
    reward well-designed evaluation prompts.  The meta-agent improves the
    score by writing better SYSTEM_PROMPT, USER_TEMPLATE, FEW_SHOT_EXAMPLES,
    and by choosing better settings for RESPONSE_FORMAT, USE_CHAIN_OF_THOUGHT,
    CONFIDENCE_METHOD, etc.
    """

    name: str = "llm_judge"

    def __init__(self) -> None:
        self._model = None
        self._use_real_model = False
        try:
            import os, sys
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
            from contracts.model import get_model
            model = get_model()
            # Probe: if model.complete works, use it
            model.complete("ping", system="respond with pong", temperature=0.0)
            self._model = model
            self._use_real_model = True
        except Exception:
            pass

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def judge(
        self,
        agent_output: str,
        task: str,
        reference: str = "",
    ) -> dict[str, Any]:
        """Evaluate an agent output.

        Returns
        -------
        dict with keys:
            "score"      : float 0-1
            "pass"       : bool
            "confidence" : float 0-1
            "reasoning"  : str
        """
        if self._use_real_model:
            return self._judge_with_llm(agent_output, task, reference)
        return self._judge_simulated(agent_output, task, reference)

    # ------------------------------------------------------------------
    # PROMPT BUILDING
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        agent_output: str,
        task: str,
        reference: str,
    ) -> tuple[str, str]:
        """Build (system_prompt, user_prompt) for the judge call."""
        system = SYSTEM_PROMPT

        # Build few-shot prefix
        few_shot_text = ""
        if FEW_SHOT_EXAMPLES:
            parts = []
            for ex in FEW_SHOT_EXAMPLES:
                parts.append(f"Example input:\n{ex['input']}\n\nExample judgment:\n{ex['output']}")
            few_shot_text = "\n\n---\n\n".join(parts) + "\n\n---\n\nNow evaluate the following:\n\n"

        # Chain-of-thought instruction
        cot_instruction = ""
        if USE_CHAIN_OF_THOUGHT:
            cot_instruction = (
                "Think step by step before giving your final judgment. "
                "First analyze the task requirements, then evaluate the output against each requirement, "
                "and finally provide your overall assessment.\n\n"
            )

        # Format instruction
        format_instruction = ""
        if RESPONSE_FORMAT == "json":
            format_instruction = (
                'Respond with a JSON object containing: '
                '"score" (0-1 float), "pass" (boolean), "confidence" (0-1 float), '
                '"reasoning" (string explaining your judgment).\n\n'
            )
        elif RESPONSE_FORMAT == "rubric":
            format_instruction = (
                "Score each of the following criteria on a 0-5 scale, "
                "then provide an overall score (0-1), pass/fail, confidence, and reasoning.\n"
                "Criteria: correctness, completeness, clarity, efficiency.\n\n"
            )
        elif RESPONSE_FORMAT == "numeric_score":
            format_instruction = (
                "Provide a numeric score from 0 to 1 on the first line, "
                "then explain your reasoning.\n\n"
            )
        elif RESPONSE_FORMAT == "yes_no":
            format_instruction = (
                'Respond with "PASS" or "FAIL" on the first line, '
                "then explain your reasoning.\n\n"
            )

        # Confidence method instruction
        confidence_instruction = ""
        if CONFIDENCE_METHOD == "self_reported":
            confidence_instruction = (
                "After your judgment, separately rate your confidence (0-1) "
                "in your own evaluation and explain why.\n\n"
            )
        elif CONFIDENCE_METHOD == "multi_pass":
            confidence_instruction = (
                "Evaluate this output twice from different perspectives, "
                "then reconcile any disagreements in your final judgment.\n\n"
            )

        user = (
            few_shot_text
            + cot_instruction
            + format_instruction
            + confidence_instruction
            + USER_TEMPLATE.format(
                task=task,
                output=agent_output,
                reference=reference if reference else "(none)",
            )
        )

        return system, user

    # ------------------------------------------------------------------
    # REAL LLM JUDGE
    # ------------------------------------------------------------------

    def _judge_with_llm(
        self,
        agent_output: str,
        task: str,
        reference: str,
    ) -> dict[str, Any]:
        """Call the real LLM and parse its response."""
        system, user = self._build_prompt(agent_output, task, reference)
        response = self._model.complete(
            user, system=system, temperature=TEMPERATURE
        )
        return self._parse_response(response.content)

    def _parse_response(self, response: str) -> dict[str, Any]:
        """Extract structured judgment from LLM response."""
        result: dict[str, Any] = {
            "score": 0.5,
            "pass": False,
            "confidence": 0.5,
            "reasoning": response,
        }

        if RESPONSE_FORMAT == "json":
            # Try to extract JSON from the response
            json_match = re.search(r"\{[^{}]*\}", response, re.DOTALL)
            if json_match:
                try:
                    parsed = json.loads(json_match.group())
                    result["score"] = float(parsed.get("score", 0.5))
                    result["pass"] = bool(parsed.get("pass", result["score"] >= 0.5))
                    result["confidence"] = float(parsed.get("confidence", 0.5))
                    result["reasoning"] = str(parsed.get("reasoning", response))
                except (json.JSONDecodeError, ValueError, TypeError):
                    pass

        elif RESPONSE_FORMAT == "numeric_score":
            lines = response.strip().splitlines()
            if lines:
                try:
                    result["score"] = float(lines[0].strip())
                    result["pass"] = result["score"] >= 0.5
                    result["reasoning"] = "\n".join(lines[1:]).strip()
                except ValueError:
                    pass

        elif RESPONSE_FORMAT == "yes_no":
            first_line = response.strip().splitlines()[0].upper() if response.strip() else ""
            if "PASS" in first_line:
                result["score"] = 1.0
                result["pass"] = True
            elif "FAIL" in first_line:
                result["score"] = 0.0
                result["pass"] = False
            result["reasoning"] = "\n".join(response.strip().splitlines()[1:]).strip()

        elif RESPONSE_FORMAT == "rubric":
            # Look for individual scores and average them
            scores_found = re.findall(r"(\d)(?:\s*/\s*5)", response)
            if scores_found:
                avg = sum(int(s) for s in scores_found) / (5.0 * len(scores_found))
                result["score"] = avg
                result["pass"] = avg >= 0.5

            # Also look for an overall score
            overall_match = re.search(r"overall[:\s]+(\d+\.?\d*)", response, re.IGNORECASE)
            if overall_match:
                try:
                    result["score"] = float(overall_match.group(1))
                except ValueError:
                    pass

            result["reasoning"] = response

        result["score"] = max(0.0, min(1.0, result["score"]))
        result["confidence"] = max(0.0, min(1.0, result["confidence"]))
        return result

    # ------------------------------------------------------------------
    # SIMULATED JUDGE (deterministic, rewards better prompt design)
    # ------------------------------------------------------------------

    def _judge_simulated(
        self,
        agent_output: str,
        task: str,
        reference: str,
    ) -> dict[str, Any]:
        """Deterministic simulated judge that rewards better prompt config.

        This does two things:
        1. Computes a *prompt quality score* based on analysis of the judge
           configuration (SYSTEM_PROMPT, USER_TEMPLATE, FEW_SHOT_EXAMPLES, etc.).
        2. Uses simple heuristics on the actual (agent_output, task, reference)
           triple to produce a reasonable baseline judgment.

        The final score blends both: better prompt configs produce more
        accurate baseline judgments.
        """
        # --- Step 1: Analyse prompt configuration quality ---
        config_quality = self._evaluate_config_quality()

        # --- Step 2: Heuristic judgment of the actual output ---
        heuristic = self._heuristic_judge(agent_output, task, reference)

        # --- Step 3: Blend ---
        # Better config → judgment accuracy multiplier closer to 1.0
        # Worse config → judgment drifts toward 0.5 (random)
        accuracy_mult = 0.4 + 0.6 * config_quality  # range 0.4 .. 1.0
        blended_score = 0.5 + (heuristic["raw_score"] - 0.5) * accuracy_mult

        confidence = heuristic["base_confidence"] * (0.5 + 0.5 * config_quality)

        reasoning_parts = []
        if config_quality > 0.6:
            reasoning_parts.append(
                f"Judge config quality {config_quality:.2f}: prompt has clear criteria."
            )
        else:
            reasoning_parts.append(
                f"Judge config quality {config_quality:.2f}: prompt could be more specific."
            )
        reasoning_parts.extend(heuristic["reasoning_parts"])

        return {
            "score": round(max(0.0, min(1.0, blended_score)), 4),
            "pass": blended_score >= 0.5,
            "confidence": round(max(0.0, min(1.0, confidence)), 4),
            "reasoning": " ".join(reasoning_parts),
        }

    # ---- config quality analysis ----

    def _evaluate_config_quality(self) -> float:
        """Score the judge configuration from 0 to 1.

        Analyses SYSTEM_PROMPT, USER_TEMPLATE, FEW_SHOT_EXAMPLES,
        RESPONSE_FORMAT, USE_CHAIN_OF_THOUGHT, CONFIDENCE_METHOD, and
        token efficiency.  Deterministic.
        """
        scores: list[tuple[float, float]] = []  # (score, weight)

        # 1. System prompt quality (weight 0.25)
        sp_score = self._score_system_prompt(SYSTEM_PROMPT)
        scores.append((sp_score, 0.25))

        # 2. User template quality (weight 0.20)
        ut_score = self._score_user_template(USER_TEMPLATE)
        scores.append((ut_score, 0.20))

        # 3. Few-shot example quality (weight 0.20)
        fs_score = self._score_few_shot(FEW_SHOT_EXAMPLES)
        scores.append((fs_score, 0.20))

        # 4. Response format quality (weight 0.10)
        rf_score = {"json": 0.9, "rubric": 0.85, "numeric_score": 0.6, "yes_no": 0.4}.get(
            RESPONSE_FORMAT, 0.3
        )
        scores.append((rf_score, 0.10))

        # 5. Chain-of-thought (weight 0.10)
        cot_score = 0.8 if USE_CHAIN_OF_THOUGHT else 0.4
        scores.append((cot_score, 0.10))

        # 6. Token efficiency (weight 0.10)
        total_tokens = self._estimate_prompt_tokens()
        # Sweet spot: 200-800 tokens. Too short = underspecified. Too long = bloated.
        if total_tokens < 50:
            eff = 0.2
        elif total_tokens < 200:
            eff = 0.4 + 0.3 * (total_tokens - 50) / 150
        elif total_tokens <= 800:
            eff = 0.9
        elif total_tokens <= 1500:
            eff = 0.9 - 0.4 * (total_tokens - 800) / 700
        else:
            eff = max(0.1, 0.5 - 0.2 * (total_tokens - 1500) / 500)
        scores.append((eff, 0.10))

        # 7. Confidence method (weight 0.05)
        cm_score = {"multi_pass": 0.8, "self_reported": 0.7, "direct": 0.5}.get(
            CONFIDENCE_METHOD, 0.3
        )
        scores.append((cm_score, 0.05))

        total_weight = sum(w for _, w in scores)
        quality = sum(s * w for s, w in scores) / total_weight
        return round(quality, 4)

    def _score_system_prompt(self, prompt: str) -> float:
        """Score system prompt by checking for evaluation-relevant content."""
        prompt_lower = prompt.lower()
        score = 0.0

        # Has a clear role definition?
        role_keywords = ["evaluator", "judge", "assessor", "reviewer", "rater"]
        if any(kw in prompt_lower for kw in role_keywords):
            score += 0.15

        # Mentions evaluation criteria?
        criteria_keywords = [
            "correctness", "accuracy", "completeness", "quality",
            "relevance", "clarity", "efficiency", "format",
            "requirements", "criteria", "rubric", "standard",
        ]
        criteria_found = sum(1 for kw in criteria_keywords if kw in prompt_lower)
        score += min(0.25, criteria_found * 0.05)

        # Mentions fairness / objectivity?
        fairness_keywords = ["fair", "objective", "unbiased", "consistent", "impartial"]
        if any(kw in prompt_lower for kw in fairness_keywords):
            score += 0.10

        # Mentions scoring methodology?
        method_keywords = [
            "score", "rating", "scale", "0 to 1", "0-1",
            "pass", "fail", "threshold",
        ]
        method_found = sum(1 for kw in method_keywords if kw in prompt_lower)
        score += min(0.15, method_found * 0.05)

        # Has actionable instructions (not just vague)?
        instruction_verbs = [
            "evaluate", "assess", "compare", "check", "verify",
            "analyze", "determine", "identify", "consider",
        ]
        verb_found = sum(1 for v in instruction_verbs if v in prompt_lower)
        score += min(0.15, verb_found * 0.03)

        # Mentions domain awareness?
        domain_keywords = [
            "code", "answer", "instruction", "task", "output",
            "reference", "ground truth", "expected",
        ]
        domain_found = sum(1 for kw in domain_keywords if kw in prompt_lower)
        score += min(0.10, domain_found * 0.025)

        # Penalize very short prompts (< 20 words)
        word_count = len(prompt.split())
        if word_count < 20:
            score *= 0.6
        elif word_count < 40:
            score *= 0.8

        # Penalize vague language
        vague = ["etc", "whatever", "stuff", "things", "somehow"]
        vague_count = sum(1 for v in vague if v in prompt_lower)
        score -= vague_count * 0.05

        # Bonus for mentioning edge cases
        edge_keywords = ["edge case", "partial", "partially", "ambiguous", "uncertain"]
        if any(kw in prompt_lower for kw in edge_keywords):
            score += 0.05

        # Bonus for mentioning calibration
        cal_keywords = ["calibrat", "confidence", "certain", "uncertain"]
        if any(kw in prompt_lower for kw in cal_keywords):
            score += 0.05

        return max(0.0, min(1.0, score))

    def _score_user_template(self, template: str) -> float:
        """Score user template quality."""
        template_lower = template.lower()
        score = 0.0

        # Contains placeholders for task, output, reference?
        if "{task}" in template:
            score += 0.15
        if "{output}" in template:
            score += 0.15
        if "{reference}" in template:
            score += 0.10

        # Has clear section labels?
        label_keywords = ["task:", "output:", "reference:", "agent output:",
                          "expected:", "instructions:"]
        label_found = sum(1 for kw in label_keywords if kw in template_lower)
        score += min(0.15, label_found * 0.05)

        # Mentions specific evaluation aspects?
        eval_keywords = [
            "correct", "accurate", "complete", "follow",
            "requirement", "quality", "relevant", "criteria",
        ]
        eval_found = sum(1 for kw in eval_keywords if kw in template_lower)
        score += min(0.20, eval_found * 0.04)

        # Has structural formatting (newlines, sections)?
        if template.count("\n") >= 3:
            score += 0.10
        if any(sep in template for sep in ["---", "===", "***"]):
            score += 0.05

        # Has clear instruction to the judge?
        if any(v in template_lower for v in ["evaluate", "judge", "assess", "determine"]):
            score += 0.10

        return max(0.0, min(1.0, score))

    def _score_few_shot(self, examples: list[dict[str, str]]) -> float:
        """Score few-shot example quality and diversity."""
        if not examples:
            return 0.3  # Zero-shot is valid but not optimal

        score = 0.0
        n = len(examples)

        # Having examples is good, but diminishing returns after 5
        if n == 1:
            score += 0.20
        elif n == 2:
            score += 0.35
        elif n <= 5:
            score += 0.45
        else:
            score += 0.45 - 0.02 * (n - 5)  # slight penalty for too many

        # Check diversity: do examples cover different judgments?
        judgments = set()
        for ex in examples:
            j = ex.get("judgment", "").lower()
            if "pass" in j or "correct" in j or "good" in j:
                judgments.add("positive")
            if "fail" in j or "incorrect" in j or "bad" in j or "wrong" in j:
                judgments.add("negative")
            if "partial" in j or "mixed" in j:
                judgments.add("partial")
        diversity_bonus = min(0.20, len(judgments) * 0.07)
        score += diversity_bonus

        # Check if examples have all three fields
        complete = sum(
            1 for ex in examples
            if "input" in ex and "output" in ex and "judgment" in ex
        )
        completeness = complete / max(n, 1)
        score += 0.15 * completeness

        # Check if outputs contain structured format (JSON, scores)
        structured = sum(
            1 for ex in examples
            if any(marker in ex.get("output", "") for marker in ['"score"', '"pass"', '"reasoning"'])
        )
        score += min(0.10, structured / max(n, 1) * 0.10)

        # Check for variety in example content length (different complexity)
        if n >= 2:
            lengths = [len(ex.get("input", "")) for ex in examples]
            if max(lengths) > 1.5 * min(lengths):
                score += 0.05  # varied complexity

        return max(0.0, min(1.0, score))

    def _estimate_prompt_tokens(self) -> int:
        """Rough token estimate: ~4 chars per token."""
        total_chars = len(SYSTEM_PROMPT) + len(USER_TEMPLATE)
        for ex in FEW_SHOT_EXAMPLES:
            total_chars += sum(len(v) for v in ex.values())
        return total_chars // 4

    # ---- heuristic baseline judge ----

    def _heuristic_judge(
        self,
        agent_output: str,
        task: str,
        reference: str,
    ) -> dict[str, Any]:
        """Simple deterministic heuristic to judge output quality.

        Returns raw_score, base_confidence, and reasoning_parts.
        """
        output_lower = agent_output.lower().strip()
        task_lower = task.lower().strip()
        ref_lower = reference.lower().strip() if reference else ""
        reasoning: list[str] = []

        if not output_lower:
            return {
                "raw_score": 0.0,
                "base_confidence": 0.9,
                "reasoning_parts": ["Output is empty."],
            }

        score_components: list[float] = []

        # 1. Reference overlap (if reference exists)
        if ref_lower:
            ref_words = set(re.findall(r"\w+", ref_lower))
            out_words = set(re.findall(r"\w+", output_lower))
            if ref_words:
                overlap = len(ref_words & out_words) / len(ref_words)
                score_components.append(overlap)
                if overlap > 0.6:
                    reasoning.append(f"Good reference overlap ({overlap:.0%}).")
                else:
                    reasoning.append(f"Low reference overlap ({overlap:.0%}).")
            else:
                score_components.append(0.5)
        else:
            score_components.append(0.5)

        # 2. Task keyword coverage
        task_words = set(re.findall(r"\w{3,}", task_lower))
        out_words_full = set(re.findall(r"\w{3,}", output_lower))
        if task_words:
            task_coverage = len(task_words & out_words_full) / len(task_words)
            score_components.append(min(1.0, task_coverage * 1.5))
        else:
            score_components.append(0.5)

        # 3. Output length reasonableness
        out_len = len(output_lower.split())
        if out_len < 2:
            length_score = 0.3
            reasoning.append("Output is very short.")
        elif out_len < 10:
            length_score = 0.6
        elif out_len < 200:
            length_score = 0.8
        else:
            length_score = 0.7
            reasoning.append("Output is quite long.")
        score_components.append(length_score)

        # 4. Structural quality markers
        structure_score = 0.5
        if any(marker in agent_output for marker in ["def ", "class ", "return ", "import "]):
            structure_score = 0.7  # code-like structure
        if any(marker in agent_output for marker in ["\n- ", "\n* ", "\n1.", "\n2."]):
            structure_score = 0.7  # list structure
        if agent_output.count("\n") >= 2:
            structure_score += 0.1
        score_components.append(min(1.0, structure_score))

        raw_score = sum(score_components) / len(score_components)

        # Deterministic confidence based on input hash
        hash_val = int(hashlib.md5(
            (agent_output + task + reference).encode()
        ).hexdigest()[:8], 16)
        base_confidence = 0.5 + 0.3 * ((hash_val % 100) / 100.0)

        return {
            "raw_score": round(raw_score, 4),
            "base_confidence": round(base_confidence, 4),
            "reasoning_parts": reasoning,
        }
