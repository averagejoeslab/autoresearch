"""
Contamination Resistance primitive.

Transforms benchmark tasks to prevent memorization while
preserving the tested capability.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any


class ContaminationGuard:
    """Create equivalent task variants that resist memorization.

    Baseline strategy
    -----------------
    * transform: replace all numbers in the task with deterministically
      different numbers (based on seed), keeping structure identical.
    * verify_equivalence: check that the transformed task has the same
      structure (length, word pattern) as the original.
    """

    name: str = "contamination_guard"

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def transform(
        self,
        task: dict[str, Any],
        seed: int,
    ) -> dict[str, Any]:
        """Create a variant of *task* that tests the same skill.

        Parameters
        ----------
        task : dict
            Must have:
              - "prompt"          : str
              - "expected_answer" : str
              - "tags"            : list[str]  (skill labels)
        seed : int
            Deterministic randomization seed.

        Returns
        -------
        dict
            Variant with same keys, different surface details.
        """
        prompt = task.get("prompt", "")
        answer = task.get("expected_answer", "")

        # Find all integers in the prompt
        numbers = re.findall(r"\b(\d+)\b", prompt)
        new_prompt = prompt
        new_answer = answer

        # Create a deterministic mapping for each unique number
        number_map: dict[str, str] = {}
        for num in set(numbers):
            # Generate a new number by hashing original + seed
            h = int(hashlib.md5(f"{num}:{seed}".encode()).hexdigest()[:8], 16)
            original = int(num)
            # Keep the same order of magnitude
            if original == 0:
                new_num = h % 10
            else:
                magnitude = len(num)
                low = 10 ** (magnitude - 1) if magnitude > 1 else 1
                high = 10 ** magnitude - 1
                new_num = low + (h % (high - low + 1))
            # Avoid mapping to the same number
            if new_num == original:
                new_num = original + 1
                if new_num > 10 ** magnitude - 1 and magnitude > 1:
                    new_num = 10 ** (magnitude - 1)
            number_map[num] = str(new_num)

        # Apply replacements (whole-word only, longest first to avoid partial matches)
        for old_num in sorted(number_map.keys(), key=len, reverse=True):
            new_num = number_map[old_num]
            new_prompt = re.sub(rf"\b{old_num}\b", new_num, new_prompt)
            new_answer = re.sub(rf"\b{old_num}\b", new_num, new_answer)

        return {
            "prompt": new_prompt,
            "expected_answer": new_answer,
            "tags": list(task.get("tags", [])),
            "variant_seed": seed,
            "original_prompt": task.get("prompt", ""),
        }

    def verify_equivalence(
        self,
        original: dict[str, Any],
        transformed: dict[str, Any],
    ) -> bool:
        """Check whether *transformed* tests the same skill as *original*.

        Parameters
        ----------
        original : dict
            Original task (with "prompt", "tags").
        transformed : dict
            Variant task.

        Returns
        -------
        bool
            True if they are structurally equivalent.
        """
        orig_prompt = original.get("prompt", "")
        trans_prompt = transformed.get("prompt", "")

        # Baseline: check structural similarity
        # 1. Same tags
        if set(original.get("tags", [])) != set(transformed.get("tags", [])):
            return False

        # 2. Same word count (within tolerance)
        orig_words = len(orig_prompt.split())
        trans_words = len(trans_prompt.split())
        if orig_words == 0:
            return trans_words == 0
        ratio = trans_words / orig_words
        if not (0.8 <= ratio <= 1.2):
            return False

        # 3. Same non-numeric word structure
        def strip_numbers(text: str) -> str:
            return re.sub(r"\b\d+\b", "NUM", text)

        if strip_numbers(orig_prompt) == strip_numbers(trans_prompt):
            return True

        # Fallback: at least 50% word overlap (excluding numbers)
        def non_numeric_words(text: str) -> set[str]:
            return {w.lower() for w in text.split() if not w.isdigit()}

        orig_set = non_numeric_words(orig_prompt)
        trans_set = non_numeric_words(trans_prompt)
        if not orig_set:
            return True
        overlap = len(orig_set & trans_set) / len(orig_set)
        return overlap >= 0.5
