"""
Difficulty Calibration primitive.

Calibrates task difficulty based on actual agent performance data
and predicts difficulty for new tasks.
"""

from __future__ import annotations

from typing import Any


class DifficultyCalibrater:
    """Calibrate and predict task difficulty.

    Baseline strategy
    -----------------
    * calibrate: assign difficulty proportional to (1 - solve_rate),
      mapped to 1-5 scale.
    * predict_difficulty: use task description length as a proxy
      (longer = harder).
    """

    name: str = "difficulty_calibrater"

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def calibrate(
        self,
        tasks: list[dict[str, Any]],
        results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Re-assign difficulty levels using actual performance data.

        Parameters
        ----------
        tasks : list[dict]
            Each dict has at least:
              - "id"         : str | int
              - "prompt"     : str
              - "difficulty" : int  (original label, may be wrong)
        results : list[dict]
            Each dict has:
              - "task_id"    : str | int
              - "solved"     : bool
              - "agent_id"   : str

        Returns
        -------
        list[dict]
            Same tasks with updated "difficulty" (int 1-5) and added
            "solve_rate" (float 0-1).
        """
        # Build solve-rate per task
        task_attempts: dict[Any, list[bool]] = {}
        for r in results:
            tid = r["task_id"]
            task_attempts.setdefault(tid, []).append(r["solved"])

        calibrated = []
        for task in tasks:
            tid = task["id"]
            attempts = task_attempts.get(tid, [])
            if attempts:
                solve_rate = sum(attempts) / len(attempts)
            else:
                solve_rate = 0.5  # no data, assume medium

            # Map solve_rate to difficulty: 0% solved = difficulty 5, 100% = difficulty 1
            difficulty = max(1, min(5, round(1 + 4 * (1 - solve_rate))))

            calibrated.append({
                **task,
                "difficulty": difficulty,
                "solve_rate": solve_rate,
            })

        return calibrated

    def predict_difficulty(self, task: dict[str, Any]) -> float:
        """Predict difficulty for a new task (no performance data).

        Parameters
        ----------
        task : dict
            Must have "prompt" : str.

        Returns
        -------
        float
            Predicted difficulty on a 1.0-5.0 scale.
        """
        # Baseline: use description length as a proxy
        prompt = task.get("prompt", "")
        word_count = len(prompt.split())

        # Map word count to 1-5 range
        # < 5 words = 1.0, > 40 words = 5.0, linear between
        difficulty = 1.0 + 4.0 * min(max(word_count - 5, 0), 35) / 35.0
        return round(min(5.0, max(1.0, difficulty)), 2)
