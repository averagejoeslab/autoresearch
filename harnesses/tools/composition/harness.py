"""
Tool composition harness – chain multiple tools for multi-step tasks.

Exports:
    ToolComposer  – plan a chain of tool calls and validate it.
"""

from __future__ import annotations

from typing import Any


class ToolComposer:
    """Baseline composer: chain tools in listed order, pass all outputs forward.

    A better implementation would parse the task, match required
    capabilities to tools, and construct explicit parameter mappings.
    """

    def plan_chain(
        self, task: str, available_tools: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Return an ordered list of tool-call specifications.

        Each element is a dict with:
            tool   – name of the tool to call
            inputs – dict mapping param names to either literal values or
                     references like ``{"from_step": 0, "output_key": "result"}``
        """
        if not available_tools:
            return []

        chain: list[dict[str, Any]] = []
        task_words = set(task.lower().split())

        # Baseline: pick tools whose descriptions share words with the task,
        # order by occurrence in the available list, and wire outputs forward.
        selected: list[dict[str, Any]] = []
        for tool in available_tools:
            text = f"{tool.get('name', '')} {tool.get('description', '')}"
            tool_words = set(text.lower().split())
            if task_words & tool_words:
                selected.append(tool)

        # If nothing matched, just use first tool
        if not selected:
            selected = available_tools[:1]

        for step_idx, tool in enumerate(selected):
            step: dict[str, Any] = {
                "tool": tool["name"],
                "inputs": {},
            }
            params = tool.get("parameters", {})
            for pname in params:
                if step_idx == 0:
                    step["inputs"][pname] = f"<from_task:{pname}>"
                else:
                    step["inputs"][pname] = {
                        "from_step": step_idx - 1,
                        "output_key": "result",
                    }
            chain.append(step)

        return chain

    def validate_chain(self, chain: list[dict[str, Any]]) -> bool:
        """Check basic structural validity of a chain.

        Rules:
        1. Non-empty chain.
        2. Every step has ``tool`` (str) and ``inputs`` (dict).
        3. Step references (``from_step``) point to earlier steps.
        """
        if not chain:
            return False

        for idx, step in enumerate(chain):
            if not isinstance(step, dict):
                return False
            if "tool" not in step or not isinstance(step["tool"], str):
                return False
            inputs = step.get("inputs", {})
            if not isinstance(inputs, dict):
                return False
            for v in inputs.values():
                if isinstance(v, dict) and "from_step" in v:
                    ref = v["from_step"]
                    if not isinstance(ref, int) or ref < 0 or ref >= idx:
                        return False
        return True
