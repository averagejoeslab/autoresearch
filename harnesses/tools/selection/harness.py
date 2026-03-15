"""
Tool selection harness – decide which tool(s) to use for a given task.

Exports:
    ToolSelector  – rank available tools for a task; decide if tools are needed.
"""

from __future__ import annotations

from typing import Any


class ToolSelector:
    """Baseline selector: always use tools, rank by keyword overlap.

    A smarter implementation might use embeddings, learned classifiers,
    or pattern-matching heuristics to determine when and which tools to
    use.
    """

    def select(self, task: str, available_tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return *available_tools* ordered by relevance to *task*.

        Each returned dict is a copy of the tool spec augmented with a
        ``relevance_score`` field.
        """
        task_words = self._tokenize(task)
        scored: list[tuple[float, int, dict[str, Any]]] = []

        for idx, tool in enumerate(available_tools):
            text = f"{tool.get('name', '')} {tool.get('description', '')} " + " ".join(
                str(v) for v in tool.get("parameters", {}).values() if isinstance(v, str)
            )
            tool_words = self._tokenize(text)
            overlap = len(task_words & tool_words) if task_words else 0
            score = overlap / max(len(task_words), 1)
            scored.append((score, idx, tool))

        scored.sort(key=lambda t: (-t[0], t[1]))
        return [
            {**tool, "relevance_score": round(s, 4)}
            for s, _, tool in scored
        ]

    def should_use_tool(self, task: str) -> bool:
        """Decide whether *task* requires a tool at all.

        Baseline: always returns True (conservative – never skips tools).
        """
        return True

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return set(text.lower().split())
