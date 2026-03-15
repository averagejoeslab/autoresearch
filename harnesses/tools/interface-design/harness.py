"""
Tool interface-design harness – enhance tool descriptions for better selection.

Exports:
    ToolDescriber  – takes raw tool specs and produces enhanced descriptions.
"""

from __future__ import annotations

from typing import Any


class ToolDescriber:
    """Baseline describer: pass-through (no enhancement).

    A real implementation would rewrite names, add examples, expand
    parameter descriptions, etc.  The baseline simply returns the
    input spec unchanged so the benchmark can establish a floor score.
    """

    def describe(self, tool_spec: dict[str, Any]) -> dict[str, Any]:
        """Return an enhanced version of *tool_spec*.

        The returned dict must contain at least:
            name         – tool name
            description  – human-readable description
            parameters   – list of parameter dicts
        Optional (may improve selection accuracy):
            examples     – list of example usage strings
            usage_hints  – additional keywords / synonyms
            category     – tool category
        """
        # Baseline: return a copy with no modifications
        return dict(tool_spec)
