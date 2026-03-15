"""
Tool-use training data generation and curation.

Generates and validates training examples for tool calling.

Exports:
    ToolDataGenerator  -- generate and validate tool-call training examples.
"""

from __future__ import annotations

from typing import Any


class ToolDataGenerator:
    """Generate and validate tool-call training examples.

    Baseline strategy
    -----------------
    Generate "Use {tool_name}" as query, fill all params with
    placeholder values. Validate by checking structural completeness.
    """

    name: str = "tool_data_generator"

    # Placeholder values by parameter type
    _PLACEHOLDERS: dict[str, Any] = {
        "string": "placeholder",
        "integer": 0,
        "number": 0.0,
        "boolean": True,
        "array": [],
        "object": {},
    }

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def generate_example(self, tool_spec: dict[str, Any]) -> dict[str, Any]:
        """Produce a training example for a given tool specification.

        Parameters
        ----------
        tool_spec : dict
            Tool specification with:
              - "name"        : str
              - "description" : str
              - "parameters"  : dict mapping param_name -> {"type": str, ...}

        Returns
        -------
        dict with:
            "user_query"    : str  -- the user's natural language request
            "tool_call"     : dict -- {"name": str, "arguments": dict}
            "tool_spec"     : dict -- the original tool spec (for reference)
        """
        name = tool_spec.get("name", "unknown_tool")
        description = tool_spec.get("description", "")
        params = tool_spec.get("parameters", {})

        # Baseline: very simple query
        query = f"Use {name}"
        if description:
            query = f"Use {name} to {description.rstrip('.').lower()}"

        # Fill all parameters with placeholder values
        arguments = {}
        for param_name, param_info in params.items():
            param_type = param_info.get("type", "string") if isinstance(param_info, dict) else "string"
            arguments[param_name] = self._PLACEHOLDERS.get(param_type, "placeholder")

        return {
            "user_query": query,
            "tool_call": {
                "name": name,
                "arguments": arguments,
            },
            "tool_spec": tool_spec,
        }

    def validate_example(self, example: dict[str, Any]) -> bool:
        """Check if a training example is valid.

        Parameters
        ----------
        example : dict
            A training example with "user_query", "tool_call", "tool_spec".

        Returns
        -------
        bool
            True if the example is structurally valid.

        Validation rules:
        1. Must have user_query (non-empty string).
        2. Must have tool_call with "name" and "arguments".
        3. Tool name must match the spec name.
        4. All required parameters must be present.
        5. No extra parameters not in the spec.
        """
        # Check user_query
        query = example.get("user_query", "")
        if not isinstance(query, str) or not query.strip():
            return False

        # Check tool_call structure
        tool_call = example.get("tool_call")
        if not isinstance(tool_call, dict):
            return False
        if "name" not in tool_call or "arguments" not in tool_call:
            return False

        # Check tool_spec exists
        spec = example.get("tool_spec")
        if not isinstance(spec, dict):
            return False

        # Tool name must match spec
        if tool_call["name"] != spec.get("name"):
            return False

        # Check parameter coverage
        spec_params = spec.get("parameters", {})
        call_args = tool_call.get("arguments", {})

        # No extra params
        for arg_name in call_args:
            if arg_name not in spec_params:
                return False

        # All required params present
        for param_name, param_info in spec_params.items():
            required = True
            if isinstance(param_info, dict):
                required = param_info.get("required", True)
            if required and param_name not in call_args:
                return False

        return True
