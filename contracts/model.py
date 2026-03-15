"""
Inner model abstraction for benchmarks.

Each benchmark needs an LLM to act as the "brain" inside the harness.
This module provides a unified interface for local and API-based models.

Usage:
    from contracts import get_model
    model = get_model()
    response = model.complete("What is 2+2?")
    response = model.complete_with_tools("Find the file", tools=[...])

Configure via environment:
    MODEL_BACKEND=local    (default) — uses local LLM
    MODEL_BACKEND=api      — uses API provider
    MODEL_NAME=...         — model name (provider-specific)
    MODEL_API_KEY=...      — API key (for api backend)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


@dataclass
class ModelResponse:
    """Unified response from any model backend."""

    content: str
    tool_calls: list[dict[str, Any]] | None = None
    usage: dict[str, int] | None = None  # {"input_tokens": ..., "output_tokens": ...}
    raw: Any = None  # provider-specific raw response


class ModelBackend:
    """Base class for model backends."""

    def complete(self, prompt: str, system: str = "", temperature: float = 0.0) -> ModelResponse:
        raise NotImplementedError

    def complete_with_tools(
        self,
        prompt: str,
        tools: list[dict[str, Any]],
        system: str = "",
        temperature: float = 0.0,
    ) -> ModelResponse:
        raise NotImplementedError


class LocalModel(ModelBackend):
    """Local LLM via llama-cpp-python or mlx-lm."""

    def __init__(self, model_name: str = ""):
        self.model_name = model_name or "default"
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        try:
            from llama_cpp import Llama

            model_path = os.environ.get("LOCAL_MODEL_PATH", "")
            if model_path:
                self._model = Llama(model_path=model_path, n_ctx=4096)
                self._backend = "llama_cpp"
                return
        except ImportError:
            pass
        raise RuntimeError(
            "No local model backend available. Install llama-cpp-python or set MODEL_BACKEND=api"
        )

    def complete(self, prompt: str, system: str = "", temperature: float = 0.0) -> ModelResponse:
        self._load()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        result = self._model.create_chat_completion(messages=messages, temperature=temperature)
        content = result["choices"][0]["message"]["content"]
        usage = result.get("usage", {})
        return ModelResponse(
            content=content,
            usage={"input_tokens": usage.get("prompt_tokens", 0), "output_tokens": usage.get("completion_tokens", 0)},
            raw=result,
        )

    def complete_with_tools(
        self, prompt: str, tools: list[dict[str, Any]], system: str = "", temperature: float = 0.0
    ) -> ModelResponse:
        tool_descriptions = "\n".join(
            f"- {t['name']}: {t.get('description', '')}" for t in tools
        )
        augmented = f"{prompt}\n\nAvailable tools:\n{tool_descriptions}\n\nRespond with the tool to use and parameters as JSON."
        return self.complete(augmented, system=system, temperature=temperature)


class APIModel(ModelBackend):
    """API-based model (any OpenAI-compatible endpoint)."""

    def __init__(self, model_name: str = ""):
        self.model_name = model_name or os.environ.get("MODEL_NAME", "gpt-4o-mini")
        self.api_key = os.environ.get("MODEL_API_KEY", "")
        self.base_url = os.environ.get("MODEL_BASE_URL", "https://api.openai.com/v1")
        self._client = None

    def _load(self):
        if self._client is not None:
            return
        try:
            from openai import OpenAI

            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        except ImportError:
            raise RuntimeError("Install openai package for API backend: pip install openai")

    def complete(self, prompt: str, system: str = "", temperature: float = 0.0) -> ModelResponse:
        self._load()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        result = self._client.chat.completions.create(
            model=self.model_name, messages=messages, temperature=temperature
        )
        choice = result.choices[0]
        usage = result.usage
        return ModelResponse(
            content=choice.message.content or "",
            usage={"input_tokens": usage.prompt_tokens, "output_tokens": usage.completion_tokens} if usage else None,
            raw=result,
        )

    def complete_with_tools(
        self, prompt: str, tools: list[dict[str, Any]], system: str = "", temperature: float = 0.0
    ) -> ModelResponse:
        self._load()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        openai_tools = [{"type": "function", "function": t} for t in tools]
        result = self._client.chat.completions.create(
            model=self.model_name, messages=messages, tools=openai_tools, temperature=temperature
        )
        choice = result.choices[0]
        tool_calls = None
        if choice.message.tool_calls:
            tool_calls = [
                {"name": tc.function.name, "arguments": tc.function.arguments}
                for tc in choice.message.tool_calls
            ]
        usage = result.usage
        return ModelResponse(
            content=choice.message.content or "",
            tool_calls=tool_calls,
            usage={"input_tokens": usage.prompt_tokens, "output_tokens": usage.completion_tokens} if usage else None,
            raw=result,
        )


def get_model(backend: str | None = None, model_name: str = "") -> ModelBackend:
    """Get a model backend based on configuration.

    Args:
        backend: "local" or "api". Defaults to MODEL_BACKEND env var or "local".
        model_name: Model name. Defaults to MODEL_NAME env var.
    """
    backend = backend or os.environ.get("MODEL_BACKEND", "local")
    if backend == "api":
        return APIModel(model_name)
    return LocalModel(model_name)
