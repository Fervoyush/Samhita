"""LLM client implementations.

Each client satisfies :class:`samhita.core.llm.LLMClient`. Adding a
provider (OpenAI, Moonshot/Kimi, local) means creating a new module
and registering a factory function below.
"""

from __future__ import annotations

from collections.abc import Callable

from samhita.core.llm import LLMClient

_FACTORIES: dict[str, Callable[..., LLMClient]] = {}


def register_llm_factory(provider: str, factory: Callable[..., LLMClient]) -> None:
    """Register a provider factory (e.g. 'anthropic' -> AnthropicClient)."""
    _FACTORIES[provider] = factory


def get_llm_client(provider: str, **kwargs: object) -> LLMClient:
    if provider not in _FACTORIES:
        available = ", ".join(sorted(_FACTORIES)) or "<none>"
        raise KeyError(f"No LLM factory for provider {provider!r}. Available: {available}")
    return _FACTORIES[provider](**kwargs)


def list_providers() -> list[str]:
    return sorted(_FACTORIES)
