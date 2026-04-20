"""Orchestrator registry — discovery + factory.

Drivers self-register via the `@register_orchestrator(name)` decorator.
The CLI / library consumers pick a driver by name at runtime.
"""

from __future__ import annotations

from typing import Any

from samhita.orchestrators.base import AgentOrchestrator

_ORCHESTRATORS: dict[str, type[AgentOrchestrator]] = {}


def register_orchestrator(name: str):
    """Decorator registering an AgentOrchestrator implementation."""

    def decorator(cls: type[AgentOrchestrator]) -> type[AgentOrchestrator]:
        if name in _ORCHESTRATORS:
            raise ValueError(f"Orchestrator {name!r} is already registered")
        cls.name = name
        _ORCHESTRATORS[name] = cls
        return cls

    return decorator


def get_orchestrator(name: str, **kwargs: Any) -> AgentOrchestrator:
    if name not in _ORCHESTRATORS:
        available = ", ".join(sorted(_ORCHESTRATORS)) or "<none registered>"
        raise KeyError(
            f"No orchestrator registered as {name!r}. Available: {available}"
        )
    return _ORCHESTRATORS[name](**kwargs)


def list_orchestrators() -> list[str]:
    return sorted(_ORCHESTRATORS)


def clear_registry() -> None:
    """Test helper — do not call from production code."""
    _ORCHESTRATORS.clear()
