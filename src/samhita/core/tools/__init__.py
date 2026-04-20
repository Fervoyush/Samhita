"""Tool registry — framework-agnostic.

Tools are pure typed async functions. Agent framework adapters wrap
them in their own conventions (LangGraph `@tool`, Agent Zero skill,
bespoke ReAct dispatch) but must not mutate the underlying functions.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from pydantic import BaseModel


@dataclass
class Tool:
    """A typed, framework-agnostic callable unit."""

    name: str
    description: str
    input_schema: type[BaseModel]
    output_schema: type[BaseModel]
    func: Callable[[BaseModel], Awaitable[BaseModel]]
    tags: list[str] = field(default_factory=list)


_REGISTRY: dict[str, Tool] = {}


def register_tool(tool: Tool, *, replace: bool = False) -> Tool:
    """Register a tool in the module-level registry.

    Idempotent by default: a second call with the same ``name`` returns
    the already-registered tool untouched. Pass ``replace=True`` to
    overwrite, which is useful for tests that need to swap in mocks.
    """
    existing = _REGISTRY.get(tool.name)
    if existing is not None and not replace:
        return existing
    _REGISTRY[tool.name] = tool
    return tool


def register_tools(tools: "list[Tool] | tuple[Tool, ...]") -> None:
    """Bulk-register tools, skipping any already-registered names."""
    for tool in tools:
        register_tool(tool)


def get_tool(name: str) -> Tool:
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY))
        raise KeyError(f"No tool {name!r}. Available: {available}")
    return _REGISTRY[name]


def list_tools(tag: str | None = None) -> list[Tool]:
    tools = list(_REGISTRY.values())
    if tag is not None:
        tools = [t for t in tools if tag in t.tags]
    return sorted(tools, key=lambda t: t.name)


def all_tools() -> dict[str, Tool]:
    return dict(_REGISTRY)


def clear_registry() -> None:
    """Test helper — clears the tool registry AND the bootstrap guard flags.

    Do not call from production code. Delayed import avoids circularity:
    :mod:`samhita.core.bootstrap` imports tool modules that live under
    ``core.tools``.
    """
    _REGISTRY.clear()
    from samhita.core.bootstrap import _reset_bootstrap_flags

    _reset_bootstrap_flags()
