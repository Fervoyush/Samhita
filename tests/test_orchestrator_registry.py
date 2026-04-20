"""Orchestrator abstraction + registry tests."""

import pytest

# Import drivers to trigger registration side-effects
import samhita.orchestrators.custom_driver  # noqa: F401
import samhita.orchestrators.langgraph_driver  # noqa: F401
from samhita.orchestrators.base import AgentOrchestrator
from samhita.orchestrators.registry import (
    get_orchestrator,
    list_orchestrators,
)


def test_drivers_registered() -> None:
    names = list_orchestrators()
    assert "langgraph" in names
    assert "custom" in names


def test_factory_returns_instance() -> None:
    orch = get_orchestrator("langgraph")
    assert isinstance(orch, AgentOrchestrator)
    assert orch.name == "langgraph"


def test_unknown_orchestrator_raises() -> None:
    with pytest.raises(KeyError):
        get_orchestrator("does_not_exist")


def test_orchestrator_abstract_methods_enforced() -> None:
    # Cannot instantiate AgentOrchestrator directly — enforced by ABC
    with pytest.raises(TypeError):
        AgentOrchestrator()  # type: ignore[abstract]
