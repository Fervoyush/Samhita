"""Custom ReAct-loop orchestrator — no agent framework.

Purpose: prove the hexagonal abstraction holds. By building the simplest
possible orchestrator that uses no external agent framework, we isolate
which value the framework actually adds. If results are comparable, the
framework is decorative. If LangGraph wins substantially, we know why.

Implementation lands in Phase 2 — for now the driver is registered so
benchmark harnesses can discover and instantiate it.
"""

from __future__ import annotations

from samhita.core.schemas import KGResult, KGSpec
from samhita.orchestrators.base import AgentOrchestrator
from samhita.orchestrators.registry import register_orchestrator


@register_orchestrator("custom")
class CustomOrchestrator(AgentOrchestrator):
    """A bare-bones plan/ReAct loop with no framework dependencies."""

    async def plan(self, nl_request: str) -> KGSpec:  # noqa: ARG002
        raise NotImplementedError("Custom planner — Phase 2")

    async def execute(self, spec: KGSpec) -> KGResult:  # noqa: ARG002
        raise NotImplementedError("Custom executor — Phase 2")
