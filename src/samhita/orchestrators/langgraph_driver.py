"""LangGraph orchestrator driver — v1 reference implementation.

Graph topology, node implementations, and tool-binding come in
Week 1 Day 6-7. This module currently registers the driver under
the `langgraph` name and stubs the two abstract methods so the
registry-discovery tests pass.
"""

from __future__ import annotations

from samhita.core.schemas import KGResult, KGSpec
from samhita.orchestrators.base import AgentOrchestrator
from samhita.orchestrators.registry import register_orchestrator


@register_orchestrator("langgraph")
class LangGraphOrchestrator(AgentOrchestrator):
    """LangGraph-based orchestration of the Samhita pipeline."""

    def __init__(self) -> None:
        # Graph compilation deferred until plan/execute are implemented
        self._graph = None

    async def plan(self, nl_request: str) -> KGSpec:  # noqa: ARG002
        raise NotImplementedError("LangGraph planner node — Week 1 Day 6")

    async def execute(self, spec: KGSpec) -> KGResult:  # noqa: ARG002
        raise NotImplementedError("LangGraph execution graph — Week 1 Day 6-7")
