"""AgentOrchestrator — the contract every framework adapter implements."""

from __future__ import annotations

from abc import ABC, abstractmethod

from samhita.core.schemas import KGResult, KGSpec


class AgentOrchestrator(ABC):
    """Contract every agent-framework adapter must implement.

    Drivers are free to use any orchestration approach internally
    (LangGraph, Agent Zero, custom ReAct loop, DSPy), as long as they
    produce the same `KGSpec -> KGResult` transformation. Swapping
    drivers is a config flag, not a rewrite.
    """

    name: str = "abstract"

    @abstractmethod
    async def plan(self, nl_request: str) -> KGSpec:
        """Parse a natural-language request into a structured KGSpec."""
        raise NotImplementedError

    @abstractmethod
    async def execute(self, spec: KGSpec) -> KGResult:
        """Run the construction pipeline and produce a KGResult."""
        raise NotImplementedError

    async def build(self, nl_request: str) -> KGResult:
        """Convenience: plan + execute in one call.

        Subclasses should rarely override this — override `plan` and
        `execute` instead.
        """
        spec = await self.plan(nl_request)
        return await self.execute(spec)
