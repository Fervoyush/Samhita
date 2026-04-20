"""LangGraph orchestrator driver — v1 reference implementation.

Graph topology:

    START -> plan -> fetch -> extract -> normalize -> flag_conflicts -> write -> END

Each node updates a `RunState`. This module ships the plan node fully
wired (LLM call -> structured KGSpec) and wires the remaining nodes as
pass-through placeholders whose implementations land alongside the
corresponding tool implementations. The graph shape itself is final.
"""

from __future__ import annotations

import time
from typing import Annotated, Any, TypedDict

from samhita.core.bootstrap import bootstrap_llm_providers, bootstrap_tools
from samhita.core.llm import LLMClient, Message
from samhita.core.llm_clients import get_llm_client
from samhita.core.prompts import PLANNER_SYSTEM_PROMPT
from samhita.core.recipes import list_recipes
from samhita.core.schemas import KGResult, KGSpec, RunState
from samhita.orchestrators.base import AgentOrchestrator
from samhita.orchestrators.registry import register_orchestrator


class _GraphState(TypedDict, total=False):
    """LangGraph state dict — thin mirror of RunState plus carried artifacts."""

    nl_request: str
    spec: KGSpec
    run_state: RunState
    fetched_papers: Annotated[list[dict[str, Any]], "fetched papers (raw)"]
    extracted_entities: list[dict[str, Any]]
    extracted_edges: list[dict[str, Any]]
    started_at: float


@register_orchestrator("langgraph")
class LangGraphOrchestrator(AgentOrchestrator):
    """LangGraph-based orchestration of the Samhita pipeline."""

    def __init__(
        self,
        llm_provider: str = "anthropic",
        llm_model: str = "claude-sonnet-4-6",
        llm: LLMClient | None = None,
    ) -> None:
        bootstrap_tools()
        bootstrap_llm_providers()
        self._llm: LLMClient = llm or get_llm_client(llm_provider, model=llm_model)
        self._graph = self._build_graph()

    # ------------------------------------------------------------------ plan
    async def plan(self, nl_request: str) -> KGSpec:
        recipes_block = _recipes_prompt_block()
        messages = [
            Message(role="system", content=PLANNER_SYSTEM_PROMPT + "\n\n" + recipes_block),
            Message(role="user", content=nl_request),
        ]
        response = await self._llm.complete(messages=messages, schema=KGSpec)
        parsed = response.parsed
        if not isinstance(parsed, KGSpec):
            raise RuntimeError(
                "Planner LLM did not return a KGSpec — check model tool-use "
                "support or schema compatibility."
            )
        # The LLM doesn't see the raw NL request in the schema; attach it.
        if not parsed.original_request:
            parsed = parsed.model_copy(update={"original_request": nl_request})
        return parsed

    # ------------------------------------------------------------------ execute
    async def execute(self, spec: KGSpec) -> KGResult:
        state: _GraphState = {
            "nl_request": spec.original_request,
            "spec": spec,
            "run_state": RunState(spec=spec, status="running"),
            "fetched_papers": [],
            "extracted_entities": [],
            "extracted_edges": [],
            "started_at": time.monotonic(),
        }
        final_state = await self._run_graph(state)

        run_state: RunState = final_state["run_state"]
        run_state.status = "completed"
        return KGResult(
            spec=spec,
            entities=[],   # populated when extract + normalize stages land
            edges=[],      # populated when extract + flag stages land
            state=run_state,
            build_duration_seconds=time.monotonic() - state["started_at"],
        )

    # ------------------------------------------------------------------ graph wiring
    def _build_graph(self) -> Any:
        """Compile the LangGraph state graph.

        Returns the compiled graph. If langgraph is not installed the
        driver falls back to a simple linear dispatcher that preserves
        the same contract (useful for environments where langgraph
        cannot be installed yet).
        """
        try:
            from langgraph.graph import END, START, StateGraph
        except ImportError:
            # Fallback linear runner — same semantics, no framework dep.
            return _LinearFallbackGraph(
                [
                    _fetch_node,
                    _extract_node,
                    _normalize_node,
                    _flag_conflicts_node,
                    _write_node,
                ]
            )

        graph: StateGraph = StateGraph(_GraphState)
        graph.add_node("fetch", _fetch_node)
        graph.add_node("extract", _extract_node)
        graph.add_node("normalize", _normalize_node)
        graph.add_node("flag_conflicts", _flag_conflicts_node)
        graph.add_node("write", _write_node)

        graph.add_edge(START, "fetch")
        graph.add_edge("fetch", "extract")
        graph.add_edge("extract", "normalize")
        graph.add_edge("normalize", "flag_conflicts")
        graph.add_edge("flag_conflicts", "write")
        graph.add_edge("write", END)

        return graph.compile()

    async def _run_graph(self, state: _GraphState) -> _GraphState:
        if isinstance(self._graph, _LinearFallbackGraph):
            return await self._graph.run(state)
        # LangGraph compiled graphs expose .ainvoke
        return await self._graph.ainvoke(state)


# ---------------------------------------------------------------------------
# Nodes (placeholder implementations — real work lands with the tool impls)
# ---------------------------------------------------------------------------


async def _fetch_node(state: _GraphState) -> _GraphState:
    # Real implementation calls the registered fetch tools based on
    # state["spec"].sources and state["spec"].seeds.
    state["run_state"].errors.append("fetch_node: stub — no papers pulled yet")
    return state


async def _extract_node(state: _GraphState) -> _GraphState:
    state["run_state"].errors.append("extract_node: stub — extraction not wired")
    return state


async def _normalize_node(state: _GraphState) -> _GraphState:
    state["run_state"].errors.append("normalize_node: stub — normalization not wired")
    return state


async def _flag_conflicts_node(state: _GraphState) -> _GraphState:
    state["run_state"].errors.append("flag_conflicts_node: stub — no edges to compare yet")
    return state


async def _write_node(state: _GraphState) -> _GraphState:
    state["run_state"].errors.append("write_node: stub — Biocypher writer pending")
    return state


# ---------------------------------------------------------------------------
# LangGraph-less fallback runner (same contract, no framework dependency)
# ---------------------------------------------------------------------------


class _LinearFallbackGraph:
    """Minimal linear dispatcher used when langgraph is not installed."""

    def __init__(self, nodes: list[Any]) -> None:
        self._nodes = nodes

    async def run(self, state: _GraphState) -> _GraphState:
        current = state
        for node in self._nodes:
            current = await node(current)
        return current


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------


def _recipes_prompt_block() -> str:
    """Inline the recipe vocabulary in the planner's system prompt."""
    lines = ["Available recipes:"]
    for recipe in list_recipes():
        name = recipe["name"].value
        desc = recipe["description"]
        etypes = ", ".join(e.value for e in recipe["entity_types"])
        rtypes = ", ".join(r.value for r in recipe["relation_types"])
        sources = ", ".join(s.value for s in recipe["sources"])
        lines.append(
            f"- {name}: {desc}\n"
            f"  entity_types: [{etypes}]\n"
            f"  relation_types: [{rtypes}]\n"
            f"  sources: [{sources}]"
        )
    return "\n".join(lines)
