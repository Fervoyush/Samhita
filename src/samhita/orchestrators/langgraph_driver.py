"""LangGraph orchestrator driver — v1 reference implementation.

Graph topology:

    START -> fetch -> extract -> normalize -> flag_conflicts -> write -> END

Each node updates a shared dict-shaped state that mirrors ``RunState``.
Nodes dispatch to registered tools rather than implementing work inline,
which keeps the agent framework thin and the tools reusable across
driver implementations.
"""

from __future__ import annotations

import time
from typing import Any, TypedDict

from samhita.core.bootstrap import bootstrap_llm_providers, bootstrap_tools
from samhita.core.fetchers import (
    fetch_chembl_for_spec,
    fetch_drugbank_for_spec,
    fetch_opentargets_for_spec,
)
from samhita.core.fetchers._helpers import slugify
from samhita.core.llm import LLMClient, Message
from samhita.core.llm_clients import get_llm_client
from samhita.core.prompts import PLANNER_SYSTEM_PROMPT
from samhita.core.recipes import list_recipes
from samhita.core.schemas import (
    Edge,
    Entity,
    EntityType,
    Identifier,
    KGResult,
    KGSpec,
    NamespaceName,
    Provenance,
    RelationType,
    RunState,
    RunStatus,
    SectionType,
    SourceName,
    SourceType,
)
from samhita.core.tools import all_tools, get_tool
from samhita.core.tools.extract import (
    ExtractedEdge,
    ExtractFromTextInput,
    ExtractFromTextOutput,
    ExtractionCandidate,
    register_extract_tools,
)
from samhita.core.tools.fetch import (
    PMCFetchInput,
    PMCFetchOutput,
    PubMedAbstractInput,
    PubMedSearchInput,
)
from samhita.core.tools.normalize import NormalizeEntityInput, NormalizeEntityOutput
from samhita.core.tools.write import BiocypherWriteInput
from samhita.orchestrators.base import AgentOrchestrator
from samhita.orchestrators.registry import register_orchestrator


class _GraphState(TypedDict, total=False):
    nl_request: str
    spec: KGSpec
    run_state: RunState
    fetched_papers: list[dict[str, Any]]
    raw_entities: list[tuple[ExtractionCandidate, Provenance]]
    raw_edges: list[tuple[ExtractedEdge, Provenance]]
    # Structured sources (OpenTargets / ChEMBL / DrugBank) produce
    # canonical Entity / Edge directly, bypassing extract + normalize.
    structured_entities: list[Entity]
    structured_edges: list[Edge]
    normalized_entities: dict[str, Entity]  # name -> Entity
    final_edges: list[Edge]
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
        register_extract_tools(self._llm)
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
        if not parsed.original_request:
            parsed = parsed.model_copy(update={"original_request": nl_request})
        return parsed

    # ------------------------------------------------------------------ execute
    async def execute(self, spec: KGSpec) -> KGResult:
        state: _GraphState = {
            "nl_request": spec.original_request,
            "spec": spec,
            "run_state": RunState(spec=spec, status=RunStatus.RUNNING),
            "fetched_papers": [],
            "raw_entities": [],
            "raw_edges": [],
            "structured_entities": [],
            "structured_edges": [],
            "normalized_entities": {},
            "final_edges": [],
            "started_at": time.monotonic(),
        }
        final_state = await self._run_graph(state)
        run_state: RunState = final_state["run_state"]
        run_state.status = RunStatus.COMPLETED

        return KGResult(
            spec=spec,
            entities=list(final_state.get("normalized_entities", {}).values()),
            edges=list(final_state.get("final_edges", [])),
            state=run_state,
            build_duration_seconds=time.monotonic() - state["started_at"],
        )

    # ------------------------------------------------------------------ graph wiring
    def _build_graph(self) -> Any:
        try:
            from langgraph.graph import END, START, StateGraph
        except ImportError:
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
        return await self._graph.ainvoke(state)


# =============================================================================
# Nodes
# =============================================================================


async def _fetch_node(state: _GraphState) -> _GraphState:
    spec: KGSpec = state["spec"]
    run_state: RunState = state["run_state"]
    papers: list[dict[str, Any]] = []
    structured_entities: list[Entity] = list(state.get("structured_entities", []))
    structured_edges: list[Edge] = list(state.get("structured_edges", []))

    # --- PubMed / PMC literature --------------------------------------------
    if SourceName.PUBMED_CENTRAL in spec.sources or SourceName.PUBMED_ABSTRACTS in spec.sources:
        query = _build_pubmed_query(spec)
        search_tool = get_tool("search_pubmed")
        abstract_tool = get_tool("fetch_pubmed_abstract")
        pmc_tool = get_tool("fetch_pmc_paper")

        search_out = await search_tool.func(
            PubMedSearchInput(query=query, max_results=min(spec.max_papers, 50))
        )
        pmids = getattr(search_out, "pmids", []) or []

        for pmid in pmids:
            abstract_out = await abstract_tool.func(PubMedAbstractInput(pmid=pmid))
            pmc_id = getattr(abstract_out, "pmc_id", None)
            sections: dict[str, str] = {}
            title = getattr(abstract_out, "title", "")

            if pmc_id and SourceName.PUBMED_CENTRAL in spec.sources:
                pmc_out = await pmc_tool.func(PMCFetchInput(pmc_id=pmc_id))
                if isinstance(pmc_out, PMCFetchOutput) and not pmc_out.error:
                    sections = dict(pmc_out.sections)
                    title = pmc_out.title or title

            if not sections:
                sections = {SectionType.ABSTRACT.value: getattr(abstract_out, "abstract", "")}

            papers.append(
                {
                    "source_type": SourceType.PMC if pmc_id else SourceType.PUBMED,
                    "source_id": f"PMC:{pmc_id}" if pmc_id else f"PMID:{pmid}",
                    "title": title,
                    "sections": sections,
                }
            )

    # Structured sources ship canonical IDs, so their output bypasses
    # extract + normalize and flows directly as Entity / Edge.
    for source, fetcher in _STRUCTURED_FETCHERS.items():
        if source not in spec.sources:
            continue
        try:
            ents, eds = await fetcher(spec)
        except Exception as exc:  # noqa: BLE001
            run_state.errors.append(f"fetch_{source.value}: {exc}")
            continue
        structured_entities.extend(ents)
        structured_edges.extend(eds)

    state["fetched_papers"] = papers
    state["structured_entities"] = structured_entities
    state["structured_edges"] = structured_edges
    run_state.fetched_documents = len(papers) + len(structured_entities)
    return state


_STRUCTURED_FETCHERS: dict[SourceName, Any] = {
    SourceName.OPENTARGETS: fetch_opentargets_for_spec,
    SourceName.CHEMBL: fetch_chembl_for_spec,
    SourceName.DRUGBANK: fetch_drugbank_for_spec,
}


async def _extract_node(state: _GraphState) -> _GraphState:
    spec: KGSpec = state["spec"]
    run_state: RunState = state["run_state"]
    extract_tool = get_tool("extract_from_text")

    raw_entities: list[tuple[ExtractionCandidate, Provenance]] = []
    raw_edges: list[tuple[ExtractedEdge, Provenance]] = []

    for paper in state.get("fetched_papers", []):
        source_id = paper["source_id"]
        source_type = paper["source_type"]
        sections: dict[str, str] = paper.get("sections", {})

        for section_name, text in sections.items():
            if not text or not text.strip():
                continue

            section_enum = SectionType.from_alias(section_name)
            payload = ExtractFromTextInput(
                text=text,
                section=section_enum,
                entity_vocabulary=spec.entity_types,
                relation_vocabulary=spec.relation_types,
                source_id=source_id,
                cache_hint=f"{spec.recipe.value}:{section_enum.value}",
            )
            out = await extract_tool.func(payload)
            if not isinstance(out, ExtractFromTextOutput):
                continue

            run_state.total_cost_usd += out.cost_usd
            if out.error:
                run_state.errors.append(f"extract:{source_id}:{section_name}: {out.error}")
                continue

            for ent in out.entities:
                prov = Provenance(
                    source_id=source_id,
                    source_type=source_type,
                    extracting_model=out.model_used,
                    model_tier=out.model_tier,
                    section=section_enum,
                    evidence_span=ent.evidence_span,
                    cost_usd=0.0,
                )
                raw_entities.append((ent, prov))

            for edge in out.edges:
                prov = Provenance(
                    source_id=source_id,
                    source_type=source_type,
                    extracting_model=out.model_used,
                    model_tier=out.model_tier,
                    section=section_enum,
                    evidence_span=edge.evidence_span,
                    cost_usd=0.0,
                )
                raw_edges.append((edge, prov))

    state["raw_entities"] = raw_entities
    state["raw_edges"] = raw_edges
    run_state.extracted_entities = len(raw_entities)
    run_state.extracted_edges = len(raw_edges)
    return state


async def _normalize_node(state: _GraphState) -> _GraphState:
    run_state: RunState = state["run_state"]
    normalize_tool = get_tool("normalize_entity")

    # Deduplicate by (name.lower(), entity_type)
    unique: dict[tuple[str, EntityType], ExtractionCandidate] = {}
    for cand, _ in state.get("raw_entities", []):
        unique.setdefault((cand.name.lower(), cand.entity_type), cand)
    for edge, _ in state.get("raw_edges", []):
        unique.setdefault((edge.subject.name.lower(), edge.subject.entity_type), edge.subject)
        unique.setdefault((edge.object.name.lower(), edge.object.entity_type), edge.object)

    normalized_by_key: dict[tuple[str, EntityType], Entity] = {}
    for key, cand in unique.items():
        out = await normalize_tool.func(
            NormalizeEntityInput(name=cand.name, entity_type=cand.entity_type)
        )
        if not isinstance(out, NormalizeEntityOutput) or out.primary_id is None:
            # Synthetic local: ID keeps the entity flowing; callers can prune.
            primary = Identifier(namespace=NamespaceName.LOCAL, value=slugify(cand.name))
            run_state.errors.append(
                f"normalize:{cand.name}:{cand.entity_type.value}: "
                f"{(out.error if isinstance(out, NormalizeEntityOutput) else 'unknown')}"
            )
        else:
            primary = out.primary_id

        normalized_by_key[key] = Entity(
            entity_type=cand.entity_type,
            name=cand.name,
            primary_id=primary,
            aliases=getattr(out, "aliases", []) if isinstance(out, NormalizeEntityOutput) else [],
        )

    # Flatten to name -> Entity (first occurrence wins)
    normalized: dict[str, Entity] = {}
    for (name_lower, _), entity in normalized_by_key.items():
        normalized.setdefault(name_lower, entity)

    # Structured sources already ship canonical Entity instances — merge them
    # in without re-normalizing. Literature-derived names take precedence on
    # collision, since they reflect what the model actually found.
    for entity in state.get("structured_entities", []):
        normalized.setdefault(entity.name.lower(), entity)

    state["normalized_entities"] = normalized
    run_state.normalized_entities = len(normalized)
    return state


async def _flag_conflicts_node(state: _GraphState) -> _GraphState:
    run_state: RunState = state["run_state"]
    normalized = state.get("normalized_entities", {})

    final_edges: list[Edge] = []
    seen: dict[tuple[str, RelationType, str], Edge] = {}

    for cand_edge, prov in state.get("raw_edges", []):
        subj = normalized.get(cand_edge.subject.name.lower())
        obj = normalized.get(cand_edge.object.name.lower())
        if subj is None or obj is None:
            continue

        key = (subj.node_id, cand_edge.relation, obj.node_id)
        edge = Edge(
            relation=cand_edge.relation,
            subject_id=subj.node_id,
            object_id=obj.node_id,
            confidence=cand_edge.confidence,
            provenance=prov,
        )

        if key in seen:
            prior = seen[key]
            if prior.provenance.source_id != edge.provenance.source_id:
                prior.dissenting_sources.append(edge.provenance.source_id)
            continue

        # Reverse-direction conflict check: same subject/object, same relation,
        # but swapped — worth flagging for downstream review.
        reverse_key = (obj.node_id, cand_edge.relation, subj.node_id)
        if reverse_key in seen:
            edge.conflict_flag = True
            edge.dissenting_sources.append(seen[reverse_key].provenance.source_id)
            run_state.flagged_conflicts += 1

        seen[key] = edge
        final_edges.append(edge)

    # Append structured edges verbatim — they already carry provenance and
    # canonical IDs. Reverse-direction conflict check runs against them too.
    for structured_edge in state.get("structured_edges", []):
        key = (
            structured_edge.subject_id,
            structured_edge.relation,
            structured_edge.object_id,
        )
        reverse_key = (
            structured_edge.object_id,
            structured_edge.relation,
            structured_edge.subject_id,
        )
        if reverse_key in seen:
            structured_edge.conflict_flag = True
            structured_edge.dissenting_sources.append(
                seen[reverse_key].provenance.source_id
            )
            run_state.flagged_conflicts += 1
        if key not in seen:
            seen[key] = structured_edge
            final_edges.append(structured_edge)

    state["final_edges"] = final_edges
    return state


async def _write_node(state: _GraphState) -> _GraphState:
    spec: KGSpec = state["spec"]
    run_state: RunState = state["run_state"]
    write_tool = get_tool("write_biocypher")

    entities = list(state.get("normalized_entities", {}).values())
    edges = list(state.get("final_edges", []))

    if not entities and not edges:
        run_state.errors.append("write: nothing to write (empty result)")
        return state

    out = await write_tool.func(
        BiocypherWriteInput(
            entities=entities,
            edges=edges,
            output_dir=f"biocypher-out/{spec.recipe.value}",
        )
    )
    if getattr(out, "error", None):
        run_state.errors.append(f"write: {out.error}")
    output_dir = getattr(out, "output_dir", None)
    if output_dir:
        run_state.output_path = output_dir
    return state


# =============================================================================
# Helpers
# =============================================================================


class _LinearFallbackGraph:
    """Minimal linear dispatcher used when langgraph is not installed."""

    def __init__(self, nodes: list[Any]) -> None:
        self._nodes = nodes

    async def run(self, state: _GraphState) -> _GraphState:
        current = state
        for node in self._nodes:
            current = await node(current)
        return current


def _recipes_prompt_block() -> str:
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


def _build_pubmed_query(spec: KGSpec) -> str:
    """Compose a PubMed query from the spec's seed entities."""
    terms: list[str] = []
    for values in spec.seeds.values():
        for v in values:
            if v:
                terms.append(f'"{v}"')
    if not terms:
        terms = [spec.original_request]
    return " OR ".join(terms)


__all__ = ["LangGraphOrchestrator"]
