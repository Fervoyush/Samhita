"""Benchmark harness tests — no real LLM or network calls."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from samhita.benchmark import (
    BenchmarkConfig,
    BenchmarkReport,
    parse_provider_spec,
    run_benchmark,
)
from samhita.core.schemas import (
    Edge,
    Entity,
    EntityType,
    Identifier,
    KGResult,
    KGSpec,
    NamespaceName,
    Provenance,
    RecipeName,
    RelationType,
    RunState,
    RunStatus,
    SectionType,
    SourceName,
    SourceType,
)


def test_parse_provider_spec_rejects_missing_colon() -> None:
    with pytest.raises(ValueError):
        parse_provider_spec("anthropic-sonnet")


def test_parse_provider_spec_ok() -> None:
    cfg = parse_provider_spec("anthropic:claude-sonnet-4-6")
    assert cfg.provider == "anthropic"
    assert cfg.model == "claude-sonnet-4-6"
    assert cfg.framework == "langgraph"
    assert cfg.label == "anthropic:claude-sonnet-4-6"


def _make_spec() -> KGSpec:
    return KGSpec(
        recipe=RecipeName.DRUG_TARGET_DISEASE,
        seeds={"disease": ["atopic dermatitis"]},
        sources=[SourceName.OPENTARGETS],
        entity_types=[EntityType.DRUG, EntityType.DISEASE],
        relation_types=[RelationType.TREATS],
        max_papers=5,
        original_request="test",
    )


def _make_result(
    spec: KGSpec,
    *,
    cost: float,
    duration: float,
    entities: list[Entity],
    edges: list[Edge],
    input_tokens: int = 1000,
    cached: int = 500,
    output_tokens: int = 100,
) -> KGResult:
    state = RunState(
        spec=spec,
        status=RunStatus.COMPLETED,
        total_cost_usd=cost,
        total_input_tokens=input_tokens,
        total_output_tokens=output_tokens,
        cached_input_tokens=cached,
        fetched_documents=1,
    )
    return KGResult(
        spec=spec,
        entities=entities,
        edges=edges,
        state=state,
        build_duration_seconds=duration,
    )


def _drug(name: str, chembl: str) -> Entity:
    return Entity(
        entity_type=EntityType.DRUG,
        name=name,
        primary_id=Identifier(namespace=NamespaceName.CHEMBL, value=chembl),
    )


def _disease(name: str, efo: str) -> Entity:
    return Entity(
        entity_type=EntityType.DISEASE,
        name=name,
        primary_id=Identifier(namespace=NamespaceName.EFO, value=efo),
    )


def _treats_edge(drug: Entity, disease: Entity, source: str) -> Edge:
    return Edge(
        relation=RelationType.TREATS,
        subject_id=drug.node_id,
        object_id=disease.node_id,
        confidence=0.9,
        provenance=Provenance(
            source_id=source,
            source_type=SourceType.OPENTARGETS,
            section=SectionType.UNKNOWN,
        ),
    )


class _FakeOrchestrator:
    def __init__(self, spec: KGSpec, result: KGResult) -> None:
        self._spec = spec
        self._result = result

    async def plan(self, nl: str) -> KGSpec:  # noqa: ARG002
        return self._spec

    async def execute(self, spec: KGSpec) -> KGResult:  # noqa: ARG002
        return self._result


async def test_run_benchmark_computes_overlaps(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = _make_spec()
    dupilumab = _drug("dupilumab", "CHEMBL1201580")
    topical = _drug("topical corticosteroids", "CHEMBL_TOPICAL")
    ad = _disease("atopic dermatitis", "EFO_0000274")

    # Provider A: 2 entities (dupilumab, ad) + 1 edge
    result_a = _make_result(
        spec,
        cost=0.03,
        duration=22.5,
        entities=[dupilumab, ad],
        edges=[_treats_edge(dupilumab, ad, "OT:A")],
        input_tokens=1200,
        cached=200,
        output_tokens=80,
    )
    # Provider B: 3 entities (dupilumab, topical, ad) + same edge
    result_b = _make_result(
        spec,
        cost=0.002,
        duration=18.1,
        entities=[dupilumab, topical, ad],
        edges=[_treats_edge(dupilumab, ad, "OT:B")],
        input_tokens=1200,
        cached=900,
        output_tokens=80,
    )

    call_count = {"n": 0}

    def _fake_orch_factory(cfg: BenchmarkConfig) -> _FakeOrchestrator:
        call_count["n"] += 1
        return _FakeOrchestrator(spec, result_a if cfg.provider == "anthropic" else result_b)

    import samhita.benchmark as bm

    monkeypatch.setattr(bm, "_build_orchestrator", _fake_orch_factory)

    report = await run_benchmark(
        request="atopic dermatitis drugs",
        configs=[
            BenchmarkConfig(provider="anthropic", model="claude-sonnet-4-6"),
            BenchmarkConfig(provider="moonshot", model="kimi-k2.5"),
        ],
        max_papers=5,
    )

    assert isinstance(report, BenchmarkReport)
    assert len(report.runs) == 2
    assert report.runs[0].label == "anthropic:claude-sonnet-4-6"
    assert report.runs[1].label == "moonshot:kimi-k2.5"
    # Kimi should be the cheaper run
    assert report.runs[1].total_cost_usd < report.runs[0].total_cost_usd
    # Cache-hit rate = cached/input
    assert report.runs[1].cache_hit_rate == pytest.approx(900 / 1200)
    assert report.runs[0].cache_hit_rate == pytest.approx(200 / 1200)

    # One pairwise overlap between A and B
    assert len(report.overlaps) == 1
    ov = report.overlaps[0]
    # 2 entities common (dupilumab, ad) out of 3 total (+topical only in B)
    assert ov.shared_entities == 2
    assert ov.total_entities == 3
    assert ov.entity_jaccard == pytest.approx(2 / 3)
    # 1 edge common out of 1 total
    assert ov.shared_edges == 1
    assert ov.edge_jaccard == pytest.approx(1.0)


async def test_run_benchmark_handles_per_provider_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _make_spec()
    dupilumab = _drug("dupilumab", "CHEMBL1201580")
    ad = _disease("atopic dermatitis", "EFO_0000274")
    good_result = _make_result(
        spec,
        cost=0.01,
        duration=5.0,
        entities=[dupilumab, ad],
        edges=[_treats_edge(dupilumab, ad, "OT")],
    )

    class _ExplodingOrch:
        async def plan(self, nl: str) -> KGSpec:  # noqa: ARG002
            return spec

        async def execute(self, spec: KGSpec) -> KGResult:  # noqa: ARG002
            raise RuntimeError("simulated moonshot outage")

    def _factory(cfg: BenchmarkConfig) -> Any:
        if cfg.provider == "moonshot":
            return _ExplodingOrch()
        return _FakeOrchestrator(spec, good_result)

    import samhita.benchmark as bm

    monkeypatch.setattr(bm, "_build_orchestrator", _factory)

    report = await run_benchmark(
        request="anything",
        configs=[
            BenchmarkConfig(provider="anthropic", model="claude-sonnet-4-6"),
            BenchmarkConfig(provider="moonshot", model="kimi-k2.5"),
        ],
    )
    statuses = {run.label: run.status for run in report.runs}
    assert statuses["anthropic:claude-sonnet-4-6"] == "completed"
    assert statuses["moonshot:kimi-k2.5"] == "failed"
    # No overlap row when one side failed
    assert report.overlaps == []
