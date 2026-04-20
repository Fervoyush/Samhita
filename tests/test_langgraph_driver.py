"""LangGraph driver smoke tests — no network, no Anthropic calls."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from samhita.core.llm import LLMClient, LLMResponse, Message
from samhita.core.schemas import (
    EntityType,
    KGSpec,
    ModelTier,
    RecipeName,
    RelationType,
    SourceName,
)


class _FakeLLM:
    """Test double that returns a canned KGSpec when given one."""

    name = "fake-llm"
    provider = "fake"
    tier = ModelTier.MID

    def __init__(self, payload: BaseModel) -> None:
        self._payload = payload
        self.calls: list[list[Message]] = []

    async def complete(
        self,
        messages: list[Message],
        schema: type[BaseModel] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        cache_hint: str | None = None,
    ) -> LLMResponse:
        self.calls.append(messages)
        return LLMResponse(
            content="",
            parsed=self._payload,
            input_tokens=100,
            output_tokens=50,
            model=self.name,
            provider=self.provider,
        )


@pytest.fixture
def fake_kgspec() -> KGSpec:
    return KGSpec(
        recipe=RecipeName.DRUG_TARGET_DISEASE,
        seeds={"disease": ["atopic dermatitis"]},
        sources=[SourceName.OPENTARGETS, SourceName.CHEMBL],
        entity_types=[EntityType.DRUG, EntityType.GENE, EntityType.DISEASE],
        relation_types=[RelationType.TARGETS, RelationType.TREATS],
        max_papers=25,
        original_request="build a KG of drugs for atopic dermatitis",
    )


async def test_plan_returns_kgspec_and_passes_nl_request(fake_kgspec: KGSpec) -> None:
    # Protocol check — fake satisfies LLMClient structurally
    fake: LLMClient = _FakeLLM(fake_kgspec)

    from samhita.orchestrators.langgraph_driver import LangGraphOrchestrator

    orch = LangGraphOrchestrator(llm=fake)
    spec = await orch.plan("build a KG of drugs for atopic dermatitis")
    assert isinstance(spec, KGSpec)
    assert spec.recipe == RecipeName.DRUG_TARGET_DISEASE
    assert "atopic dermatitis" in spec.seeds["disease"]
    assert spec.original_request == "build a KG of drugs for atopic dermatitis"


def _no_sources_spec() -> KGSpec:
    """Spec with no sources — avoids real network calls in unit tests.

    Full-pipeline coverage lives in test_langgraph_end_to_end.py, which
    mocks fetch tools; the tests here only verify plumbing + return shape.
    """
    return KGSpec(
        recipe=RecipeName.DRUG_TARGET_DISEASE,
        seeds={"disease": ["atopic dermatitis"]},
        sources=[],
        entity_types=[EntityType.DRUG, EntityType.GENE, EntityType.DISEASE],
        relation_types=[RelationType.TARGETS, RelationType.TREATS],
        max_papers=1,
        original_request="build a KG of drugs for atopic dermatitis",
    )


async def test_execute_returns_kgresult() -> None:
    from samhita.core.schemas import RunStatus

    spec = _no_sources_spec()
    fake: LLMClient = _FakeLLM(spec)
    from samhita.orchestrators.langgraph_driver import LangGraphOrchestrator

    orch = LangGraphOrchestrator(llm=fake)
    result = await orch.execute(spec)
    assert result.spec == spec
    assert result.state.status == RunStatus.COMPLETED
    assert result.state.fetched_documents == 0
    assert result.entities == []
    assert result.edges == []


async def test_build_is_plan_plus_execute() -> None:
    from samhita.core.schemas import RunStatus

    spec = _no_sources_spec()
    fake = _FakeLLM(spec)
    from samhita.orchestrators.langgraph_driver import LangGraphOrchestrator

    orch = LangGraphOrchestrator(llm=fake)
    result = await orch.build("build a KG of drugs for atopic dermatitis")
    assert result.state.status == RunStatus.COMPLETED
    assert len(fake.calls) == 1  # single planner call


async def test_planner_wrong_shape_raises(fake_kgspec: KGSpec) -> None:  # noqa: ARG001
    class _Junk(BaseModel):
        x: int = 0

    fake: LLMClient = _FakeLLM(_Junk())
    from samhita.orchestrators.langgraph_driver import LangGraphOrchestrator

    orch = LangGraphOrchestrator(llm=fake)
    with pytest.raises(RuntimeError):
        await orch.plan("anything")
