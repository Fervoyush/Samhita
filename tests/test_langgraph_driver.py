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


async def test_execute_returns_kgresult(fake_kgspec: KGSpec) -> None:
    fake: LLMClient = _FakeLLM(fake_kgspec)
    from samhita.orchestrators.langgraph_driver import LangGraphOrchestrator

    orch = LangGraphOrchestrator(llm=fake)
    result = await orch.execute(fake_kgspec)
    assert result.spec == fake_kgspec
    assert result.state.status == "completed"
    # Nodes are currently stubs — they record their stub status via errors
    assert any("stub" in e for e in result.state.errors)


async def test_build_is_plan_plus_execute(fake_kgspec: KGSpec) -> None:
    fake = _FakeLLM(fake_kgspec)
    from samhita.orchestrators.langgraph_driver import LangGraphOrchestrator

    orch = LangGraphOrchestrator(llm=fake)
    result = await orch.build("build a KG of drugs for atopic dermatitis")
    assert result.state.status == "completed"
    assert len(fake.calls) == 1  # single planner call


async def test_planner_wrong_shape_raises(fake_kgspec: KGSpec) -> None:  # noqa: ARG001
    class _Junk(BaseModel):
        x: int = 0

    fake: LLMClient = _FakeLLM(_Junk())
    from samhita.orchestrators.langgraph_driver import LangGraphOrchestrator

    orch = LangGraphOrchestrator(llm=fake)
    with pytest.raises(RuntimeError):
        await orch.plan("anything")
