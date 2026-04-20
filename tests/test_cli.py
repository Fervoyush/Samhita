"""CLI smoke tests via typer.testing."""

from __future__ import annotations

import os
from typing import Any

import pytest
from pydantic import BaseModel
from typer.testing import CliRunner

from samhita.core.llm import LLMClient, LLMResponse, Message
from samhita.core.schemas import (
    EntityType,
    KGResult,
    KGSpec,
    ModelTier,
    RecipeName,
    RelationType,
    RunState,
    SourceName,
)


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def canned_spec() -> KGSpec:
    return KGSpec(
        recipe=RecipeName.DRUG_TARGET_DISEASE,
        seeds={"disease": ["atopic dermatitis"]},
        sources=[SourceName.OPENTARGETS],
        entity_types=[EntityType.DRUG, EntityType.GENE, EntityType.DISEASE],
        relation_types=[RelationType.TARGETS, RelationType.TREATS],
        max_papers=10,
        original_request="build a KG of drugs for atopic dermatitis",
    )


class _FakeOrchestrator:
    name = "fake"

    def __init__(self, spec: KGSpec, result: KGResult) -> None:
        self._spec = spec
        self._result = result

    async def plan(self, nl_request: str) -> KGSpec:  # noqa: ARG002
        return self._spec

    async def execute(self, spec: KGSpec) -> KGResult:  # noqa: ARG002
        return self._result

    async def build(self, nl_request: str) -> KGResult:
        spec = await self.plan(nl_request)
        return await self.execute(spec)


def test_version_command(runner: CliRunner) -> None:
    from samhita.cli import app

    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "Samhita" in result.output


def test_list_orchestrators(runner: CliRunner) -> None:
    from samhita.cli import app

    result = runner.invoke(app, ["list-orchestrators"])
    assert result.exit_code == 0
    assert "langgraph" in result.output


def test_list_recipes(runner: CliRunner) -> None:
    from samhita.cli import app

    result = runner.invoke(app, ["list-recipes"])
    assert result.exit_code == 0
    assert "drug_target_disease" in result.output


def test_build_without_api_key_fails_fast(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    from samhita.cli import app

    result = runner.invoke(app, ["build", "anything"])
    assert result.exit_code == 2
    assert "ANTHROPIC_API_KEY" in result.output


def test_plan_command_prints_spec(
    runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    canned_spec: KGSpec,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

    # Patch the factory so the CLI gets our fake orchestrator
    import samhita.cli as cli_mod

    dummy_result = KGResult(
        spec=canned_spec,
        entities=[],
        edges=[],
        state=RunState(spec=canned_spec, status="completed"),
    )

    def _fake_build_orch(framework: str, provider: str, model: str):  # noqa: ARG001, ANN202
        return _FakeOrchestrator(canned_spec, dummy_result)

    monkeypatch.setattr(cli_mod, "_build_orchestrator", _fake_build_orch)

    from samhita.cli import app

    result = runner.invoke(app, ["plan", "atopic dermatitis drugs"])
    assert result.exit_code == 0
    assert "drug_target_disease" in result.output
    assert "atopic dermatitis" in result.output


def test_build_dry_run_stops_before_execute(
    runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    canned_spec: KGSpec,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    import samhita.cli as cli_mod

    dummy_result = KGResult(
        spec=canned_spec,
        entities=[],
        edges=[],
        state=RunState(spec=canned_spec, status="completed"),
    )

    class _TrackingOrch(_FakeOrchestrator):
        def __init__(self, spec: KGSpec, result: KGResult) -> None:
            super().__init__(spec, result)
            self.execute_calls = 0

        async def execute(self, spec: KGSpec) -> KGResult:  # noqa: ARG002
            self.execute_calls += 1
            return self._result

    orch = _TrackingOrch(canned_spec, dummy_result)
    monkeypatch.setattr(cli_mod, "_build_orchestrator", lambda *a, **k: orch)

    from samhita.cli import app

    result = runner.invoke(app, ["build", "test request", "--dry-run"])
    assert result.exit_code == 0
    assert orch.execute_calls == 0
    assert "dry-run" in result.output.lower()


def test_build_runs_full_pipeline(
    runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    canned_spec: KGSpec,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    import samhita.cli as cli_mod

    state = RunState(
        spec=canned_spec,
        status="completed",
        fetched_documents=3,
        extracted_entities=5,
        extracted_edges=4,
        normalized_entities=5,
        total_cost_usd=0.0123,
    )
    result_obj = KGResult(spec=canned_spec, entities=[], edges=[], state=state)
    monkeypatch.setattr(
        cli_mod, "_build_orchestrator", lambda *a, **k: _FakeOrchestrator(canned_spec, result_obj)
    )

    from samhita.cli import app

    result = runner.invoke(app, ["build", "test request"])
    assert result.exit_code == 0
    assert "Run summary" in result.output
    assert "0.0123" in result.output or "0.01" in result.output
