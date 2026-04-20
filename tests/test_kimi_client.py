"""Moonshot (Kimi K2.5) client smoke tests — no real API calls."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel

from samhita.core.bootstrap import bootstrap_llm_providers
from samhita.core.llm import Message
from samhita.core.llm_clients import get_llm_client, list_providers
from samhita.core.llm_clients.kimi import KimiClient


class _EmitPayload(BaseModel):
    answer: str
    score: float


def _mock_openai_response(
    *,
    content: str = "",
    tool_args: dict[str, Any] | None = None,
    input_tokens: int = 1000,
    cached_tokens: int = 400,
    output_tokens: int = 80,
) -> Any:
    tool_calls: list[Any] = []
    if tool_args is not None:
        import json as _json

        tool_calls = [
            SimpleNamespace(
                function=SimpleNamespace(arguments=_json.dumps(tool_args))
            )
        ]
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=message)
    usage = SimpleNamespace(
        prompt_tokens=input_tokens,
        completion_tokens=output_tokens,
        prompt_tokens_details=SimpleNamespace(cached_tokens=cached_tokens),
    )
    return SimpleNamespace(choices=[choice], usage=usage)


class _FakeAsyncOpenAI:
    def __init__(self, response: Any) -> None:
        class _Chat:
            class _Completions:
                async def create(self, **_: Any) -> Any:
                    return response

            completions = _Completions()

        self.chat = _Chat()


def test_moonshot_provider_registered() -> None:
    bootstrap_llm_providers(force=True)
    assert "moonshot" in list_providers()


def test_factory_builds_kimi_client() -> None:
    bootstrap_llm_providers(force=True)
    client = get_llm_client("moonshot", model="kimi-k2.5")
    assert isinstance(client, KimiClient)
    assert client.model == "kimi-k2.5"
    assert client.provider == "moonshot"


def test_cost_estimate_honors_cache_discount() -> None:
    client = KimiClient(model="kimi-k2.5")
    # 1M input with 500K cached + 100K output
    cost = client._estimate_cost(  # type: ignore[attr-defined]
        input_tokens=1_000_000,
        cached_tokens=500_000,
        output_tokens=100_000,
    )
    # 500K fresh × $0.60 + 500K cached × $0.15 + 100K output × $2.50
    expected = 500_000 * 0.60 / 1e6 + 500_000 * 0.15 / 1e6 + 100_000 * 2.50 / 1e6
    assert cost == expected


async def test_complete_parses_structured_tool_call(monkeypatch: pytest.MonkeyPatch) -> None:
    client = KimiClient(model="kimi-k2.5")

    response = _mock_openai_response(
        tool_args={"answer": "hello", "score": 0.91},
        input_tokens=2000,
        cached_tokens=1600,
        output_tokens=120,
    )

    def _fake_sdk() -> Any:
        return _FakeAsyncOpenAI(response)

    monkeypatch.setattr(client, "_sdk", _fake_sdk)

    out = await client.complete(
        messages=[
            Message(role="system", content="You are a helper."),
            Message(role="user", content="run."),
        ],
        schema=_EmitPayload,
    )
    assert isinstance(out.parsed, _EmitPayload)
    assert out.parsed.answer == "hello"
    assert out.parsed.score == 0.91
    assert out.input_tokens == 2000
    assert out.cached_tokens == 1600
    assert out.output_tokens == 120
    # Non-zero cost with cache discount applied
    assert out.cost_usd > 0
    assert out.cost_usd < 2000 * 0.60 / 1e6 + 120 * 2.50 / 1e6


async def test_complete_without_schema_returns_raw_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = KimiClient(model="kimi-k2.5")
    response = _mock_openai_response(content="raw text")

    monkeypatch.setattr(client, "_sdk", lambda: _FakeAsyncOpenAI(response))

    out = await client.complete(messages=[Message(role="user", content="hi")])
    assert out.parsed is None
    assert out.content == "raw text"
