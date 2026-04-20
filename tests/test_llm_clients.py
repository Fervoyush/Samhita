"""LLM client registry smoke tests."""

from __future__ import annotations

from samhita.core.bootstrap import bootstrap_llm_providers
from samhita.core.llm_clients import get_llm_client, list_providers


def test_anthropic_provider_registered() -> None:
    bootstrap_llm_providers()
    providers = list_providers()
    assert "anthropic" in providers


def test_anthropic_cost_estimate() -> None:
    bootstrap_llm_providers()
    client = get_llm_client("anthropic", model="claude-sonnet-4-6")
    # Private method access is acceptable in unit tests of the same module.
    cost = client._estimate_cost(  # type: ignore[attr-defined]
        input_tokens=1_000_000,
        cached_tokens=500_000,
        output_tokens=100_000,
    )
    # 500K fresh input @ $3/M + 500K cached @ $0.30/M + 100K output @ $15/M
    expected = 500_000 * 3.0 / 1e6 + 500_000 * 0.3 / 1e6 + 100_000 * 15.0 / 1e6
    assert cost == expected
