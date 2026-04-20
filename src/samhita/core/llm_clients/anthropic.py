"""Anthropic (Claude) client implementing :class:`LLMClient`.

Structured output uses the tool-calling mechanism: when a schema is
provided, the model is forced to call a single `emit_structured` tool
whose input schema is the caller's Pydantic model.
"""

from __future__ import annotations

import json
import os
from typing import Any

from pydantic import BaseModel

from samhita.core.llm import LLMClient, LLMResponse, Message
from samhita.core.llm_clients import register_llm_factory
from samhita.core.schemas import ModelTier

# Per 1M tokens (USD). Values are illustrative defaults and overridable
# via constructor — update when Anthropic publishes new prices.
_DEFAULT_PRICING: dict[str, tuple[float, float, float]] = {
    # model_id: (input, cached_input, output)
    "claude-sonnet-4-6": (3.00, 0.30, 15.00),
    "claude-opus-4-7": (15.00, 1.50, 75.00),
    "claude-haiku-4-5-20251001": (1.00, 0.10, 5.00),
}


class AnthropicClient:
    """LLMClient implementation using the `anthropic` SDK."""

    provider = "anthropic"

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        api_key: str | None = None,
        tier: ModelTier = ModelTier.MID,
        pricing: tuple[float, float, float] | None = None,
    ) -> None:
        self.name = model
        self.model = model
        self.tier = tier
        self._api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self._pricing = pricing or _DEFAULT_PRICING.get(model, (3.00, 0.30, 15.00))
        self._client: Any | None = None  # lazy

    def _sdk(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import anthropic  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "anthropic SDK not installed; `pip install anthropic`."
            ) from exc
        self._client = anthropic.AsyncAnthropic(api_key=self._api_key or None)
        return self._client

    async def complete(
        self,
        messages: list[Message],
        schema: type[BaseModel] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        cache_hint: str | None = None,  # noqa: ARG002 — not used by Anthropic-side cache
    ) -> LLMResponse:
        client = self._sdk()

        system_parts = [m.content for m in messages if m.role == "system"]
        chat_messages = [
            {"role": m.role, "content": m.content}
            for m in messages
            if m.role != "system"
        ]
        system_text = "\n\n".join(system_parts) if system_parts else None

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": chat_messages,
        }
        if system_text is not None:
            kwargs["system"] = system_text

        if schema is not None:
            schema_dict = schema.model_json_schema()
            kwargs["tools"] = [
                {
                    "name": "emit_structured",
                    "description": f"Return a validated {schema.__name__} payload.",
                    "input_schema": schema_dict,
                }
            ]
            kwargs["tool_choice"] = {"type": "tool", "name": "emit_structured"}

        response = await client.messages.create(**kwargs)

        parsed: BaseModel | None = None
        content_text = ""
        for block in getattr(response, "content", []) or []:
            block_type = getattr(block, "type", None)
            if block_type == "text":
                content_text += getattr(block, "text", "") or ""
            elif block_type == "tool_use" and schema is not None:
                raw = getattr(block, "input", {}) or {}
                if isinstance(raw, str):
                    raw = json.loads(raw)
                parsed = schema.model_validate(raw)

        usage = getattr(response, "usage", None)
        input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        cached_tokens = int(getattr(usage, "cache_read_input_tokens", 0) or 0)

        cost = self._estimate_cost(input_tokens, cached_tokens, output_tokens)

        return LLMResponse(
            content=content_text,
            parsed=parsed,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            cost_usd=cost,
            model=self.model,
            provider=self.provider,
        )

    def _estimate_cost(
        self, input_tokens: int, cached_tokens: int, output_tokens: int
    ) -> float:
        input_rate, cached_rate, output_rate = self._pricing
        fresh_input = max(0, input_tokens - cached_tokens)
        return (
            fresh_input * input_rate / 1_000_000
            + cached_tokens * cached_rate / 1_000_000
            + output_tokens * output_rate / 1_000_000
        )


def _factory(**kwargs: Any) -> LLMClient:
    return AnthropicClient(**kwargs)


register_llm_factory("anthropic", _factory)
