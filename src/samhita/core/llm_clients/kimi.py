"""Moonshot Kimi client implementing :class:`LLMClient`.

Kimi K2.5 is OpenAI-SDK-compatible at ``https://api.moonshot.ai/v1``.
Structured output goes through OpenAI-style function calling (``tools``
+ ``tool_choice``). Prompt caching is automatic on Moonshot's side
(75% discount on cached tokens) — no ``cache_control`` marker needed,
just consistent prompt prefixes across calls.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any

from pydantic import BaseModel

from samhita.core.llm import LLMClient, LLMResponse, Message
from samhita.core.llm_clients import register_llm_factory
from samhita.core.schemas import ModelTier

# Per 1M tokens (USD). Cached_input reflects Moonshot's 75% auto discount.
_DEFAULT_PRICING: dict[str, tuple[float, float, float]] = {
    # model_id: (input, cached_input, output)
    "kimi-k2.5": (0.60, 0.15, 2.50),
    "kimi-k2-thinking": (0.60, 0.15, 2.50),
    "kimi-k2-thinking-turbo": (0.60, 0.15, 2.50),
}

_DEFAULT_MODEL = "kimi-k2.5"
_BASE_URL = "https://api.moonshot.ai/v1"


@lru_cache(maxsize=64)
def _tool_from_schema(schema: type[BaseModel]) -> dict[str, Any]:
    """Convert a Pydantic schema to an OpenAI-style tool definition."""
    return {
        "type": "function",
        "function": {
            "name": "emit_structured",
            "description": f"Return a validated {schema.__name__} payload.",
            "parameters": schema.model_json_schema(),
        },
    }


class KimiClient:
    """LLMClient implementation using the OpenAI SDK against Moonshot."""

    provider = "moonshot"

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        api_key: str | None = None,
        base_url: str = _BASE_URL,
        tier: ModelTier = ModelTier.CHEAP,
        pricing: tuple[float, float, float] | None = None,
    ) -> None:
        self.name = model
        self.model = model
        self.tier = tier
        self._api_key = api_key or os.getenv("MOONSHOT_API_KEY", "")
        self._base_url = base_url
        self._pricing = pricing or _DEFAULT_PRICING.get(model, (0.60, 0.15, 2.50))
        self._client: Any | None = None  # lazy

    def _sdk(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import openai  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "openai SDK not installed; `pip install openai`."
            ) from exc
        self._client = openai.AsyncOpenAI(
            api_key=self._api_key or None,
            base_url=self._base_url,
        )
        return self._client

    async def complete(
        self,
        messages: list[Message],
        schema: type[BaseModel] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        cache_hint: str | None = None,  # noqa: ARG002 — Moonshot caches by prefix
    ) -> LLMResponse:
        client = self._sdk()

        openai_messages = [
            {"role": m.role, "content": m.content} for m in messages
        ]
        # The kimi-k2.* models reject any temperature other than 1; honour the
        # constraint transparently rather than force every caller to special-case
        # per-provider.
        effective_temperature = 1.0 if self.model.startswith("kimi-k2") else temperature
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": openai_messages,
            "temperature": effective_temperature,
            "max_tokens": max_tokens,
        }
        if schema is not None:
            kwargs["tools"] = [_tool_from_schema(schema)]
            # kimi-k2 family is thinking-always and rejects forced tool_choice;
            # let the model decide and fall back to parsing JSON text below.
            kwargs["tool_choice"] = (
                "auto"
                if self.model.startswith("kimi-k2")
                else {"type": "function", "function": {"name": "emit_structured"}}
            )

        response = await client.chat.completions.create(**kwargs)

        parsed: BaseModel | None = None
        content_text = ""
        choice = response.choices[0] if response.choices else None
        if choice is not None:
            msg = choice.message
            content_text = getattr(msg, "content", "") or ""
            if schema is not None:
                parsed = _parse_structured(msg, schema, content_text)

        usage = getattr(response, "usage", None)
        input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        cached_tokens = _read_cached_tokens(usage)
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


def _parse_structured(
    msg: Any, schema: type[BaseModel], content_text: str
) -> BaseModel | None:
    """Extract a validated schema instance from a chat-completion message.

    Order of attempts:
      1. Native OpenAI-style ``tool_calls`` (the happy path for non-thinking
         models that obey forced tool_choice).
      2. Raw JSON embedded in the text content — the fallback for thinking
         models which rejected the forced tool_choice and answered with a
         JSON blob instead.
    """
    tool_calls = getattr(msg, "tool_calls", None) or []
    if tool_calls:
        raw_args = tool_calls[0].function.arguments
        if isinstance(raw_args, str):
            raw_args = json.loads(raw_args)
        return schema.model_validate(raw_args)

    if not content_text:
        return None

    payload = _extract_json_object(content_text)
    if payload is None:
        return None
    try:
        return schema.model_validate(payload)
    except Exception:  # noqa: BLE001
        return None


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Pull a top-level JSON object out of free-form text.

    Handles the three shapes thinking models typically emit:
      - pure JSON
      - JSON inside a ```json fenced block
      - JSON after some leading prose
    """
    stripped = text.strip()
    # Strip markdown fences if present.
    if stripped.startswith("```"):
        fence_end = stripped.find("\n")
        if fence_end != -1:
            stripped = stripped[fence_end + 1 :]
        if stripped.endswith("```"):
            stripped = stripped[: -len("```")]
        stripped = stripped.strip()

    try:
        value = json.loads(stripped)
        if isinstance(value, dict):
            return value
    except Exception:  # noqa: BLE001
        pass

    # Last resort: find the first balanced { ... } in the text.
    start = stripped.find("{")
    if start == -1:
        return None
    depth = 0
    for idx in range(start, len(stripped)):
        ch = stripped[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = stripped[start : idx + 1]
                try:
                    value = json.loads(candidate)
                    if isinstance(value, dict):
                        return value
                except Exception:  # noqa: BLE001
                    return None
    return None


def _read_cached_tokens(usage: Any) -> int:
    """Moonshot reports cached tokens inside ``prompt_tokens_details`` when
    present. Fall back to 0 if the provider hasn't populated the field.
    """
    if usage is None:
        return 0
    details = getattr(usage, "prompt_tokens_details", None)
    if details is None:
        return 0
    return int(getattr(details, "cached_tokens", 0) or 0)


def _factory(**kwargs: Any) -> LLMClient:
    return KimiClient(**kwargs)


register_llm_factory("moonshot", _factory)
