"""LLM client abstraction — framework and provider neutral.

Any code that calls an LLM goes through this protocol. Swapping
between Anthropic / OpenAI / Kimi / local is a config flag, not a rewrite.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from samhita.core.schemas import ModelTier


class Message(BaseModel):
    role: str = Field(description="'system', 'user', or 'assistant'")
    content: str


class LLMResponse(BaseModel):
    """Unified LLM response envelope."""

    content: str = Field(description="raw text content (if no schema provided)")
    parsed: BaseModel | None = Field(
        default=None, description="schema-validated payload if a schema was passed"
    )
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    cost_usd: float = 0.0
    model: str = Field(description="resolved model id")
    provider: str = Field(description="'anthropic' | 'openai' | 'moonshot' | 'local'")


@runtime_checkable
class LLMClient(Protocol):
    """Minimal contract every LLM provider implementation must satisfy."""

    name: str
    tier: ModelTier
    provider: str

    async def complete(
        self,
        messages: list[Message],
        schema: type[BaseModel] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        cache_hint: str | None = None,
    ) -> LLMResponse:
        """Complete a conversation, optionally returning a validated payload.

        Args:
            messages: conversation history
            schema: if provided, response is validated + returned in `parsed`
            temperature: sampling temperature
            max_tokens: upper bound on output
            cache_hint: opaque key used by cache-aware providers (Kimi, Claude)
                to route to a cached context segment
        """
        ...
