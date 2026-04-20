"""ID normalization tools — schemas and stubs.

Deterministic namespace resolution with LLM fallback only where
deterministic lookups fail. Implementations land in Week 1 Day 5.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from samhita.core.schemas import EntityType, Identifier


class NormalizeEntityInput(BaseModel):
    name: str
    entity_type: EntityType
    context_hint: str | None = Field(
        default=None,
        description="surrounding text to disambiguate (e.g. species context)",
    )


class NormalizeEntityOutput(BaseModel):
    name: str
    entity_type: EntityType
    primary_id: Identifier | None = None
    aliases: list[Identifier] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    method: str = Field(
        default="deterministic",
        description="'deterministic' | 'llm_fallback' | 'failed'",
    )
    error: str | None = None
