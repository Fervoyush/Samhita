"""LLM-powered extraction tools — schemas and stubs.

Section-aware extraction of biomedical entities and relations from text.
Implementations land in Week 1 Day 6.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from samhita.core.schemas import (
    EntityType,
    ModelTier,
    RelationType,
    SectionType,
)


class ExtractionCandidate(BaseModel):
    """A candidate entity extracted from text before normalization."""

    name: str
    entity_type: EntityType
    evidence_span: str


class ExtractedEdge(BaseModel):
    """A candidate edge extracted from text before ID normalization."""

    subject: ExtractionCandidate
    relation: RelationType
    object: ExtractionCandidate
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_span: str


class ExtractFromTextInput(BaseModel):
    text: str
    section: SectionType
    entity_vocabulary: list[EntityType]
    relation_vocabulary: list[RelationType]
    source_id: str
    cache_hint: str | None = Field(
        default=None,
        description="opaque key for cache-aware LLM providers",
    )


class ExtractFromTextOutput(BaseModel):
    entities: list[ExtractionCandidate]
    edges: list[ExtractedEdge]
    model_used: str
    model_tier: ModelTier
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    cost_usd: float = 0.0
