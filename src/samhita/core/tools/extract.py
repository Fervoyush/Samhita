"""Section-aware biomedical extraction via an injected LLMClient.

Extraction is the one tool family that *requires* an LLM, so it can't
be registered by the framework-agnostic `bootstrap_tools()`. Instead,
orchestrator drivers build an extraction tool bound to their LLM via
`make_extract_from_text_tool(llm)` and register it at startup.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from samhita.core.llm import LLMClient, Message
from samhita.core.prompts import EXTRACTION_SYSTEM_PROMPT_TEMPLATE
from samhita.core.schemas import (
    EntityType,
    ModelTier,
    RelationType,
    SectionType,
)
from samhita.core.tools import Tool, register_tool


# ---------------------------------------------------------------------------
# I/O schemas
# ---------------------------------------------------------------------------


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
    max_chunk_chars: int = Field(
        default=20_000,
        description="upper bound on text length sent to the LLM in one call",
    )


class ExtractFromTextOutput(BaseModel):
    entities: list[ExtractionCandidate] = Field(default_factory=list)
    edges: list[ExtractedEdge] = Field(default_factory=list)
    model_used: str = ""
    model_tier: ModelTier = ModelTier.MID
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    cost_usd: float = 0.0
    error: str | None = None


class _LLMPayload(BaseModel):
    """What the LLM returns — a subset of ExtractFromTextOutput."""

    entities: list[ExtractionCandidate] = Field(default_factory=list)
    edges: list[ExtractedEdge] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Implementation factory
# ---------------------------------------------------------------------------


def make_extract_from_text_tool(llm: LLMClient) -> Tool:
    """Build an extraction tool bound to the given LLM."""

    async def _impl(payload: ExtractFromTextInput) -> ExtractFromTextOutput:
        text = payload.text[: payload.max_chunk_chars]
        if not text.strip():
            return ExtractFromTextOutput(
                model_used=getattr(llm, "name", ""),
                model_tier=getattr(llm, "tier", ModelTier.MID),
                error="empty text",
            )

        entity_vocab = ", ".join(e.value for e in payload.entity_vocabulary)
        relation_vocab = ", ".join(r.value for r in payload.relation_vocabulary)

        system = EXTRACTION_SYSTEM_PROMPT_TEMPLATE.format(section=payload.section.value)
        user = (
            f"Source ID: {payload.source_id}\n"
            f"Entity vocabulary (use only these types): {entity_vocab}\n"
            f"Relation vocabulary (use only these): {relation_vocab}\n\n"
            f"Text:\n{text}"
        )

        try:
            response = await llm.complete(
                messages=[
                    Message(role="system", content=system),
                    Message(role="user", content=user),
                ],
                schema=_LLMPayload,
                temperature=0.0,
                max_tokens=4096,
                cache_hint=payload.cache_hint,
            )
        except Exception as exc:  # noqa: BLE001
            return ExtractFromTextOutput(
                model_used=getattr(llm, "name", ""),
                model_tier=getattr(llm, "tier", ModelTier.MID),
                error=str(exc),
            )

        parsed = response.parsed
        if not isinstance(parsed, _LLMPayload):
            return ExtractFromTextOutput(
                model_used=response.model,
                model_tier=getattr(llm, "tier", ModelTier.MID),
                error="LLM did not return a structured extraction payload",
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                cached_tokens=response.cached_tokens,
                cost_usd=response.cost_usd,
            )

        # Filter: drop edges/entities whose types aren't in the declared vocab
        allowed_etypes = set(payload.entity_vocabulary)
        allowed_rtypes = set(payload.relation_vocabulary)

        filtered_entities = [
            c for c in parsed.entities if c.entity_type in allowed_etypes
        ]
        filtered_edges = [
            e
            for e in parsed.edges
            if e.relation in allowed_rtypes
            and e.subject.entity_type in allowed_etypes
            and e.object.entity_type in allowed_etypes
        ]

        return ExtractFromTextOutput(
            entities=filtered_entities,
            edges=filtered_edges,
            model_used=response.model,
            model_tier=getattr(llm, "tier", ModelTier.MID),
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cached_tokens=response.cached_tokens,
            cost_usd=response.cost_usd,
        )

    return Tool(
        name="extract_from_text",
        description=(
            "Extract biomedical entities and typed relations from a section "
            "of text using the injected LLM. Returns strictly within the "
            "provided entity + relation vocabularies."
        ),
        input_schema=ExtractFromTextInput,
        output_schema=ExtractFromTextOutput,
        func=_impl,
        tags=["extract", "llm"],
    )


def register_extract_tools(llm: LLMClient) -> None:
    """Register the LLM-bound extraction tool (idempotent)."""
    register_tool(make_extract_from_text_tool(llm))
