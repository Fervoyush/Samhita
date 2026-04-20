"""Extract-tool smoke tests using a fake LLM."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from samhita.core.llm import LLMClient, LLMResponse, Message
from samhita.core.schemas import (
    EntityType,
    ModelTier,
    RelationType,
    SectionType,
)
from samhita.core.tools.extract import (
    ExtractedEdge,
    ExtractFromTextInput,
    ExtractFromTextOutput,
    ExtractionCandidate,
    _LLMPayload,
    make_extract_from_text_tool,
)


class _FakeLLM:
    name = "fake-llm"
    provider = "fake"
    tier = ModelTier.MID

    def __init__(self, payload: BaseModel | Exception) -> None:
        self._payload = payload

    async def complete(
        self,
        messages: list[Message],
        schema: type[BaseModel] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        cache_hint: str | None = None,
    ) -> LLMResponse:
        if isinstance(self._payload, Exception):
            raise self._payload
        return LLMResponse(
            content="",
            parsed=self._payload,
            input_tokens=120,
            output_tokens=40,
            cached_tokens=80,
            cost_usd=0.0005,
            model=self.name,
            provider=self.provider,
        )


def _input() -> ExtractFromTextInput:
    return ExtractFromTextInput(
        text="Imatinib inhibits BCR-ABL in chronic myeloid leukemia.",
        section=SectionType.RESULTS,
        entity_vocabulary=[EntityType.DRUG, EntityType.GENE, EntityType.DISEASE],
        relation_vocabulary=[RelationType.INHIBITS, RelationType.TREATS],
        source_id="PMID:12345",
    )


async def test_extract_returns_typed_output() -> None:
    payload = _LLMPayload(
        entities=[
            ExtractionCandidate(
                name="imatinib",
                entity_type=EntityType.DRUG,
                evidence_span="Imatinib",
            ),
            ExtractionCandidate(
                name="BCR-ABL",
                entity_type=EntityType.GENE,
                evidence_span="BCR-ABL",
            ),
        ],
        edges=[
            ExtractedEdge(
                subject=ExtractionCandidate(
                    name="imatinib",
                    entity_type=EntityType.DRUG,
                    evidence_span="Imatinib",
                ),
                relation=RelationType.INHIBITS,
                object=ExtractionCandidate(
                    name="BCR-ABL",
                    entity_type=EntityType.GENE,
                    evidence_span="BCR-ABL",
                ),
                confidence=0.92,
                evidence_span="Imatinib inhibits BCR-ABL",
            )
        ],
    )

    fake: LLMClient = _FakeLLM(payload)
    tool = make_extract_from_text_tool(fake)
    out = await tool.func(_input())
    assert isinstance(out, ExtractFromTextOutput)
    assert len(out.entities) == 2
    assert len(out.edges) == 1
    assert out.edges[0].relation == RelationType.INHIBITS
    assert out.cost_usd == 0.0005
    assert out.cached_tokens == 80


async def test_extract_filters_out_of_vocab_relations() -> None:
    payload = _LLMPayload(
        entities=[],
        edges=[
            ExtractedEdge(
                subject=ExtractionCandidate(
                    name="imatinib", entity_type=EntityType.DRUG, evidence_span=""
                ),
                # TARGETS not in input's relation_vocabulary
                relation=RelationType.TARGETS,
                object=ExtractionCandidate(
                    name="BCR-ABL", entity_type=EntityType.GENE, evidence_span=""
                ),
                confidence=0.8,
                evidence_span="",
            )
        ],
    )
    fake: LLMClient = _FakeLLM(payload)
    tool = make_extract_from_text_tool(fake)
    out = await tool.func(_input())
    assert out.edges == []


async def test_extract_empty_text() -> None:
    fake: LLMClient = _FakeLLM(_LLMPayload(entities=[], edges=[]))
    tool = make_extract_from_text_tool(fake)
    payload = _input()
    payload.text = "   "
    out = await tool.func(payload)
    assert out.error == "empty text"


async def test_extract_llm_exception_is_surfaced() -> None:
    fake: LLMClient = _FakeLLM(RuntimeError("quota exceeded"))
    tool = make_extract_from_text_tool(fake)
    out = await tool.func(_input())
    assert out.error == "quota exceeded"
