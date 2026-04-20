"""End-to-end LangGraph pipeline smoke test — all external I/O mocked.

The test patches module-level functions (search_pubmed, fetch_pmc_paper,
normalize_entity) BEFORE calling ``bootstrap_tools()``, so the Tool
objects in the registry close over the patched callables. No network,
no real LLM, no Biocypher required.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel

from samhita.core.llm import LLMClient, LLMResponse, Message
from samhita.core.schemas import (
    EntityType,
    Identifier,
    KGResult,
    KGSpec,
    ModelTier,
    RecipeName,
    RelationType,
    SourceName,
)
from samhita.core.tools.extract import (
    ExtractedEdge,
    ExtractionCandidate,
    _LLMPayload,
)
from samhita.core.tools.fetch import (
    PMCFetchOutput,
    PubMedAbstractOutput,
    PubMedSearchOutput,
)
from samhita.core.tools.normalize import NormalizeEntityOutput


class _ScriptedLLM:
    """Returns canned plan + extract payloads based on requested schema."""

    name = "scripted-llm"
    provider = "fake"
    tier = ModelTier.MID

    def __init__(self, plan: KGSpec, extract_payload: _LLMPayload) -> None:
        self._plan = plan
        self._extract = extract_payload

    async def complete(
        self,
        messages: list[Message],
        schema: type[BaseModel] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        cache_hint: str | None = None,
    ) -> LLMResponse:
        parsed: BaseModel
        if schema is KGSpec:
            parsed = self._plan
        elif schema is _LLMPayload:
            parsed = self._extract
        else:
            parsed = self._extract
        return LLMResponse(
            content="",
            parsed=parsed,
            input_tokens=200,
            output_tokens=80,
            cached_tokens=150,
            cost_usd=0.001,
            model=self.name,
            provider=self.provider,
        )


async def test_end_to_end_mocked_pipeline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import samhita.core.tools.fetch as fetch_mod
    import samhita.core.tools.normalize as norm_mod
    import samhita.core.tools.write as write_mod

    # --- Patch external callables BEFORE bootstrap -----------------------
    async def _fake_search(_):  # noqa: ANN001
        return PubMedSearchOutput(pmids=["111"], total_hits=1)

    async def _fake_abstract(_):  # noqa: ANN001
        return PubMedAbstractOutput(
            pmid="111",
            title="Dupilumab in AD",
            abstract="Dupilumab targets IL-4Rα and treats atopic dermatitis.",
            pmc_id="PMC111",
        )

    async def _fake_pmc(_):  # noqa: ANN001
        return PMCFetchOutput(
            pmc_id="PMC111",
            title="Dupilumab in AD",
            sections={
                "results": "Dupilumab targets IL-4Rα.",
                "discussion": "Dupilumab treats atopic dermatitis.",
            },
        )

    ns_map = {
        EntityType.DRUG: "ChEMBL",
        EntityType.GENE: "HGNC",
        EntityType.DISEASE: "MONDO",
    }

    async def _fake_normalize(payload):  # noqa: ANN001
        return NormalizeEntityOutput(
            name=payload.name,
            entity_type=payload.entity_type,
            primary_id=Identifier(
                namespace=ns_map.get(payload.entity_type, "local"),
                value=payload.name.upper(),
            ),
        )

    monkeypatch.setattr(fetch_mod, "search_pubmed", _fake_search)
    monkeypatch.setattr(fetch_mod, "fetch_pubmed_abstract", _fake_abstract)
    monkeypatch.setattr(fetch_mod, "fetch_pmc_paper", _fake_pmc)
    monkeypatch.setattr(norm_mod, "normalize_entity", _fake_normalize)

    # Force write tool to target tmp_path regardless of what the node requests
    original_write = write_mod.write_biocypher

    async def _tmp_write(payload):  # noqa: ANN001
        payload.output_dir = str(tmp_path)
        return await original_write(payload)

    monkeypatch.setattr(write_mod, "write_biocypher", _tmp_write)

    # --- Now clear + re-bootstrap so Tool.func picks up the patches ------
    from samhita.core.bootstrap import bootstrap_tools
    from samhita.core.tools import clear_registry

    clear_registry()
    bootstrap_tools()

    # --- Canned plan + extract payloads -----------------------------------
    planned_spec = KGSpec(
        recipe=RecipeName.DRUG_TARGET_DISEASE,
        seeds={"disease": ["atopic dermatitis"]},
        sources=[SourceName.PUBMED_CENTRAL],
        entity_types=[EntityType.DRUG, EntityType.GENE, EntityType.DISEASE],
        relation_types=[RelationType.TARGETS, RelationType.TREATS],
        max_papers=1,
        original_request="build a KG of drugs for atopic dermatitis",
    )

    extract_payload = _LLMPayload(
        entities=[
            ExtractionCandidate(
                name="dupilumab", entity_type=EntityType.DRUG, evidence_span="dupilumab"
            ),
            ExtractionCandidate(
                name="IL4R", entity_type=EntityType.GENE, evidence_span="IL-4Rα"
            ),
            ExtractionCandidate(
                name="atopic dermatitis",
                entity_type=EntityType.DISEASE,
                evidence_span="atopic dermatitis",
            ),
        ],
        edges=[
            ExtractedEdge(
                subject=ExtractionCandidate(
                    name="dupilumab", entity_type=EntityType.DRUG, evidence_span="dupilumab"
                ),
                relation=RelationType.TARGETS,
                object=ExtractionCandidate(
                    name="IL4R", entity_type=EntityType.GENE, evidence_span="IL-4Rα"
                ),
                confidence=0.95,
                evidence_span="dupilumab targets IL-4Rα",
            ),
            ExtractedEdge(
                subject=ExtractionCandidate(
                    name="dupilumab", entity_type=EntityType.DRUG, evidence_span="dupilumab"
                ),
                relation=RelationType.TREATS,
                object=ExtractionCandidate(
                    name="atopic dermatitis",
                    entity_type=EntityType.DISEASE,
                    evidence_span="atopic dermatitis",
                ),
                confidence=0.99,
                evidence_span="dupilumab treats atopic dermatitis",
            ),
        ],
    )

    llm: LLMClient = _ScriptedLLM(plan=planned_spec, extract_payload=extract_payload)

    from samhita.orchestrators.langgraph_driver import LangGraphOrchestrator

    orch = LangGraphOrchestrator(llm=llm)
    result: KGResult = await orch.build("build a KG of drugs for atopic dermatitis")

    from samhita.core.schemas import RunStatus

    assert result.state.status == RunStatus.COMPLETED
    assert result.state.fetched_documents >= 1
    assert result.state.extracted_entities >= 3
    assert result.state.extracted_edges >= 2
    assert len(result.entities) >= 3
    assert len(result.edges) >= 2
    assert result.state.output_path is not None

    # The fallback CSV writer should have produced files in tmp_path
    assert (tmp_path / "nodes.csv").exists() or (tmp_path / "edges.csv").exists()
