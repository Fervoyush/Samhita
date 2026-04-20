"""Structured-fetcher tests — mocked low-level tools, no network."""

from __future__ import annotations

from typing import Any

import pytest

from samhita.core.bootstrap import bootstrap_tools
from samhita.core.fetchers import (
    fetch_chembl_for_spec,
    fetch_drugbank_for_spec,
    fetch_opentargets_for_spec,
)
from samhita.core.schemas import (
    EntityType,
    KGSpec,
    RecipeName,
    RelationType,
    SourceName,
)
from samhita.core.tools import clear_registry, get_tool
from samhita.core.tools.fetch import (
    ChEMBLQueryOutput,
    DrugBankLookupOutput,
    OpenTargetsQueryOutput,
)


@pytest.fixture(autouse=True)
def _fresh_registry() -> None:
    clear_registry()
    bootstrap_tools()


def _spec(**overrides: Any) -> KGSpec:
    base = {
        "recipe": RecipeName.DRUG_TARGET_DISEASE,
        "seeds": {"disease": ["atopic dermatitis"]},
        "sources": [SourceName.OPENTARGETS],
        "entity_types": [EntityType.DRUG, EntityType.GENE, EntityType.DISEASE],
        "relation_types": [RelationType.TARGETS, RelationType.TREATS],
        "original_request": "test",
    }
    base.update(overrides)
    return KGSpec(**base)


# -- OpenTargets -------------------------------------------------------------


async def test_opentargets_disease_seed_produces_entities_and_edges() -> None:
    """Search → disease-targets query should yield a disease + target + edge."""
    queue = [
        # Search result (disease lookup)
        OpenTargetsQueryOutput(
            data={"search": {"hits": [{"id": "EFO_0000274", "name": "atopic eczema"}]}}
        ),
        # Disease-targets payload
        OpenTargetsQueryOutput(
            data={
                "disease": {
                    "associatedTargets": {
                        "rows": [
                            {
                                "score": 0.82,
                                "target": {
                                    "id": "ENSG00000125538",
                                    "approvedSymbol": "IL1B",
                                    "approvedName": "interleukin 1 beta",
                                },
                            }
                        ]
                    },
                    "knownDrugs": {
                        "rows": [
                            {"drug": {"id": "CHEMBL1201580", "name": "DUPILUMAB"}, "phase": 4}
                        ]
                    },
                }
            }
        ),
    ]

    tool = get_tool("query_opentargets")

    async def _stub(_):  # noqa: ANN001
        return queue.pop(0) if queue else OpenTargetsQueryOutput(data={})

    tool.func = _stub  # type: ignore[assignment]

    entities, edges = await fetch_opentargets_for_spec(
        _spec(seeds={"disease": ["atopic dermatitis"]})
    )
    ids = {e.node_id for e in entities}
    assert "EFO:EFO_0000274" in ids
    assert "Ensembl:ENSG00000125538" in ids
    assert "ChEMBL:CHEMBL1201580" in ids

    # Should include an associated_with (gene -> disease) and a treats (drug -> disease)
    relations = {(e.subject_id, e.relation, e.object_id) for e in edges}
    assert (
        "Ensembl:ENSG00000125538",
        RelationType.ASSOCIATED_WITH,
        "EFO:EFO_0000274",
    ) in relations
    assert (
        "ChEMBL:CHEMBL1201580",
        RelationType.TREATS,
        "EFO:EFO_0000274",
    ) in relations


async def test_opentargets_no_hit_yields_nothing() -> None:
    async def _stub(_):  # noqa: ANN001
        return OpenTargetsQueryOutput(data={"search": {"hits": []}})

    get_tool("query_opentargets").func = _stub  # type: ignore[assignment]
    entities, edges = await fetch_opentargets_for_spec(_spec())
    assert entities == [] and edges == []


# -- ChEMBL ------------------------------------------------------------------


async def test_chembl_drug_seed_produces_mechanism_edges() -> None:
    queue = [
        # molecule/search
        ChEMBLQueryOutput(data={"molecules": [{"molecule_chembl_id": "CHEMBL941"}]}),
        # mechanism
        ChEMBLQueryOutput(
            data={
                "mechanisms": [
                    {
                        "target_chembl_id": "CHEMBL1862",
                        "mechanism_of_action": "ABL1 tyrosine kinase inhibitor",
                        "action_type": "INHIBITOR",
                    }
                ]
            }
        ),
    ]
    tool = get_tool("query_chembl")

    async def _stub(_):  # noqa: ANN001
        return queue.pop(0) if queue else ChEMBLQueryOutput(data={})

    tool.func = _stub  # type: ignore[assignment]

    entities, edges = await fetch_chembl_for_spec(
        _spec(seeds={"drug": ["imatinib"]}, sources=[SourceName.CHEMBL])
    )

    ids = {e.node_id for e in entities}
    assert "ChEMBL:CHEMBL941" in ids
    assert "ChEMBL:CHEMBL1862" in ids

    assert any(e.relation == RelationType.INHIBITS for e in edges)


# -- DrugBank ----------------------------------------------------------------


async def test_drugbank_skipped_without_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SAMHITA_DRUGBANK_PATH", raising=False)
    entities, edges = await fetch_drugbank_for_spec(_spec())
    assert entities == [] and edges == []


async def test_drugbank_with_env_uses_lookup_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SAMHITA_DRUGBANK_PATH", "/tmp/fake_dump.json")

    async def _stub(_):  # noqa: ANN001
        return DrugBankLookupOutput(
            drug_id="imatinib",
            name="IMATINIB",
            targets=[
                {
                    "gene_name": "ABL1",
                    "uniprot": "P00519",
                }
            ],
            indications=["chronic myeloid leukemia"],
        )

    get_tool("lookup_drugbank").func = _stub  # type: ignore[assignment]

    entities, edges = await fetch_drugbank_for_spec(
        _spec(seeds={"drug": ["imatinib"]}, sources=[SourceName.DRUGBANK])
    )
    ids = {e.node_id for e in entities}
    assert "DrugBank:imatinib" in ids
    assert "UniProt:P00519" in ids
    assert any(e.relation == RelationType.TARGETS for e in edges)
    assert any(e.relation == RelationType.INDICATED_FOR for e in edges)
