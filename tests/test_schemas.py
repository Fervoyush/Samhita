"""Schema smoke tests."""

from samhita.core.schemas import (
    Entity,
    EntityType,
    Identifier,
    KGSpec,
    Provenance,
    RecipeName,
    SourceName,
    SourceType,
)


def test_entity_roundtrip() -> None:
    e = Entity(
        entity_type=EntityType.DRUG,
        name="imatinib",
        primary_id=Identifier(namespace="ChEMBL", value="CHEMBL941"),
    )
    assert e.node_id == "ChEMBL:CHEMBL941"


def test_identifier_is_frozen() -> None:
    ident = Identifier(namespace="HGNC", value="1097")
    assert str(ident) == "HGNC:1097"


def test_kgspec_defaults() -> None:
    spec = KGSpec(
        recipe=RecipeName.DRUG_TARGET_DISEASE,
        seeds={"disease": ["atopic dermatitis"]},
        sources=[SourceName.OPENTARGETS, SourceName.CHEMBL],
        entity_types=[EntityType.DRUG, EntityType.DISEASE],
        relation_types=[],
        original_request="build me a KG of drugs for AD",
    )
    assert spec.max_papers == 500
    assert spec.max_depth == 2


def test_provenance_defaults() -> None:
    p = Provenance(source_id="PMID:12345", source_type=SourceType.PUBMED)
    assert p.cost_usd == 0.0
    assert p.cache_hit is False
