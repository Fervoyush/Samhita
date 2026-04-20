"""DrugBank spec-aware fetcher.

DrugBank's full dataset is license-restricted; users must supply a
local JSON dump (path via ``$SAMHITA_DRUGBANK_PATH`` or CLI flag).
When the dump is absent this fetcher returns empty lists silently.
"""

from __future__ import annotations

import os
from typing import Any

from samhita.core.schemas import (
    Edge,
    Entity,
    EntityType,
    Identifier,
    KGSpec,
    Provenance,
    RelationType,
    SectionType,
    SourceType,
)
from samhita.core.tools import get_tool
from samhita.core.tools.fetch import DrugBankLookupInput


async def fetch_drugbank_for_spec(spec: KGSpec) -> tuple[list[Entity], list[Edge]]:
    local_path = os.getenv("SAMHITA_DRUGBANK_PATH", "")
    if not local_path:
        # DrugBank is optional — skip silently rather than erroring.
        return [], []

    tool = get_tool("lookup_drugbank")
    entities: dict[str, Entity] = {}
    edges: list[Edge] = []

    for drug_name in spec.seeds.get("drug", []):
        out = await tool.func(
            DrugBankLookupInput(drug_id=drug_name, local_dump_path=local_path)
        )
        if getattr(out, "error", None) or not getattr(out, "name", ""):
            continue

        drug_entity = _entity(
            EntityType.DRUG,
            out.name or drug_name,
            Identifier(namespace="DrugBank", value=out.drug_id),
        )
        _merge(entities, drug_entity)

        for target in out.targets or []:
            symbol = (target.get("gene_name") or target.get("name") or "").strip()
            external_id = target.get("uniprot") or target.get("id") or symbol
            if not symbol or not external_id:
                continue
            target_entity = _entity(
                EntityType.GENE,
                symbol,
                Identifier(
                    namespace="UniProt" if target.get("uniprot") else "DrugBank",
                    value=str(external_id),
                ),
            )
            _merge(entities, target_entity)
            edges.append(
                _edge(
                    subject=drug_entity,
                    relation=RelationType.TARGETS,
                    object_=target_entity,
                    source_id=f"DrugBank:{out.drug_id}",
                    confidence=0.9,
                )
            )

        for indication in out.indications or []:
            indication_name = str(indication).strip()
            if not indication_name:
                continue
            disease_entity = _entity(
                EntityType.DISEASE,
                indication_name,
                Identifier(namespace="local", value=_slug(indication_name)),
            )
            _merge(entities, disease_entity)
            edges.append(
                _edge(
                    subject=drug_entity,
                    relation=RelationType.INDICATED_FOR,
                    object_=disease_entity,
                    source_id=f"DrugBank:{out.drug_id}",
                    confidence=0.95,
                )
            )

    return list(entities.values()), edges


def _entity(etype: EntityType, name: str, primary: Identifier) -> Entity:
    return Entity(entity_type=etype, name=name, primary_id=primary)


def _merge(store: dict[str, Entity], entity: Entity) -> None:
    if entity.node_id not in store:
        store[entity.node_id] = entity


def _edge(
    *,
    subject: Entity,
    relation: RelationType,
    object_: Entity,
    source_id: str,
    confidence: float,
    properties: dict[str, Any] | None = None,
) -> Edge:
    return Edge(
        relation=relation,
        subject_id=subject.node_id,
        object_id=object_.node_id,
        confidence=max(0.0, min(1.0, float(confidence))),
        provenance=Provenance(
            source_id=source_id,
            source_type=SourceType.DRUGBANK,
            extracting_model=None,
            section=SectionType.UNKNOWN,
        ),
        properties=properties or {},
    )


def _slug(value: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in value.lower()).strip("_")[:64]
