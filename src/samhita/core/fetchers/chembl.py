"""ChEMBL spec-aware fetcher.

Turns drug seeds into ChEMBL REST calls and produces Samhita-typed
Entity / Edge objects with Provenance. Supported seed:

- ``drug``: molecule/search -> mechanism (mechanism_of_action + target)

OpenTargets already covers target-/disease-centric queries at a higher
level of curation, so this fetcher is deliberately drug-centric.
"""

from __future__ import annotations

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
from samhita.core.tools.fetch import ChEMBLQueryInput


async def fetch_chembl_for_spec(spec: KGSpec) -> tuple[list[Entity], list[Edge]]:
    tool = get_tool("query_chembl")

    entities: dict[str, Entity] = {}
    edges: list[Edge] = []

    for drug_name in spec.seeds.get("drug", []):
        chembl_id = await _resolve_drug_to_chembl_id(tool, drug_name)
        if not chembl_id:
            continue
        drug_entity = _entity(
            EntityType.DRUG,
            drug_name,
            Identifier(namespace="ChEMBL", value=chembl_id),
        )
        _merge(entities, drug_entity)

        mech_out = await tool.func(
            ChEMBLQueryInput(
                endpoint="mechanism",
                params={"molecule_chembl_id": chembl_id},
            )
        )
        if getattr(mech_out, "error", None):
            continue

        for mech in (mech_out.data or {}).get("mechanisms", []) or []:
            target_id = mech.get("target_chembl_id")
            if not target_id:
                continue
            target_name = mech.get("mechanism_of_action") or target_id
            target_entity = _entity(
                EntityType.PROTEIN,
                target_name,
                Identifier(namespace="ChEMBL", value=target_id),
            )
            _merge(entities, target_entity)

            action = str(mech.get("action_type") or "").lower()
            relation = RelationType.TARGETS
            if "inhibit" in action:
                relation = RelationType.INHIBITS
            elif "activ" in action or "agonist" in action:
                relation = RelationType.ACTIVATES

            edges.append(
                _edge(
                    subject=drug_entity,
                    relation=relation,
                    object_=target_entity,
                    source_id=f"ChEMBL:mechanism:{chembl_id}",
                    confidence=0.9,
                    properties={
                        "action_type": mech.get("action_type") or "",
                        "mechanism": mech.get("mechanism_of_action") or "",
                    },
                )
            )

    return list(entities.values()), edges


async def _resolve_drug_to_chembl_id(tool: Any, name: str) -> str | None:
    out = await tool.func(
        ChEMBLQueryInput(
            endpoint="molecule/search",
            params={"q": name, "limit": 1},
        )
    )
    if getattr(out, "error", None):
        return None
    molecules = (out.data or {}).get("molecules") or []
    if not molecules:
        return None
    return molecules[0].get("molecule_chembl_id")


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
            source_type=SourceType.CHEMBL,
            extracting_model=None,
            section=SectionType.UNKNOWN,
        ),
        properties=properties or {},
    )
