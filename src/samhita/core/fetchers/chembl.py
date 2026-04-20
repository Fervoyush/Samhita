"""ChEMBL spec-aware fetcher.

Turns drug seeds into ChEMBL REST calls and produces Samhita-typed
Entity / Edge objects with Provenance. Drug-centric: OpenTargets
already covers target- and disease-centric queries at a higher level
of curation.
"""

from __future__ import annotations

from typing import Any

from samhita.core.fetchers._helpers import make_edge, make_entity, merge_entity
from samhita.core.schemas import (
    Edge,
    Entity,
    EntityType,
    Identifier,
    KGSpec,
    NamespaceName,
    RelationType,
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
        drug_entity = make_entity(
            EntityType.DRUG,
            drug_name,
            Identifier(namespace=NamespaceName.CHEMBL, value=chembl_id),
        )
        merge_entity(entities, drug_entity)

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
            target_entity = make_entity(
                EntityType.PROTEIN,
                target_name,
                Identifier(namespace=NamespaceName.CHEMBL, value=target_id),
            )
            merge_entity(entities, target_entity)

            edges.append(
                make_edge(
                    subject=drug_entity,
                    relation=_relation_from_action(mech.get("action_type")),
                    object_=target_entity,
                    source_type=SourceType.CHEMBL,
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


def _relation_from_action(action_type: str | None) -> RelationType:
    action = str(action_type or "").lower()
    if "inhibit" in action:
        return RelationType.INHIBITS
    if "activ" in action or "agonist" in action:
        return RelationType.ACTIVATES
    return RelationType.TARGETS
