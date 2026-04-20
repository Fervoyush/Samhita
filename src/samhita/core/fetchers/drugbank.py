"""DrugBank spec-aware fetcher.

DrugBank's full dataset is license-restricted; users must supply a
local JSON dump (path via ``$SAMHITA_DRUGBANK_PATH`` or CLI flag).
When the dump is absent this fetcher returns empty lists silently.
"""

from __future__ import annotations

import os

from samhita.core.fetchers._helpers import make_edge, make_entity, merge_entity, slugify
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
from samhita.core.tools.fetch import DrugBankLookupInput


async def fetch_drugbank_for_spec(spec: KGSpec) -> tuple[list[Entity], list[Edge]]:
    local_path = os.getenv("SAMHITA_DRUGBANK_PATH", "")
    if not local_path:
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

        drug_entity = make_entity(
            EntityType.DRUG,
            out.name or drug_name,
            Identifier(namespace=NamespaceName.DRUGBANK, value=out.drug_id),
        )
        merge_entity(entities, drug_entity)

        for target in out.targets or []:
            symbol = (target.get("gene_name") or target.get("name") or "").strip()
            external_id = target.get("uniprot") or target.get("id") or symbol
            if not symbol or not external_id:
                continue
            namespace = (
                NamespaceName.UNIPROT if target.get("uniprot") else NamespaceName.DRUGBANK
            )
            target_entity = make_entity(
                EntityType.GENE,
                symbol,
                Identifier(namespace=namespace, value=str(external_id)),
            )
            merge_entity(entities, target_entity)
            edges.append(
                make_edge(
                    subject=drug_entity,
                    relation=RelationType.TARGETS,
                    object_=target_entity,
                    source_type=SourceType.DRUGBANK,
                    source_id=f"DrugBank:{out.drug_id}",
                    confidence=0.9,
                )
            )

        for indication in out.indications or []:
            indication_name = str(indication).strip()
            if not indication_name:
                continue
            disease_entity = make_entity(
                EntityType.DISEASE,
                indication_name,
                Identifier(
                    namespace=NamespaceName.LOCAL, value=slugify(indication_name)
                ),
            )
            merge_entity(entities, disease_entity)
            edges.append(
                make_edge(
                    subject=drug_entity,
                    relation=RelationType.INDICATED_FOR,
                    object_=disease_entity,
                    source_type=SourceType.DRUGBANK,
                    source_id=f"DrugBank:{out.drug_id}",
                    confidence=0.95,
                )
            )

    return list(entities.values()), edges
