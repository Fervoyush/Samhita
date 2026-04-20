"""OpenTargets spec-aware fetcher.

Turns KGSpec seeds into OpenTargets GraphQL calls and returns
Samhita-typed Entity / Edge objects with full Provenance.

Supported seeds:
- `disease`: search disease name → EFO ID → associated targets + indications
- `drug`:    search drug name → ChEMBL ID → linked targets + indications
- `gene`:    search gene symbol → Ensembl ID → associated diseases
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
from samhita.core.tools.fetch import OpenTargetsQueryInput

_SEARCH_QUERY = """
query Search($q: String!, $entityNames: [String!]) {
  search(queryString: $q, entityNames: $entityNames) {
    hits { id name entity }
  }
}
"""

_DISEASE_TARGETS_QUERY = """
query DiseaseTargets($efoId: String!, $size: Int!) {
  disease(efoId: $efoId) {
    id
    name
    associatedTargets(page: {index: 0, size: $size}) {
      rows {
        score
        target { id approvedSymbol approvedName }
      }
    }
    knownDrugs(size: $size) {
      rows {
        drug { id name }
        phase
      }
    }
  }
}
"""

_DRUG_DETAIL_QUERY = """
query DrugDetail($chemblId: String!) {
  drug(chemblId: $chemblId) {
    id
    name
    linkedTargets { rows { id approvedSymbol } }
    indications {
      rows { disease { id name } }
    }
  }
}
"""

_TARGET_DISEASES_QUERY = """
query TargetDiseases($ensemblId: String!, $size: Int!) {
  target(ensemblId: $ensemblId) {
    id
    approvedSymbol
    approvedName
    associatedDiseases(page: {index: 0, size: $size}) {
      rows {
        score
        disease { id name }
      }
    }
  }
}
"""


async def fetch_opentargets_for_spec(
    spec: KGSpec,
) -> tuple[list[Entity], list[Edge]]:
    """Fetch structured entities + edges from OpenTargets for the given spec."""
    tool = get_tool("query_opentargets")

    entities: dict[str, Entity] = {}
    edges: list[Edge] = []
    per_source_cap = min(spec.max_papers, 50)

    async def _search(query_string: str, entity_name: str) -> dict[str, Any] | None:
        out = await tool.func(
            OpenTargetsQueryInput(
                gql_query=_SEARCH_QUERY,
                variables={"q": query_string, "entityNames": [entity_name]},
            )
        )
        if getattr(out, "error", None):
            return None
        hits = (out.data.get("search") or {}).get("hits") or []
        return hits[0] if hits else None

    # -- Disease seeds -------------------------------------------------------
    for disease_name in spec.seeds.get("disease", []):
        hit = await _search(disease_name, "disease")
        if not hit or not hit.get("id"):
            continue
        efo_id = hit["id"]
        disease_entity = _entity(
            EntityType.DISEASE,
            hit.get("name") or disease_name,
            Identifier(namespace="EFO", value=efo_id),
        )
        _merge(entities, disease_entity)

        out = await tool.func(
            OpenTargetsQueryInput(
                gql_query=_DISEASE_TARGETS_QUERY,
                variables={"efoId": efo_id, "size": per_source_cap},
            )
        )
        if getattr(out, "error", None):
            continue

        disease_block = (out.data or {}).get("disease") or {}
        for row in (disease_block.get("associatedTargets") or {}).get("rows", []) or []:
            tgt = row.get("target") or {}
            if not tgt.get("id"):
                continue
            gene_entity = _entity(
                EntityType.GENE,
                tgt.get("approvedSymbol") or tgt.get("approvedName") or tgt["id"],
                Identifier(namespace="Ensembl", value=tgt["id"]),
            )
            _merge(entities, gene_entity)
            edges.append(
                _edge(
                    subject=gene_entity,
                    relation=RelationType.ASSOCIATED_WITH,
                    object_=disease_entity,
                    source_id=f"OpenTargets:disease:{efo_id}",
                    confidence=float(row.get("score") or 0.0) or 0.5,
                )
            )

        for row in (disease_block.get("knownDrugs") or {}).get("rows", []) or []:
            drug = row.get("drug") or {}
            if not drug.get("id"):
                continue
            drug_entity = _entity(
                EntityType.DRUG,
                drug.get("name") or drug["id"],
                Identifier(namespace="ChEMBL", value=drug["id"]),
            )
            _merge(entities, drug_entity)
            edges.append(
                _edge(
                    subject=drug_entity,
                    relation=RelationType.TREATS,
                    object_=disease_entity,
                    source_id=f"OpenTargets:disease:{efo_id}",
                    confidence=0.9,
                    properties={"phase": row.get("phase")} if row.get("phase") is not None else {},
                )
            )

    # -- Drug seeds ---------------------------------------------------------
    for drug_name in spec.seeds.get("drug", []):
        hit = await _search(drug_name, "drug")
        if not hit or not hit.get("id"):
            continue
        chembl_id = hit["id"]
        drug_entity = _entity(
            EntityType.DRUG,
            hit.get("name") or drug_name,
            Identifier(namespace="ChEMBL", value=chembl_id),
        )
        _merge(entities, drug_entity)

        out = await tool.func(
            OpenTargetsQueryInput(
                gql_query=_DRUG_DETAIL_QUERY, variables={"chemblId": chembl_id}
            )
        )
        if getattr(out, "error", None):
            continue

        drug_block = (out.data or {}).get("drug") or {}
        for row in (drug_block.get("linkedTargets") or {}).get("rows", []) or []:
            if not row.get("id"):
                continue
            gene_entity = _entity(
                EntityType.GENE,
                row.get("approvedSymbol") or row["id"],
                Identifier(namespace="Ensembl", value=row["id"]),
            )
            _merge(entities, gene_entity)
            edges.append(
                _edge(
                    subject=drug_entity,
                    relation=RelationType.TARGETS,
                    object_=gene_entity,
                    source_id=f"OpenTargets:drug:{chembl_id}",
                    confidence=0.95,
                )
            )
        for row in (drug_block.get("indications") or {}).get("rows", []) or []:
            disease = row.get("disease") or {}
            if not disease.get("id"):
                continue
            disease_entity = _entity(
                EntityType.DISEASE,
                disease.get("name") or disease["id"],
                Identifier(namespace="EFO", value=disease["id"]),
            )
            _merge(entities, disease_entity)
            edges.append(
                _edge(
                    subject=drug_entity,
                    relation=RelationType.INDICATED_FOR,
                    object_=disease_entity,
                    source_id=f"OpenTargets:drug:{chembl_id}",
                    confidence=0.95,
                )
            )

    # -- Gene seeds ---------------------------------------------------------
    for gene_name in spec.seeds.get("gene", []):
        hit = await _search(gene_name, "target")
        if not hit or not hit.get("id"):
            continue
        ensembl_id = hit["id"]
        gene_entity = _entity(
            EntityType.GENE,
            hit.get("name") or gene_name,
            Identifier(namespace="Ensembl", value=ensembl_id),
        )
        _merge(entities, gene_entity)

        out = await tool.func(
            OpenTargetsQueryInput(
                gql_query=_TARGET_DISEASES_QUERY,
                variables={"ensemblId": ensembl_id, "size": per_source_cap},
            )
        )
        if getattr(out, "error", None):
            continue

        target_block = (out.data or {}).get("target") or {}
        for row in (target_block.get("associatedDiseases") or {}).get("rows", []) or []:
            disease = row.get("disease") or {}
            if not disease.get("id"):
                continue
            disease_entity = _entity(
                EntityType.DISEASE,
                disease.get("name") or disease["id"],
                Identifier(namespace="EFO", value=disease["id"]),
            )
            _merge(entities, disease_entity)
            edges.append(
                _edge(
                    subject=gene_entity,
                    relation=RelationType.ASSOCIATED_WITH,
                    object_=disease_entity,
                    source_id=f"OpenTargets:target:{ensembl_id}",
                    confidence=float(row.get("score") or 0.0) or 0.5,
                )
            )

    return list(entities.values()), edges


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _entity(etype: EntityType, name: str, primary: Identifier) -> Entity:
    return Entity(entity_type=etype, name=name, primary_id=primary)


def _merge(store: dict[str, Entity], entity: Entity) -> None:
    key = entity.node_id
    if key not in store:
        store[key] = entity


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
            source_type=SourceType.OPENTARGETS,
            extracting_model=None,
            section=SectionType.UNKNOWN,
        ),
        properties=properties or {},
    )
