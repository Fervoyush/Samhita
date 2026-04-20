"""ID normalization tools — deterministic lookups against public services.

Each normalizer returns a typed output with `method` indicating how the
ID was obtained (`deterministic` | `failed`). No LLM fallback in v1;
add that later as a separate tool if the failure rate is material.
"""

from __future__ import annotations

from typing import Any

import httpx
from pydantic import BaseModel, Field

from samhita.core.schemas import EntityType, Identifier
from samhita.core.tools import Tool, register_tools

_DEFAULT_TIMEOUT = httpx.Timeout(20.0, connect=5.0)
_DEFAULT_HEADERS = {
    "User-Agent": "samhita-kg-builder/0.0.1 (+https://github.com/Fervoyush/Samhita)",
    "Accept": "application/json",
}


class NormalizeEntityInput(BaseModel):
    name: str
    entity_type: EntityType
    context_hint: str | None = Field(
        default=None,
        description="surrounding text to disambiguate (e.g. species)",
    )


class NormalizeEntityOutput(BaseModel):
    name: str
    entity_type: EntityType
    primary_id: Identifier | None = None
    aliases: list[Identifier] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    method: str = Field(default="deterministic")
    error: str | None = None


# =============================================================================
# Gene / protein normalization via mygene.info
# =============================================================================


_MYGENE_ENDPOINT = "https://mygene.info/v3/query"


async def normalize_gene(payload: NormalizeEntityInput) -> NormalizeEntityOutput:
    """Resolve a gene symbol/name to HGNC / Entrez / Ensembl / UniProt IDs."""
    params = {
        "q": payload.name,
        "species": "human",
        "fields": "symbol,name,entrezgene,ensembl,uniprot,HGNC",
        "size": 1,
    }
    try:
        async with httpx.AsyncClient(
            timeout=_DEFAULT_TIMEOUT, headers=_DEFAULT_HEADERS
        ) as client:
            resp = await client.get(_MYGENE_ENDPOINT, params=params)
            resp.raise_for_status()
            body = resp.json()
    except Exception as exc:  # noqa: BLE001
        return NormalizeEntityOutput(
            name=payload.name,
            entity_type=payload.entity_type,
            method="failed",
            error=str(exc),
        )

    hits = body.get("hits") or []
    if not hits:
        return NormalizeEntityOutput(
            name=payload.name,
            entity_type=payload.entity_type,
            method="failed",
            error="no hits on mygene.info",
        )

    hit = hits[0]
    aliases: list[Identifier] = []

    hgnc = hit.get("HGNC")
    if hgnc:
        aliases.append(Identifier(namespace="HGNC", value=str(hgnc)))

    entrez = hit.get("entrezgene")
    if entrez:
        aliases.append(Identifier(namespace="NCBIGene", value=str(entrez)))

    ens = hit.get("ensembl")
    if isinstance(ens, dict) and ens.get("gene"):
        aliases.append(Identifier(namespace="Ensembl", value=str(ens["gene"])))
    elif isinstance(ens, list):
        for item in ens:
            if isinstance(item, dict) and item.get("gene"):
                aliases.append(Identifier(namespace="Ensembl", value=str(item["gene"])))

    uniprot = hit.get("uniprot")
    if isinstance(uniprot, dict):
        swissprot = uniprot.get("Swiss-Prot")
        if swissprot:
            if isinstance(swissprot, list):
                aliases.extend(Identifier(namespace="UniProt", value=str(v)) for v in swissprot)
            else:
                aliases.append(Identifier(namespace="UniProt", value=str(swissprot)))

    # Pick HGNC as primary if present; otherwise first alias
    primary: Identifier | None = None
    for alias in aliases:
        if alias.namespace == "HGNC":
            primary = alias
            break
    if primary is None and aliases:
        primary = aliases[0]

    if primary is None:
        return NormalizeEntityOutput(
            name=payload.name,
            entity_type=payload.entity_type,
            method="failed",
            error="hit returned no usable identifiers",
        )

    return NormalizeEntityOutput(
        name=payload.name,
        entity_type=payload.entity_type,
        primary_id=primary,
        aliases=[a for a in aliases if a != primary],
    )


# =============================================================================
# Disease / phenotype / pathway normalization via OLS (EBI)
# =============================================================================


_OLS_SEARCH = "https://www.ebi.ac.uk/ols4/api/search"

# Ontology preferences per entity type (used as OLS `ontology` filter).
_OLS_ONTOLOGIES: dict[EntityType, str] = {
    EntityType.DISEASE: "mondo",
    EntityType.PHENOTYPE: "hp",
    EntityType.PATHWAY: "reactome",
    EntityType.CELL_TYPE: "cl",
    EntityType.TISSUE: "uberon",
}


async def normalize_via_ols(payload: NormalizeEntityInput) -> NormalizeEntityOutput:
    """Normalize disease / phenotype / pathway terms via OLS4."""
    ontology = _OLS_ONTOLOGIES.get(payload.entity_type)
    if ontology is None:
        return NormalizeEntityOutput(
            name=payload.name,
            entity_type=payload.entity_type,
            method="failed",
            error=f"no OLS ontology configured for {payload.entity_type}",
        )

    params = {
        "q": payload.name,
        "ontology": ontology,
        "type": "class",
        "exact": "false",
        "rows": 1,
    }
    try:
        async with httpx.AsyncClient(
            timeout=_DEFAULT_TIMEOUT, headers=_DEFAULT_HEADERS
        ) as client:
            resp = await client.get(_OLS_SEARCH, params=params)
            resp.raise_for_status()
            body = resp.json()
    except Exception as exc:  # noqa: BLE001
        return NormalizeEntityOutput(
            name=payload.name,
            entity_type=payload.entity_type,
            method="failed",
            error=str(exc),
        )

    docs = (body.get("response") or {}).get("docs") or []
    if not docs:
        return NormalizeEntityOutput(
            name=payload.name,
            entity_type=payload.entity_type,
            method="failed",
            error="no OLS matches",
        )

    top = docs[0]
    obo_id = top.get("obo_id") or top.get("short_form") or ""
    if not obo_id or ":" not in obo_id:
        # obo_id looks like 'MONDO:0004980' — reject if it's missing the colon
        return NormalizeEntityOutput(
            name=payload.name,
            entity_type=payload.entity_type,
            method="failed",
            error=f"unusable OLS id: {obo_id!r}",
        )

    namespace, value = obo_id.split(":", 1)
    return NormalizeEntityOutput(
        name=payload.name,
        entity_type=payload.entity_type,
        primary_id=Identifier(namespace=namespace, value=value),
    )


# =============================================================================
# Drug normalization via ChEMBL molecule search
# =============================================================================


_CHEMBL_MOLECULE_SEARCH = "https://www.ebi.ac.uk/chembl/api/data/molecule/search"


async def normalize_drug(payload: NormalizeEntityInput) -> NormalizeEntityOutput:
    """Resolve a drug name to a ChEMBL ID via the molecule/search endpoint."""
    params: dict[str, Any] = {"q": payload.name, "format": "json", "limit": 1}
    try:
        async with httpx.AsyncClient(
            timeout=_DEFAULT_TIMEOUT, headers=_DEFAULT_HEADERS
        ) as client:
            resp = await client.get(_CHEMBL_MOLECULE_SEARCH, params=params)
            resp.raise_for_status()
            body = resp.json()
    except Exception as exc:  # noqa: BLE001
        return NormalizeEntityOutput(
            name=payload.name,
            entity_type=payload.entity_type,
            method="failed",
            error=str(exc),
        )

    molecules = body.get("molecules") or []
    if not molecules:
        return NormalizeEntityOutput(
            name=payload.name,
            entity_type=payload.entity_type,
            method="failed",
            error="no ChEMBL match",
        )

    top = molecules[0]
    chembl_id = top.get("molecule_chembl_id")
    if not chembl_id:
        return NormalizeEntityOutput(
            name=payload.name,
            entity_type=payload.entity_type,
            method="failed",
            error="ChEMBL record missing molecule_chembl_id",
        )

    return NormalizeEntityOutput(
        name=payload.name,
        entity_type=payload.entity_type,
        primary_id=Identifier(namespace="ChEMBL", value=str(chembl_id)),
    )


# =============================================================================
# Dispatcher
# =============================================================================


async def normalize_entity(payload: NormalizeEntityInput) -> NormalizeEntityOutput:
    """Dispatch to the right normalizer based on entity_type."""
    etype = payload.entity_type

    if etype in (EntityType.GENE, EntityType.PROTEIN):
        return await normalize_gene(payload)
    if etype == EntityType.DRUG:
        return await normalize_drug(payload)
    if etype in _OLS_ONTOLOGIES:
        return await normalize_via_ols(payload)

    return NormalizeEntityOutput(
        name=payload.name,
        entity_type=etype,
        method="failed",
        error=f"no normalizer registered for entity_type={etype}",
    )


# =============================================================================
# Registration
# =============================================================================


def register_normalize_tools() -> None:
    # Tools built per call so monkeypatched module attributes flow through.
    register_tools(
        [
            Tool(
                name="normalize_entity",
                description=(
                    "Dispatch to the right normalizer (mygene for genes, OLS for "
                    "diseases/phenotypes/pathways, ChEMBL for drugs)."
                ),
                input_schema=NormalizeEntityInput,
                output_schema=NormalizeEntityOutput,
                func=normalize_entity,
                tags=["normalize"],
            ),
            Tool(
                name="normalize_gene",
                description="Resolve a gene/protein name to HGNC/Entrez/Ensembl/UniProt IDs via mygene.info.",
                input_schema=NormalizeEntityInput,
                output_schema=NormalizeEntityOutput,
                func=normalize_gene,
                tags=["normalize", "gene"],
            ),
            Tool(
                name="normalize_drug",
                description="Resolve a drug name to a ChEMBL ID via the molecule/search endpoint.",
                input_schema=NormalizeEntityInput,
                output_schema=NormalizeEntityOutput,
                func=normalize_drug,
                tags=["normalize", "drug"],
            ),
            Tool(
                name="normalize_via_ols",
                description="Resolve disease/phenotype/pathway/cell/tissue terms via OLS4.",
                input_schema=NormalizeEntityInput,
                output_schema=NormalizeEntityOutput,
                func=normalize_via_ols,
                tags=["normalize", "ols"],
            ),
        ]
    )
