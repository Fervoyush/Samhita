"""Data-source fetch tools — typed I/O schemas and async implementations.

Each source exposes one or more typed async functions that are registered
in the central tool registry (see `core.tools.register_fetch_tools`).
Every implementation returns a typed Pydantic model and never raises —
errors are surfaced in the `error` field so agent orchestrators can
decide how to handle them.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
from pydantic import BaseModel, Field

from samhita.core.tools import Tool, register_tool

# ---------------------------------------------------------------------------
# Shared HTTP defaults
# ---------------------------------------------------------------------------

_DEFAULT_TIMEOUT = httpx.Timeout(30.0, connect=10.0)
_DEFAULT_HEADERS = {
    "User-Agent": "samhita-kg-builder/0.0.1 (+https://github.com/Fervoyush/Samhita)",
    "Accept": "application/json",
}


# =============================================================================
# PubMed Central (full-text via PMCGrab)
# =============================================================================


class PMCFetchInput(BaseModel):
    pmc_id: str = Field(description="PubMed Central ID, e.g. 'PMC1234567' or '1234567'.")


class PMCFetchOutput(BaseModel):
    pmc_id: str
    title: str = ""
    sections: dict[str, str] = Field(
        default_factory=dict,
        description="section_name -> concatenated text, e.g. {'methods': '...'}",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


async def fetch_pmc_paper(payload: PMCFetchInput) -> PMCFetchOutput:
    """Fetch a PMC paper via PMCGrab.

    PMCGrab's public API changes occasionally; we adapt at call time so
    this tool survives minor surface shifts. If the library is not
    installed the tool returns a typed error instead of raising.
    """
    try:
        import pmcgrab  # type: ignore[import-not-found]
    except ImportError:
        return PMCFetchOutput(
            pmc_id=payload.pmc_id,
            error="pmcgrab is not installed; `pip install pmcgrab`.",
        )

    def _pick_callable() -> Any:
        # PMCGrab exposes several fetch entry points across versions.
        for candidate in ("fetch", "get_paper", "get_article", "fetch_pmc"):
            fn = getattr(pmcgrab, candidate, None)
            if callable(fn):
                return fn
        return None

    fetch_fn = _pick_callable()
    if fetch_fn is None:
        return PMCFetchOutput(
            pmc_id=payload.pmc_id,
            error="No recognized PMCGrab entry point found on installed module.",
        )

    try:
        result = fetch_fn(payload.pmc_id)
    except Exception as exc:  # noqa: BLE001 — surface all errors to agent
        return PMCFetchOutput(pmc_id=payload.pmc_id, error=str(exc))

    return _normalize_pmc_result(payload.pmc_id, result)


def _normalize_pmc_result(pmc_id: str, raw: Any) -> PMCFetchOutput:
    """Best-effort adaptation of PMCGrab's return value into our schema.

    Accepts dict-like JSON, objects with `.title` + `.sections`, or raw JSON
    strings. Unknown shapes yield an error rather than a crash.
    """
    # Raw JSON string
    if isinstance(raw, (str, bytes)):
        try:
            raw = json.loads(raw)
        except Exception as exc:  # noqa: BLE001
            return PMCFetchOutput(pmc_id=pmc_id, error=f"unparseable PMCGrab output: {exc}")

    # Dict-style (the README implies JSON output with sections)
    if isinstance(raw, dict):
        title = raw.get("title", "") or raw.get("metadata", {}).get("title", "")
        body = raw.get("body") or raw.get("sections") or {}
        if isinstance(body, list):
            # list of {section, text} shape
            sections: dict[str, str] = {}
            for item in body:
                if not isinstance(item, dict):
                    continue
                key = (item.get("section") or item.get("name") or "unknown").lower()
                text = item.get("text") or item.get("content") or ""
                sections[key] = (sections.get(key, "") + "\n" + text).strip()
        elif isinstance(body, dict):
            sections = {k.lower(): v for k, v in body.items() if isinstance(v, str)}
        else:
            sections = {}
        metadata = raw.get("metadata", {}) if isinstance(raw.get("metadata"), dict) else {}
        return PMCFetchOutput(
            pmc_id=pmc_id, title=str(title), sections=sections, metadata=metadata
        )

    # Attribute-style object
    title = getattr(raw, "title", "") or ""
    sections_attr = getattr(raw, "sections", None)
    sections: dict[str, str] = {}
    if isinstance(sections_attr, dict):
        sections = {str(k).lower(): str(v) for k, v in sections_attr.items()}
    elif isinstance(sections_attr, (list, tuple)):
        for item in sections_attr:
            name = (
                getattr(item, "name", None)
                or getattr(item, "type", None)
                or "unknown"
            )
            text = getattr(item, "text", "") or getattr(item, "content", "")
            key = str(name).lower()
            sections[key] = (sections.get(key, "") + "\n" + str(text)).strip()
    metadata = getattr(raw, "metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    return PMCFetchOutput(pmc_id=pmc_id, title=str(title), sections=sections, metadata=metadata)


# =============================================================================
# PubMed abstracts (E-utilities fallback)
# =============================================================================


_EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


class PubMedSearchInput(BaseModel):
    query: str
    max_results: int = Field(default=100, ge=1, le=10_000)
    filters: dict[str, str] = Field(default_factory=dict)


class PubMedSearchOutput(BaseModel):
    pmids: list[str] = Field(default_factory=list)
    total_hits: int = 0
    error: str | None = None


async def search_pubmed(payload: PubMedSearchInput) -> PubMedSearchOutput:
    """Search PubMed via E-utilities esearch, returning a list of PMIDs."""
    # Apply simple filter operators (e.g. {'date': '2025:2026[dp]'})
    term = payload.query
    for key, value in payload.filters.items():
        term = f"{term} AND {value}[{key}]" if key and value else term

    params = {
        "db": "pubmed",
        "term": term,
        "retmax": str(payload.max_results),
        "retmode": "json",
    }
    try:
        async with httpx.AsyncClient(
            timeout=_DEFAULT_TIMEOUT, headers=_DEFAULT_HEADERS
        ) as client:
            resp = await client.get(f"{_EUTILS_BASE}/esearch.fcgi", params=params)
            resp.raise_for_status()
            body = resp.json()
    except Exception as exc:  # noqa: BLE001
        return PubMedSearchOutput(error=str(exc))

    esearch = body.get("esearchresult", {})
    return PubMedSearchOutput(
        pmids=list(esearch.get("idlist", []) or []),
        total_hits=int(esearch.get("count", 0) or 0),
    )


class PubMedAbstractInput(BaseModel):
    pmid: str


class PubMedAbstractOutput(BaseModel):
    pmid: str
    title: str = ""
    abstract: str = ""
    doi: str | None = None
    pmc_id: str | None = None
    error: str | None = None


async def fetch_pubmed_abstract(payload: PubMedAbstractInput) -> PubMedAbstractOutput:
    """Fetch a single PubMed record (esummary + efetch) and extract abstract text."""
    params = {
        "db": "pubmed",
        "id": payload.pmid,
        "rettype": "abstract",
        "retmode": "xml",
    }
    try:
        async with httpx.AsyncClient(
            timeout=_DEFAULT_TIMEOUT, headers=_DEFAULT_HEADERS
        ) as client:
            resp = await client.get(f"{_EUTILS_BASE}/efetch.fcgi", params=params)
            resp.raise_for_status()
            xml_text = resp.text
    except Exception as exc:  # noqa: BLE001
        return PubMedAbstractOutput(pmid=payload.pmid, error=str(exc))

    title, abstract, doi, pmc_id = _parse_pubmed_xml(xml_text)
    return PubMedAbstractOutput(
        pmid=payload.pmid, title=title, abstract=abstract, doi=doi, pmc_id=pmc_id
    )


def _parse_pubmed_xml(xml_text: str) -> tuple[str, str, str | None, str | None]:
    """Very light XML scrape of a PubMed efetch record.

    Uses the stdlib ElementTree to avoid pulling in lxml. Returns
    (title, abstract, doi, pmc_id), with empty strings / None where
    fields are missing.
    """
    try:
        import xml.etree.ElementTree as ET

        root = ET.fromstring(xml_text)
    except Exception:  # noqa: BLE001
        return "", "", None, None

    title_el = root.find(".//ArticleTitle")
    abs_parts = [el.text or "" for el in root.findall(".//Abstract/AbstractText")]
    doi = None
    pmc_id = None
    for aid in root.findall(".//ArticleId"):
        id_type = (aid.get("IdType") or "").lower()
        if id_type == "doi" and aid.text:
            doi = aid.text.strip()
        elif id_type == "pmc" and aid.text:
            pmc_id = aid.text.strip()

    return (
        (title_el.text or "").strip() if title_el is not None else "",
        " ".join(part.strip() for part in abs_parts if part).strip(),
        doi,
        pmc_id,
    )


# =============================================================================
# OpenTargets (GraphQL)
# =============================================================================


_OPENTARGETS_ENDPOINT = "https://api.platform.opentargets.org/api/v4/graphql"


class OpenTargetsQueryInput(BaseModel):
    gql_query: str
    variables: dict[str, Any] = Field(default_factory=dict)


class OpenTargetsQueryOutput(BaseModel):
    data: dict[str, Any] = Field(default_factory=dict)
    errors: list[dict[str, Any]] = Field(default_factory=list)
    error: str | None = None


async def query_opentargets(payload: OpenTargetsQueryInput) -> OpenTargetsQueryOutput:
    """POST a GraphQL query against the OpenTargets Platform API."""
    body = {"query": payload.gql_query, "variables": payload.variables}
    headers = {**_DEFAULT_HEADERS, "Content-Type": "application/json"}
    try:
        async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT, headers=headers) as client:
            resp = await client.post(_OPENTARGETS_ENDPOINT, json=body)
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:  # noqa: BLE001
        return OpenTargetsQueryOutput(error=str(exc))

    return OpenTargetsQueryOutput(
        data=data.get("data", {}) or {},
        errors=list(data.get("errors", []) or []),
    )


# =============================================================================
# ChEMBL (REST)
# =============================================================================


_CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data"


class ChEMBLQueryInput(BaseModel):
    endpoint: str = Field(description="e.g. 'molecule', 'target', 'activity'")
    params: dict[str, Any] = Field(default_factory=dict)


class ChEMBLQueryOutput(BaseModel):
    data: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


async def query_chembl(payload: ChEMBLQueryInput) -> ChEMBLQueryOutput:
    """GET a ChEMBL REST endpoint. Defaults to JSON."""
    endpoint = payload.endpoint.strip("/")
    params = {"format": "json", **payload.params}
    try:
        async with httpx.AsyncClient(
            timeout=_DEFAULT_TIMEOUT, headers=_DEFAULT_HEADERS
        ) as client:
            resp = await client.get(f"{_CHEMBL_BASE}/{endpoint}", params=params)
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:  # noqa: BLE001
        return ChEMBLQueryOutput(error=str(exc))

    return ChEMBLQueryOutput(data=data)


# =============================================================================
# DrugBank (minimal local-file lookup)
# =============================================================================


class DrugBankLookupInput(BaseModel):
    drug_id: str
    local_dump_path: str | None = Field(
        default=None,
        description=(
            "Optional path to a DrugBank Open Data JSON index. "
            "Full DrugBank is license-restricted; users must supply their own dump."
        ),
    )


class DrugBankLookupOutput(BaseModel):
    drug_id: str
    name: str = ""
    targets: list[dict[str, Any]] = Field(default_factory=list)
    indications: list[str] = Field(default_factory=list)
    error: str | None = None


async def lookup_drugbank(payload: DrugBankLookupInput) -> DrugBankLookupOutput:
    """Lookup a drug in a user-provided DrugBank JSON index.

    DrugBank's full database is license-restricted, so Samhita does not
    ship it. If a local JSON index keyed by DrugBank ID is available,
    this tool reads it. Otherwise it returns a typed error describing
    the missing requirement.
    """
    if not payload.local_dump_path:
        return DrugBankLookupOutput(
            drug_id=payload.drug_id,
            error=(
                "No DrugBank local_dump_path provided. "
                "Set SAMHITA_DRUGBANK_PATH or pass local_dump_path explicitly."
            ),
        )

    try:
        with open(payload.local_dump_path, encoding="utf-8") as fh:
            index = json.load(fh)
    except Exception as exc:  # noqa: BLE001
        return DrugBankLookupOutput(drug_id=payload.drug_id, error=str(exc))

    record = index.get(payload.drug_id)
    if record is None:
        return DrugBankLookupOutput(
            drug_id=payload.drug_id, error=f"{payload.drug_id!r} not found in local dump"
        )

    return DrugBankLookupOutput(
        drug_id=payload.drug_id,
        name=str(record.get("name", "")),
        targets=list(record.get("targets", []) or []),
        indications=list(record.get("indications", []) or []),
    )


# =============================================================================
# Registration
# =============================================================================


def register_fetch_tools() -> None:
    """Register every fetch tool in the central registry.

    Called by :func:`samhita.core.bootstrap.bootstrap_tools`. Idempotent —
    safe to call multiple times (silently skips already-registered tools).
    """
    tools: list[Tool] = [
        Tool(
            name="fetch_pmc_paper",
            description="Fetch a PubMed Central paper's section-aware text via PMCGrab.",
            input_schema=PMCFetchInput,
            output_schema=PMCFetchOutput,
            func=fetch_pmc_paper,
            tags=["fetch", "pmc", "fulltext"],
        ),
        Tool(
            name="search_pubmed",
            description="Search PubMed and return a list of PMIDs for a given query.",
            input_schema=PubMedSearchInput,
            output_schema=PubMedSearchOutput,
            func=search_pubmed,
            tags=["fetch", "pubmed", "search"],
        ),
        Tool(
            name="fetch_pubmed_abstract",
            description="Fetch a single PubMed abstract + metadata by PMID.",
            input_schema=PubMedAbstractInput,
            output_schema=PubMedAbstractOutput,
            func=fetch_pubmed_abstract,
            tags=["fetch", "pubmed", "abstract"],
        ),
        Tool(
            name="query_opentargets",
            description="POST a GraphQL query to the OpenTargets Platform API.",
            input_schema=OpenTargetsQueryInput,
            output_schema=OpenTargetsQueryOutput,
            func=query_opentargets,
            tags=["fetch", "opentargets", "structured"],
        ),
        Tool(
            name="query_chembl",
            description="GET a ChEMBL REST endpoint (e.g. 'molecule', 'target').",
            input_schema=ChEMBLQueryInput,
            output_schema=ChEMBLQueryOutput,
            func=query_chembl,
            tags=["fetch", "chembl", "structured"],
        ),
        Tool(
            name="lookup_drugbank",
            description="Lookup a drug in a user-provided DrugBank JSON dump.",
            input_schema=DrugBankLookupInput,
            output_schema=DrugBankLookupOutput,
            func=lookup_drugbank,
            tags=["fetch", "drugbank", "structured"],
        ),
    ]

    from samhita.core.tools import all_tools

    existing = all_tools()
    for tool in tools:
        if tool.name in existing:
            continue
        register_tool(tool)
