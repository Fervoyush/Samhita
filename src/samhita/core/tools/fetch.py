"""Data-source fetch tools — schemas and stubs.

Implementations land in the Week 1 Day 4-5 scaffolding step. For now
this module only declares the typed I/O and placeholder functions so
the orchestrators can wire against a stable interface.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# --- PubMed Central (full-text via PMCGrab) ----------------------------------


class PMCFetchInput(BaseModel):
    pmc_id: str = Field(description="PubMed Central ID, e.g. 'PMC1234567'")


class PMCFetchOutput(BaseModel):
    pmc_id: str
    title: str
    sections: dict[str, str] = Field(
        description="section_name -> concatenated text, e.g. {'methods': '...'}"
    )
    metadata: dict = Field(default_factory=dict)
    error: str | None = None


# --- PubMed abstracts (E-utilities fallback) ---------------------------------


class PubMedSearchInput(BaseModel):
    query: str
    max_results: int = 100
    filters: dict[str, str] = Field(default_factory=dict)


class PubMedSearchOutput(BaseModel):
    pmids: list[str]
    total_hits: int


class PubMedAbstractInput(BaseModel):
    pmid: str


class PubMedAbstractOutput(BaseModel):
    pmid: str
    title: str
    abstract: str
    doi: str | None = None
    pmc_id: str | None = None


# --- OpenTargets -------------------------------------------------------------


class OpenTargetsQueryInput(BaseModel):
    gql_query: str
    variables: dict = Field(default_factory=dict)


class OpenTargetsQueryOutput(BaseModel):
    data: dict


# --- ChEMBL ------------------------------------------------------------------


class ChEMBLQueryInput(BaseModel):
    endpoint: str = Field(description="e.g. 'molecule', 'target', 'activity'")
    params: dict = Field(default_factory=dict)


class ChEMBLQueryOutput(BaseModel):
    data: dict


# --- DrugBank ----------------------------------------------------------------


class DrugBankLookupInput(BaseModel):
    drug_id: str


class DrugBankLookupOutput(BaseModel):
    drug_id: str
    name: str
    targets: list[dict] = Field(default_factory=list)
    indications: list[str] = Field(default_factory=list)


# --- Placeholder implementations ---------------------------------------------


async def _not_implemented(_: BaseModel) -> BaseModel:
    """Scheduled for Day 4-5 of Week 1."""
    raise NotImplementedError("Fetch tool implementation pending (Week 1, Day 4-5)")
