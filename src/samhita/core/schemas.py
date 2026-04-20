"""Core data schemas — framework-agnostic.

Every layer in Samhita speaks in these types. Agent frameworks (LangGraph,
Agent Zero, custom) serialize to/from them but must not mutate them.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


def _now() -> datetime:
    return datetime.now(timezone.utc)


class EntityType(str, Enum):
    """Biomedical entity types recognized by Samhita v1."""

    DRUG = "drug"
    GENE = "gene"
    PROTEIN = "protein"
    DISEASE = "disease"
    PATHWAY = "pathway"
    PHENOTYPE = "phenotype"
    CELL_TYPE = "cell_type"
    TISSUE = "tissue"
    VARIANT = "variant"


class RelationType(str, Enum):
    """Biomedical relationship types recognized by Samhita v1."""

    TARGETS = "targets"
    INHIBITS = "inhibits"
    ACTIVATES = "activates"
    TREATS = "treats"
    INDICATED_FOR = "indicated_for"
    ASSOCIATED_WITH = "associated_with"
    CAUSES = "causes"
    PART_OF = "part_of"
    REGULATES = "regulates"
    INTERACTS_WITH = "interacts_with"


class SectionType(str, Enum):
    """PubMed Central paper sections detected by PMCGrab."""

    ABSTRACT = "abstract"
    INTRODUCTION = "introduction"
    METHODS = "methods"
    RESULTS = "results"
    DISCUSSION = "discussion"
    UNKNOWN = "unknown"

    @classmethod
    def from_alias(cls, name: str) -> "SectionType":
        """Map a PMCGrab section label (or alias) to a SectionType."""
        key = (name or "").strip().lower()
        return _SECTION_ALIASES.get(key, cls.UNKNOWN)


_SECTION_ALIASES: dict[str, SectionType] = {
    "abstract": SectionType.ABSTRACT,
    "introduction": SectionType.INTRODUCTION,
    "intro": SectionType.INTRODUCTION,
    "background": SectionType.INTRODUCTION,
    "methods": SectionType.METHODS,
    "method": SectionType.METHODS,
    "materials and methods": SectionType.METHODS,
    "methodology": SectionType.METHODS,
    "results": SectionType.RESULTS,
    "result": SectionType.RESULTS,
    "findings": SectionType.RESULTS,
    "discussion": SectionType.DISCUSSION,
    "conclusion": SectionType.DISCUSSION,
    "conclusions": SectionType.DISCUSSION,
}


class ModelTier(str, Enum):
    """Which tier of model produced an output (for cost telemetry)."""

    FRONTIER = "frontier"              # Opus 4.7, GPT-5
    MID = "mid"                        # Sonnet 4.6, Haiku
    CHEAP = "cheap"                    # Kimi K2.5, Gemini Flash
    DETERMINISTIC = "deterministic"    # no LLM call — rule-based


class SourceType(str, Enum):
    PUBMED = "pubmed"
    PMC = "pmc"
    OPENTARGETS = "opentargets"
    CHEMBL = "chembl"
    DRUGBANK = "drugbank"
    OTHER = "other"


class SourceName(str, Enum):
    """V1 supported source names (as the planner references them)."""

    PUBMED_CENTRAL = "pubmed_central"
    PUBMED_ABSTRACTS = "pubmed_abstracts"
    OPENTARGETS = "opentargets"
    CHEMBL = "chembl"
    DRUGBANK = "drugbank"


class RecipeName(str, Enum):
    """V1 schema recipes."""

    DRUG_TARGET_DISEASE = "drug_target_disease"
    DISEASE_GENE_PATHWAY = "disease_gene_pathway"


class NamespaceName(str, Enum):
    """Canonical identifier namespaces used across Samhita.

    Using an enum here turns namespace typos (e.g. ``"Chembl"`` vs
    ``"ChEMBL"``) into import-time errors instead of silent ID mismatches.
    Because this is a ``str, Enum``, members pass Pydantic validation
    anywhere ``namespace: str`` is expected.
    """

    HGNC = "HGNC"
    NCBI_GENE = "NCBIGene"
    ENSEMBL = "Ensembl"
    UNIPROT = "UniProt"
    CHEMBL = "ChEMBL"
    DRUGBANK = "DrugBank"
    MONDO = "MONDO"
    EFO = "EFO"
    HP = "HP"
    DOID = "DOID"
    REACTOME = "Reactome"
    KEGG = "KEGG"
    CL = "CL"
    UBERON = "UBERON"
    LOCAL = "local"


class RunStatus(str, Enum):
    """Lifecycle status for a single Samhita build run."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class Identifier(BaseModel):
    """A namespace-qualified identifier for a biomedical entity."""

    model_config = ConfigDict(frozen=True)

    namespace: str = Field(description="e.g. 'HGNC', 'UniProt', 'ChEMBL', 'MONDO'")
    value: str

    def __str__(self) -> str:
        return f"{self.namespace}:{self.value}"


class Provenance(BaseModel):
    """Traceability for a single extracted claim or edge."""

    source_id: str = Field(description="PMID, DOI, API endpoint, dataset ID, etc.")
    source_type: SourceType
    extracting_model: str | None = Field(
        default=None,
        description="model id, e.g. 'claude-sonnet-4-6' or null if deterministic",
    )
    model_tier: ModelTier = ModelTier.DETERMINISTIC
    section: SectionType = SectionType.UNKNOWN
    evidence_span: str | None = Field(
        default=None,
        description="verbatim text snippet supporting the claim (for audit)",
    )
    extracted_at: datetime = Field(default_factory=_now)
    cost_usd: float = 0.0
    cache_hit: bool = False


class Entity(BaseModel):
    """A biomedical entity (node in the KG)."""

    entity_type: EntityType
    name: str = Field(description="canonical display name")
    primary_id: Identifier = Field(description="primary namespace ID after normalization")
    aliases: list[Identifier] = Field(default_factory=list)
    synonyms: list[str] = Field(default_factory=list)
    description: str | None = None

    @property
    def node_id(self) -> str:
        return str(self.primary_id)


class Edge(BaseModel):
    """A typed relationship between two entities with provenance."""

    relation: RelationType
    subject_id: str = Field(description="primary_id of the subject entity")
    object_id: str = Field(description="primary_id of the object entity")
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    provenance: Provenance
    properties: dict[str, str | float | int | bool] = Field(default_factory=dict)
    conflict_flag: bool = Field(
        default=False,
        description="true if another source asserts a contradictory edge",
    )
    dissenting_sources: list[str] = Field(default_factory=list)


class KGSpec(BaseModel):
    """Structured specification parsed from a natural-language request.

    The planner agent produces this; the executor consumes it.
    """

    recipe: RecipeName
    seeds: dict[str, list[str]] = Field(
        description="seed entities, e.g. {'disease': ['atopic dermatitis']}"
    )
    sources: list[SourceName]
    entity_types: list[EntityType]
    relation_types: list[RelationType]
    max_papers: int = 500
    max_depth: int = Field(default=2, description="graph traversal depth from seeds")
    original_request: str = Field(description="verbatim user request")
    planner_notes: str | None = Field(
        default=None,
        description="assumptions the planner made for ambiguous requests",
    )


class RunState(BaseModel):
    """Canonical intermediate state any orchestrator can expose.

    Framework adapters serialize to/from this so progress reporting
    and resumability are framework-independent.
    """

    spec: KGSpec
    fetched_documents: int = 0
    extracted_entities: int = 0
    extracted_edges: int = 0
    normalized_entities: int = 0
    flagged_conflicts: int = 0
    total_cost_usd: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    cached_input_tokens: int = 0
    errors: list[str] = Field(default_factory=list)
    status: RunStatus = RunStatus.PENDING
    output_path: str | None = Field(
        default=None,
        description="filesystem path of the written KG artifact (set by the write node)",
    )

    @property
    def cache_hit_rate(self) -> float:
        """Fraction of input tokens that hit the provider's prompt cache."""
        if self.total_input_tokens <= 0:
            return 0.0
        return self.cached_input_tokens / self.total_input_tokens


class KGResult(BaseModel):
    """Output of a completed Samhita build run."""

    spec: KGSpec
    entities: list[Entity]
    edges: list[Edge]
    state: RunState
    biocypher_output_path: str | None = None
    neo4j_uri: str | None = None
    build_duration_seconds: float = 0.0
    completed_at: datetime = Field(default_factory=_now)

    @property
    def edge_count(self) -> int:
        return len(self.edges)

    @property
    def entity_count(self) -> int:
        return len(self.entities)
