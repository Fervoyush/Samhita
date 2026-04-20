"""Shared helpers for structured-source fetchers.

Extracted from the per-source fetcher modules (opentargets / chembl /
drugbank) which were each re-implementing the same Entity / Edge /
Provenance construction.
"""

from __future__ import annotations

from typing import Any

from samhita.core.schemas import (
    Edge,
    Entity,
    EntityType,
    Identifier,
    Provenance,
    RelationType,
    SectionType,
    SourceType,
)


def make_entity(
    etype: EntityType,
    name: str,
    primary: Identifier,
) -> Entity:
    return Entity(entity_type=etype, name=name, primary_id=primary)


def merge_entity(store: dict[str, Entity], entity: Entity) -> None:
    """Insert entity by primary_id; first write wins."""
    store.setdefault(entity.node_id, entity)


def make_edge(
    *,
    subject: Entity,
    relation: RelationType,
    object_: Entity,
    source_type: SourceType,
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
            source_type=source_type,
            extracting_model=None,
            section=SectionType.UNKNOWN,
        ),
        properties=properties or {},
    )


def slugify(value: str, max_len: int = 64) -> str:
    """Turn an arbitrary label into a local: identifier value."""
    slug = "".join(c if c.isalnum() else "_" for c in value.lower()).strip("_")
    return slug[:max_len]
