"""Biocypher output tools — schemas and stubs.

Translates Samhita's `Entity` and `Edge` types into Biocypher-compatible
node/edge tuples and drives writing to Neo4j. Implementations land in
Week 1 Day 5.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from samhita.core.schemas import Edge, Entity


class BiocypherWriteInput(BaseModel):
    entities: list[Entity]
    edges: list[Edge]
    output_dir: str = Field(default="biocypher-out")
    neo4j_uri: str = Field(default="bolt://localhost:7687")
    neo4j_user: str = Field(default="neo4j")


class BiocypherWriteOutput(BaseModel):
    output_dir: str
    nodes_written: int
    edges_written: int
    neo4j_loaded: bool = False
    error: str | None = None
