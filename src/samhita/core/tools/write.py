"""Biocypher output tool.

Converts Samhita `Entity` / `Edge` objects into Biocypher-compatible
node/edge tuples and writes them to disk. If the `biocypher` package is
not installed, the tool falls back to a simple CSV + JSON output that
Biocypher can later ingest.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from samhita.core.schemas import Edge, Entity
from samhita.core.tools import Tool, register_tools


class BiocypherWriteInput(BaseModel):
    entities: list[Entity]
    edges: list[Edge]
    output_dir: str = Field(default="biocypher-out")
    neo4j_uri: str | None = Field(
        default=None,
        description="If provided, attempt to populate Neo4j as well.",
    )


class BiocypherWriteOutput(BaseModel):
    output_dir: str
    nodes_written: int = 0
    edges_written: int = 0
    neo4j_loaded: bool = False
    backend: str = Field(
        default="csv",
        description="'biocypher' if the library was used, otherwise 'csv'.",
    )
    node_files: list[str] = Field(default_factory=list)
    edge_files: list[str] = Field(default_factory=list)
    error: str | None = None


async def write_biocypher(payload: BiocypherWriteInput) -> BiocypherWriteOutput:
    """Write nodes + edges to disk.

    Prefers Biocypher *only* when a schema config is available — Biocypher
    logs errors but does not raise when schema types are missing, so
    running it unconfigured leaves silently-empty output. The CSV fallback
    is deterministic and always produces inspectable output.
    """
    out_dir = Path(payload.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    schema_path = _resolve_biocypher_schema()
    if schema_path is not None:
        try:
            return _write_via_biocypher(payload, out_dir, schema_path)
        except Exception as exc:  # noqa: BLE001
            csv_out = _write_via_csv(payload, out_dir)
            csv_out.error = f"biocypher path failed ({exc}); wrote CSV fallback"
            return csv_out

    return _write_via_csv(payload, out_dir)


def _resolve_biocypher_schema() -> Path | None:
    """Return a schema config path if Biocypher + a schema are both available.

    Precedence:
      1. ``$SAMHITA_BIOCYPHER_SCHEMA`` env var
      2. ``config/biocypher_schema_config.yaml`` in the current working dir

    Returns ``None`` (forcing the CSV fallback) if Biocypher is not
    installed or no schema config is found. Running Biocypher without a
    schema config yields silent-error output, so we refuse to go there.
    """
    try:
        import biocypher  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        return None

    env = os.getenv("SAMHITA_BIOCYPHER_SCHEMA", "").strip()
    if env and Path(env).exists():
        return Path(env)

    default = Path.cwd() / "config" / "biocypher_schema_config.yaml"
    if default.exists():
        return default

    return None


def _write_via_biocypher(
    payload: BiocypherWriteInput, out_dir: Path, schema_path: Path
) -> BiocypherWriteOutput:
    """Use Biocypher to emit Neo4j-admin-import-compatible files."""
    from biocypher import BioCypher  # type: ignore[import-not-found]

    bc = BioCypher(
        output_directory=str(out_dir),
        schema_config_path=str(schema_path),
    )

    node_tuples: list[tuple[str, str, dict[str, Any]]] = []
    for entity in payload.entities:
        props: dict[str, Any] = {"name": entity.name}
        if entity.description:
            props["description"] = entity.description
        if entity.synonyms:
            props["synonyms"] = list(entity.synonyms)
        node_tuples.append((entity.node_id, entity.entity_type.value, props))

    edge_tuples: list[tuple[str | None, str, str, str, dict[str, Any]]] = []
    for edge in payload.edges:
        props = {
            "confidence": edge.confidence,
            "source_id": edge.provenance.source_id,
            "source_type": edge.provenance.source_type.value,
            "section": edge.provenance.section.value,
            "extracting_model": edge.provenance.extracting_model or "",
            "conflict_flag": edge.conflict_flag,
        }
        props.update(edge.properties)
        edge_tuples.append(
            (None, edge.subject_id, edge.object_id, edge.relation.value, props)
        )

    bc.add(node_tuples)
    bc.add(edge_tuples)
    bc.write_import_call()

    # Biocypher writes into `output_directory/biocypher-<timestamp>`; collect
    # whatever CSVs are present.
    node_files = sorted(str(p) for p in out_dir.rglob("*-part000.csv"))
    edge_files = sorted(str(p) for p in out_dir.rglob("*-part000.tsv"))

    return BiocypherWriteOutput(
        output_dir=str(out_dir),
        nodes_written=len(node_tuples),
        edges_written=len(edge_tuples),
        backend="biocypher",
        node_files=node_files,
        edge_files=edge_files,
    )


def _write_via_csv(
    payload: BiocypherWriteInput, out_dir: Path
) -> BiocypherWriteOutput:
    """Fallback: write nodes.csv + edges.csv + schema.json when Biocypher is unavailable."""
    nodes_path = out_dir / "nodes.csv"
    edges_path = out_dir / "edges.csv"
    schema_path = out_dir / "schema.json"

    with nodes_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["id", "type", "name", "description"])
        for entity in payload.entities:
            writer.writerow(
                [
                    entity.node_id,
                    entity.entity_type.value,
                    entity.name,
                    entity.description or "",
                ]
            )

    with edges_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "subject",
                "object",
                "relation",
                "confidence",
                "source_id",
                "section",
                "model",
                "conflict_flag",
            ]
        )
        for edge in payload.edges:
            writer.writerow(
                [
                    edge.subject_id,
                    edge.object_id,
                    edge.relation.value,
                    edge.confidence,
                    edge.provenance.source_id,
                    edge.provenance.section.value,
                    edge.provenance.extracting_model or "",
                    edge.conflict_flag,
                ]
            )

    schema = {
        "entity_types": sorted({e.entity_type.value for e in payload.entities}),
        "relation_types": sorted({e.relation.value for e in payload.edges}),
        "node_count": len(payload.entities),
        "edge_count": len(payload.edges),
    }
    schema_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")

    return BiocypherWriteOutput(
        output_dir=str(out_dir),
        nodes_written=len(payload.entities),
        edges_written=len(payload.edges),
        backend="csv",
        node_files=[str(nodes_path)],
        edge_files=[str(edges_path)],
    )


def register_write_tools() -> None:
    # Tools built per call so monkeypatched `write_biocypher` flows through.
    register_tools(
        [
            Tool(
                name="write_biocypher",
                description=(
                    "Write a set of entities and edges to disk via Biocypher "
                    "(preferred) or a CSV + schema.json fallback."
                ),
                input_schema=BiocypherWriteInput,
                output_schema=BiocypherWriteOutput,
                func=write_biocypher,
                tags=["write", "biocypher"],
            ),
        ]
    )
