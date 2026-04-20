"""Biocypher/CSV writer smoke tests — no real Biocypher dependency."""

from __future__ import annotations

from pathlib import Path

import pytest

from samhita.core.bootstrap import bootstrap_tools
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
from samhita.core.tools import all_tools, clear_registry
from samhita.core.tools.write import (
    BiocypherWriteInput,
    BiocypherWriteOutput,
    write_biocypher,
)


@pytest.fixture(autouse=True)
def _fresh_registry() -> None:
    clear_registry()
    bootstrap_tools()


def _sample_entities() -> list[Entity]:
    return [
        Entity(
            entity_type=EntityType.DRUG,
            name="imatinib",
            primary_id=Identifier(namespace="ChEMBL", value="CHEMBL941"),
        ),
        Entity(
            entity_type=EntityType.GENE,
            name="ABL1",
            primary_id=Identifier(namespace="HGNC", value="76"),
        ),
    ]


def _sample_edges() -> list[Edge]:
    return [
        Edge(
            relation=RelationType.TARGETS,
            subject_id="ChEMBL:CHEMBL941",
            object_id="HGNC:76",
            confidence=0.95,
            provenance=Provenance(
                source_id="PMID:12345",
                source_type=SourceType.PUBMED,
                section=SectionType.RESULTS,
            ),
        )
    ]


def test_write_tool_registered() -> None:
    assert "write_biocypher" in all_tools()


async def test_csv_fallback_writes_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Force biocypher ImportError path to exercise the CSV fallback
    import builtins

    real_import = builtins.__import__

    def raising_import(name: str, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name == "biocypher":
            raise ImportError("pretend unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", raising_import)

    out = await write_biocypher(
        BiocypherWriteInput(
            entities=_sample_entities(),
            edges=_sample_edges(),
            output_dir=str(tmp_path),
        )
    )
    assert isinstance(out, BiocypherWriteOutput)
    assert out.backend == "csv"
    assert out.nodes_written == 2
    assert out.edges_written == 1

    nodes_csv = tmp_path / "nodes.csv"
    edges_csv = tmp_path / "edges.csv"
    schema_json = tmp_path / "schema.json"
    assert nodes_csv.exists()
    assert edges_csv.exists()
    assert schema_json.exists()

    assert "ChEMBL:CHEMBL941" in nodes_csv.read_text()
    assert "targets" in edges_csv.read_text()
