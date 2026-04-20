"""Normalization tool smoke tests — no network."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from samhita.core.bootstrap import bootstrap_tools
from samhita.core.schemas import EntityType
from samhita.core.tools import all_tools, clear_registry
from samhita.core.tools.normalize import (
    NormalizeEntityInput,
    NormalizeEntityOutput,
    normalize_drug,
    normalize_entity,
    normalize_gene,
    normalize_via_ols,
)


@pytest.fixture(autouse=True)
def _fresh_registry() -> None:
    clear_registry()
    bootstrap_tools()


def test_normalize_tools_registered() -> None:
    names = {t.name for t in all_tools().values()}
    assert {"normalize_entity", "normalize_gene", "normalize_drug", "normalize_via_ols"} <= names


class _Resp:
    def __init__(self, payload: dict[str, Any], status: int = 200) -> None:
        self._payload = payload
        self.status_code = status

    def json(self) -> dict[str, Any]:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("err", request=None, response=None)  # type: ignore[arg-type]


class _Client:
    def __init__(self, *, response: _Resp, **_: Any) -> None:
        self._response = response

    async def __aenter__(self) -> "_Client":
        return self

    async def __aexit__(self, *_: Any) -> None:
        return None

    async def get(self, *_: Any, **__: Any) -> _Resp:
        return self._response


def _patch_async_client(monkeypatch: Any, payload: dict[str, Any]) -> None:
    def _factory(**kwargs: Any) -> _Client:
        return _Client(response=_Resp(payload))

    monkeypatch.setattr(httpx, "AsyncClient", _factory)


async def test_normalize_gene_picks_hgnc_primary(monkeypatch: Any) -> None:
    body = {
        "hits": [
            {
                "symbol": "EGFR",
                "HGNC": "3236",
                "entrezgene": 1956,
                "ensembl": {"gene": "ENSG00000146648"},
                "uniprot": {"Swiss-Prot": "P00533"},
            }
        ]
    }
    _patch_async_client(monkeypatch, body)

    out = await normalize_gene(
        NormalizeEntityInput(name="EGFR", entity_type=EntityType.GENE)
    )
    assert isinstance(out, NormalizeEntityOutput)
    assert out.primary_id is not None
    assert out.primary_id.namespace == "HGNC"
    assert out.primary_id.value == "3236"
    assert any(a.namespace == "UniProt" for a in out.aliases)


async def test_normalize_gene_no_hits(monkeypatch: Any) -> None:
    _patch_async_client(monkeypatch, {"hits": []})
    out = await normalize_gene(
        NormalizeEntityInput(name="XYZQ123", entity_type=EntityType.GENE)
    )
    assert out.method == "failed"
    assert "no hits" in (out.error or "").lower()


async def test_normalize_via_ols_parses_mondo(monkeypatch: Any) -> None:
    body = {
        "response": {
            "docs": [
                {"obo_id": "MONDO:0004980", "label": "atopic eczema"},
            ]
        }
    }
    _patch_async_client(monkeypatch, body)

    out = await normalize_via_ols(
        NormalizeEntityInput(name="atopic dermatitis", entity_type=EntityType.DISEASE)
    )
    assert out.primary_id is not None
    assert out.primary_id.namespace == "MONDO"
    assert out.primary_id.value == "0004980"


async def test_normalize_drug_chembl(monkeypatch: Any) -> None:
    body = {"molecules": [{"molecule_chembl_id": "CHEMBL941", "pref_name": "IMATINIB"}]}
    _patch_async_client(monkeypatch, body)

    out = await normalize_drug(
        NormalizeEntityInput(name="imatinib", entity_type=EntityType.DRUG)
    )
    assert out.primary_id is not None
    assert out.primary_id.namespace == "ChEMBL"
    assert out.primary_id.value == "CHEMBL941"


async def test_normalize_entity_dispatches(monkeypatch: Any) -> None:
    # Any entity type the dispatcher recognizes should route — use DISEASE.
    body = {
        "response": {
            "docs": [{"obo_id": "MONDO:0004980"}]
        }
    }
    _patch_async_client(monkeypatch, body)

    out = await normalize_entity(
        NormalizeEntityInput(name="atopic dermatitis", entity_type=EntityType.DISEASE)
    )
    assert out.primary_id is not None
    assert out.primary_id.namespace == "MONDO"


async def test_normalize_entity_rejects_unsupported_type() -> None:
    out = await normalize_entity(
        NormalizeEntityInput(name="foo", entity_type=EntityType.VARIANT)
    )
    assert out.method == "failed"
    assert "no normalizer" in (out.error or "").lower()
