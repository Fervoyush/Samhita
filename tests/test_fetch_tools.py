"""Fetch tool smoke tests — no network I/O.

Network calls are monkey-patched; we verify that:
- Tools are registered after bootstrap
- Each tool returns its declared output schema
- Errors from the network are surfaced in the typed `error` field
  rather than raised as exceptions.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from samhita.core.bootstrap import bootstrap_tools
from samhita.core.tools import all_tools, clear_registry
from samhita.core.tools.fetch import (
    ChEMBLQueryInput,
    ChEMBLQueryOutput,
    OpenTargetsQueryInput,
    OpenTargetsQueryOutput,
    PMCFetchInput,
    PMCFetchOutput,
    PubMedAbstractInput,
    PubMedAbstractOutput,
    PubMedSearchInput,
    PubMedSearchOutput,
    fetch_pmc_paper,
    fetch_pubmed_abstract,
    query_chembl,
    query_opentargets,
    search_pubmed,
)


@pytest.fixture(autouse=True)
def _fresh_registry() -> None:
    clear_registry()
    bootstrap_tools()


def test_bootstrap_registers_fetch_tools() -> None:
    names = {t.name for t in all_tools().values()}
    expected = {
        "fetch_pmc_paper",
        "search_pubmed",
        "fetch_pubmed_abstract",
        "query_opentargets",
        "query_chembl",
        "lookup_drugbank",
    }
    assert expected.issubset(names)


async def test_pmc_without_library_returns_typed_error(monkeypatch: Any) -> None:
    # Force ImportError path
    import builtins

    real_import = builtins.__import__

    def raising_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "pmcgrab":
            raise ImportError("pretend not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", raising_import)
    out = await fetch_pmc_paper(PMCFetchInput(pmc_id="PMC1234567"))
    assert isinstance(out, PMCFetchOutput)
    assert out.error is not None
    assert "pmcgrab" in out.error.lower()


class _MockResponse:
    def __init__(self, *, json_body: Any = None, text: str = "", status_code: int = 200) -> None:
        self._json = json_body
        self.text = text
        self.status_code = status_code

    def json(self) -> Any:
        return self._json

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("error", request=None, response=None)  # type: ignore[arg-type]


class _MockAsyncClient:
    def __init__(self, *, get_response: _MockResponse | None = None, post_response: _MockResponse | None = None, **_: Any) -> None:
        self._get_response = get_response
        self._post_response = post_response

    async def __aenter__(self) -> "_MockAsyncClient":
        return self

    async def __aexit__(self, *_: Any) -> None:
        return None

    async def get(self, *_: Any, **__: Any) -> _MockResponse:
        assert self._get_response is not None
        return self._get_response

    async def post(self, *_: Any, **__: Any) -> _MockResponse:
        assert self._post_response is not None
        return self._post_response


async def test_search_pubmed_parses_esearch_body(monkeypatch: Any) -> None:
    body = {"esearchresult": {"idlist": ["111", "222"], "count": "2"}}

    def _factory(**kwargs: Any) -> _MockAsyncClient:
        return _MockAsyncClient(get_response=_MockResponse(json_body=body))

    monkeypatch.setattr(httpx, "AsyncClient", _factory)
    out = await search_pubmed(PubMedSearchInput(query="atopic dermatitis"))
    assert isinstance(out, PubMedSearchOutput)
    assert out.pmids == ["111", "222"]
    assert out.total_hits == 2
    assert out.error is None


async def test_fetch_pubmed_abstract_parses_xml(monkeypatch: Any) -> None:
    xml = (
        "<PubmedArticleSet><PubmedArticle><MedlineCitation>"
        "<Article><ArticleTitle>Test Title</ArticleTitle>"
        "<Abstract><AbstractText>Hello world.</AbstractText></Abstract>"
        "</Article></MedlineCitation>"
        "<PubmedData><ArticleIdList>"
        "<ArticleId IdType='doi'>10.1/abc</ArticleId>"
        "<ArticleId IdType='pmc'>PMC999</ArticleId>"
        "</ArticleIdList></PubmedData>"
        "</PubmedArticle></PubmedArticleSet>"
    )

    def _factory(**kwargs: Any) -> _MockAsyncClient:
        return _MockAsyncClient(get_response=_MockResponse(text=xml))

    monkeypatch.setattr(httpx, "AsyncClient", _factory)
    out = await fetch_pubmed_abstract(PubMedAbstractInput(pmid="123"))
    assert isinstance(out, PubMedAbstractOutput)
    assert out.title == "Test Title"
    assert "Hello world" in out.abstract
    assert out.doi == "10.1/abc"
    assert out.pmc_id == "PMC999"


async def test_opentargets_returns_data(monkeypatch: Any) -> None:
    body = {"data": {"target": {"id": "ENSG0"}}, "errors": []}

    def _factory(**kwargs: Any) -> _MockAsyncClient:
        return _MockAsyncClient(post_response=_MockResponse(json_body=body))

    monkeypatch.setattr(httpx, "AsyncClient", _factory)
    out = await query_opentargets(
        OpenTargetsQueryInput(gql_query="{ target(id: \"ENSG0\") { id } }")
    )
    assert isinstance(out, OpenTargetsQueryOutput)
    assert out.data == {"target": {"id": "ENSG0"}}
    assert out.error is None


async def test_chembl_returns_data(monkeypatch: Any) -> None:
    body = {"molecules": [{"molecule_chembl_id": "CHEMBL25"}]}

    def _factory(**kwargs: Any) -> _MockAsyncClient:
        return _MockAsyncClient(get_response=_MockResponse(json_body=body))

    monkeypatch.setattr(httpx, "AsyncClient", _factory)
    out = await query_chembl(
        ChEMBLQueryInput(endpoint="molecule", params={"molecule_chembl_id": "CHEMBL25"})
    )
    assert isinstance(out, ChEMBLQueryOutput)
    assert "molecules" in out.data
