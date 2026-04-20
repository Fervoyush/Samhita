"""Spec-aware fetchers for structured biomedical sources.

Low-level HTTP tools live in :mod:`samhita.core.tools.fetch` — they accept
a typed request and return raw data. The fetchers in this package sit
one level up: they take a :class:`KGSpec` (plus access to the tool
registry) and produce fully-typed :class:`Entity` / :class:`Edge`
objects with Samhita Provenance, ready to merge into the KG.

Structured sources (OpenTargets, ChEMBL, DrugBank) already ship canonical
IDs, so their output bypasses the LLM extractor and the normalizer.
"""

from samhita.core.fetchers.chembl import fetch_chembl_for_spec
from samhita.core.fetchers.drugbank import fetch_drugbank_for_spec
from samhita.core.fetchers.opentargets import fetch_opentargets_for_spec

__all__ = [
    "fetch_chembl_for_spec",
    "fetch_drugbank_for_spec",
    "fetch_opentargets_for_spec",
]
