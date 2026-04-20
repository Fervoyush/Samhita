# Samhita

*The compendium-builder for biomedical knowledge.*

Samhita turns a natural-language request into a queryable biomedical knowledge graph. Describe what you want — *"build me a KG of drugs and targets for atopic dermatitis"* — and Samhita orchestrates an agent over public biomedical sources (PubMed Central, OpenTargets, ChEMBL, DrugBank) to fetch, extract, normalize, and assemble a Biocypher-compatible KG in Neo4j. Then query it via typed tools.

**Status:** early development. Planning locked 2026-04-19; Phase 1 MVP in progress.

---

## What Samhita does

```
Natural-language request
         │
         ▼
┌────────────────────────────────────────┐
│  Planner agent (LangGraph)             │
│  → structured KG spec                  │
└───────────┬────────────────────────────┘
            │
            ▼
┌────────────────────────────────────────┐
│  Source orchestrator                   │
│  → PubMed Central (full text)          │
│  → OpenTargets (GraphQL)               │
│  → ChEMBL (REST)                       │
│  → DrugBank (bulk)                     │
│  → PubMed abstracts (fallback)         │
└───────────┬────────────────────────────┘
            │
            ▼
┌────────────────────────────────────────┐
│  Section-aware extractor               │
│  (Abstract/Intro/Methods/Results/Disc) │
│  → entities + relations with context   │
└───────────┬────────────────────────────┘
            │
            ▼
┌────────────────────────────────────────┐
│  ID normalizer + conflict flagger      │
└───────────┬────────────────────────────┘
            │
            ▼
┌────────────────────────────────────────┐
│  Biocypher writer → Neo4j              │
└───────────┬────────────────────────────┘
            │
            ▼
  Query MCP server (Phase 2)
```

## Why Samhita

- **Natural-language driven.** No adapter-writing, no Cypher, no schema authoring. Describe what you want in plain English.
- **Full-text, section-aware.** Uses [PMCGrab](https://github.com/rajdeepmondaldotcom/pmcgrab) for PMC full-text with section detection — claims in Methods are treated differently from claims in Discussion.
- **Provenance by construction.** Every edge carries source, model, timestamp, and cost.
- **Cost-optimized (Phase 3).** Planned Kimi K2.5 harness with cache-aware prompt construction — 25–40× cheaper than naive GPT/Claude extraction at equivalent quality.

## Roadmap

| Phase | Scope | Status |
|---|---|---|
| 1 | End-to-end MVP: NL → KG in Neo4j + Streamlit demo | In progress |
| 2 | Application MCP: typed query tools over the KG, registered on BioContextAI | Planned |
| 3 | **Kimi K2.5 harness**: cache-aware prompts, model routing, cost benchmark vs. Opus/Sonnet | Planned |

## V1 scope

- **Sources (5):** PubMed Central via PMCGrab, PubMed E-utilities (fallback), OpenTargets GraphQL, ChEMBL REST, DrugBank bulk
- **Recipes (2):** `drug-target-disease`, `disease-gene-pathway`
- **Extraction model:** Claude Sonnet 4.6 (swapped to Kimi K2.5 in Phase 3)
- **Backend:** Biocypher → Neo4j
- **Out of scope for v1:** PDF parsing, figure/image understanding, conflict *resolution* (flag-only), custom user schemas, incremental updates, non-English papers

## Related work

Samhita composes existing ecosystem components rather than competing with them. Honest positioning against prior art:

| Tool | Relation to Samhita |
|---|---|
| [Biocypher](https://github.com/biocypher/biocypher) | KG construction framework — used as library |
| [PMCGrab](https://github.com/rajdeepmondaldotcom/pmcgrab) | PMC full-text extraction — used as dependency |
| [BioMCP](https://github.com/genomoncology/biomcp) | MCP federation over biomedical APIs — different scope (query, not construction) |
| [Biomni](https://github.com/snap-stanford/Biomni) | General-purpose biomedical agent with 150 tools — Samhita is narrower, focused on KG construction |
| [KARMA](https://openreview.net/pdf?id=k0wyi4cOGy) | Multi-agent KG enrichment with conflict resolution — Samhita v1 flags but does not resolve conflicts |
| [Open Targets Platform](https://platform.opentargets.org/) | Pre-built target-disease scoring — Samhita builds your own query-specific KG instead |
| [PrimeKG](https://github.com/mims-harvard/PrimeKG) | Static precision medicine KG — Samhita produces fresh KGs per query |
| [KG-Orchestra](https://www.biorxiv.org/content/10.64898/2026.02.18.706536v1) | Multi-agent KG *enrichment* — Samhita does cold-start construction from NL specs |

## Install

*Coming soon — project scaffolding in progress.*

## License

Apache 2.0.
