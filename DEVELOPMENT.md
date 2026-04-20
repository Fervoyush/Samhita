# Samhita — Development Workflow

Notes for working on Samhita without burning context / tokens on
repeated full-codebase reads.

## Codebase map via graphify

[graphify](https://github.com/safishamsi/graphify) produces a compact
JSON + Markdown map of the repo. Run it once; AI coding assistants
(Claude Code, Codex, Cursor) can query the map instead of re-reading
every file on each task.

### Install (one-time)

```bash
pipx install graphifyy      # note the double y on PyPI
graphify install            # installs the Claude Code skill
```

> The package is `graphifyy`. The CLI stays `graphify`.

### Generate the map

From the repo root:

```bash
/graphify .                 # inside Claude Code / Cursor / Codex
# or
graphify build .            # plain CLI (if available in your version)
```

This writes into `graphify-out/`:

| File | Purpose |
|---|---|
| `graph.json` | Structured graph — the canonical artifact |
| `GRAPH_REPORT.md` | One-page human summary (read this first) |
| `graph.html` | Interactive browser visualization |
| `cache/` | SHA256 cache for incremental re-runs |

`graphify-out/` is gitignored. Each developer generates their own.

### Query the map from an AI assistant

Spin up graphify's MCP server pointed at the fresh graph:

```bash
python -m graphify.serve graphify-out/graph.json
```

Then wire it into your assistant's MCP config. The server exposes
`query_graph`, `get_node`, `get_neighbors`, `shortest_path` as tools,
so the assistant can answer "where does X get set?" / "what calls Y?"
with a graph lookup instead of a repo-wide grep + file read.

### Rebuild discipline

The graph is stale the moment code changes. Re-run:

- after merging a PR that touches file structure or function signatures
- before a heavy refactoring session
- when the assistant starts giving answers that feel out of date

The SHA256 cache makes re-runs cheap after the first build.

---

## Local python setup

```bash
cd ~/Documents/Samhita
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Run tests:

```bash
pytest -v
```

Run the CLI sanity checks:

```bash
samhita version
samhita list-orchestrators
samhita list-recipes
```

---

## Secrets

Set whichever API keys your planned model calls require:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export NEO4J_PASSWORD=...        # only if you plan to load into Neo4j
```

None of these are committed; `.env` is gitignored.

---

## Architecture invariant

`src/samhita/core/` never imports from `src/samhita/orchestrators/`,
and never imports any agent framework (LangGraph, Agent Zero, DSPy).
If you find yourself adding such an import in core, that's the signal
to put the code in an orchestrator driver or a tool wrapper instead.
