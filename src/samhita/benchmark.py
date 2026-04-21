"""Cross-provider benchmark for Samhita builds.

Runs the same :class:`KGSpec` through multiple ``(provider, model)``
pairs and emits a comparison table + a reusable JSON report.

Powers ``samhita benchmark``. The primary use case is the Phase-3
headline comparison (Anthropic Sonnet vs. Moonshot Kimi K2.5) on a
fixed corpus, but the command is n-way.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from samhita.core.schemas import Edge, Entity, KGResult, KGSpec
from samhita.orchestrators.registry import get_orchestrator


@dataclass
class BenchmarkConfig:
    """One (provider, model) pair to benchmark."""

    provider: str
    model: str
    framework: str = "langgraph"

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.model}"


@dataclass
class RunMetrics:
    """Per-provider metrics for a single benchmark run."""

    label: str
    status: str
    duration_s: float
    total_cost_usd: float
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int
    cache_hit_rate: float
    entity_count: int
    edge_count: int
    flagged_conflicts: int
    errors: list[str] = field(default_factory=list)


@dataclass
class OverlapMetrics:
    """Pairwise Jaccard overlap between two runs' outputs."""

    a: str
    b: str
    entity_jaccard: float
    edge_jaccard: float
    shared_entities: int
    shared_edges: int
    total_entities: int
    total_edges: int


@dataclass
class BenchmarkReport:
    request: str
    spec: dict[str, Any]
    runs: list[RunMetrics]
    overlaps: list[OverlapMetrics]
    started_at: str
    completed_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "request": self.request,
            "spec": self.spec,
            "runs": [run.__dict__ for run in self.runs],
            "overlaps": [o.__dict__ for o in self.overlaps],
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }


def parse_provider_spec(value: str) -> BenchmarkConfig:
    """Parse a ``provider:model`` string into a :class:`BenchmarkConfig`."""
    if ":" not in value:
        raise ValueError(
            f"Expected 'provider:model', got {value!r} (example: 'moonshot:kimi-k2.5')"
        )
    provider, _, model = value.partition(":")
    return BenchmarkConfig(provider=provider.strip(), model=model.strip())


async def run_benchmark(
    request: str,
    configs: list[BenchmarkConfig],
    *,
    max_papers: int | None = None,
    plan_with: BenchmarkConfig | None = None,
    parallel: bool = False,
) -> BenchmarkReport:
    """Plan once, execute once per config, aggregate metrics.

    ``plan_with`` picks which provider runs the planner; if omitted,
    the first entry in ``configs`` does it. Every config then executes
    the same :class:`KGSpec` so output deltas reflect the LLM alone,
    not the planner.

    Executions run **sequentially** by default — the upstream fetch
    tools (NCBI E-utilities in particular) rate-limit aggressively,
    and concurrent provider runs compete for that quota and distort
    results. Pass ``parallel=True`` only for speed-testing on corpora
    that don't hit rate-limited providers.
    """
    if not configs:
        raise ValueError("benchmark requires at least one (provider, model)")

    planner_config = plan_with or configs[0]
    planner = _build_orchestrator(planner_config)
    started_at = datetime.now(timezone.utc).isoformat()
    spec = await planner.plan(request)
    if max_papers is not None and max_papers > 0:
        spec = spec.model_copy(update={"max_papers": max_papers})

    results: list[KGResult | Exception]
    if parallel:
        results = await asyncio.gather(
            *(_execute_one(cfg, spec) for cfg in configs),
            return_exceptions=True,
        )
    else:
        results = []
        for cfg in configs:
            try:
                results.append(await _execute_one(cfg, spec))
            except Exception as exc:  # noqa: BLE001
                results.append(exc)

    runs: list[RunMetrics] = []
    succeeded: list[tuple[BenchmarkConfig, KGResult]] = []
    for cfg, outcome in zip(configs, results, strict=True):
        if isinstance(outcome, Exception):
            runs.append(
                RunMetrics(
                    label=cfg.label,
                    status="failed",
                    duration_s=0.0,
                    total_cost_usd=0.0,
                    input_tokens=0,
                    output_tokens=0,
                    cached_input_tokens=0,
                    cache_hit_rate=0.0,
                    entity_count=0,
                    edge_count=0,
                    flagged_conflicts=0,
                    errors=[str(outcome)],
                )
            )
            continue
        runs.append(_metrics_from(cfg, outcome))
        succeeded.append((cfg, outcome))

    overlaps = _pairwise_overlaps(succeeded)
    completed_at = datetime.now(timezone.utc).isoformat()

    return BenchmarkReport(
        request=request,
        spec=spec.model_dump(mode="json"),
        runs=runs,
        overlaps=overlaps,
        started_at=started_at,
        completed_at=completed_at,
    )


async def _execute_one(cfg: BenchmarkConfig, spec: KGSpec) -> KGResult:
    orch = _build_orchestrator(cfg)
    start = time.monotonic()
    result = await orch.execute(spec)
    # In case the orchestrator didn't set it:
    if result.build_duration_seconds <= 0:
        result.build_duration_seconds = time.monotonic() - start
    return result


def _build_orchestrator(cfg: BenchmarkConfig):  # noqa: ANN202
    return get_orchestrator(
        cfg.framework, llm_provider=cfg.provider, llm_model=cfg.model
    )


def _metrics_from(cfg: BenchmarkConfig, result: KGResult) -> RunMetrics:
    state = result.state
    return RunMetrics(
        label=cfg.label,
        status=state.status.value,
        duration_s=result.build_duration_seconds,
        total_cost_usd=state.total_cost_usd,
        input_tokens=state.total_input_tokens,
        output_tokens=state.total_output_tokens,
        cached_input_tokens=state.cached_input_tokens,
        cache_hit_rate=state.cache_hit_rate,
        entity_count=len(result.entities),
        edge_count=len(result.edges),
        flagged_conflicts=state.flagged_conflicts,
        errors=list(state.errors[-4:]),
    )


def _pairwise_overlaps(
    succeeded: list[tuple[BenchmarkConfig, KGResult]],
) -> list[OverlapMetrics]:
    overlaps: list[OverlapMetrics] = []
    for i, (cfg_a, res_a) in enumerate(succeeded):
        entity_ids_a = _entity_ids(res_a.entities)
        edge_keys_a = _edge_keys(res_a.edges)
        for cfg_b, res_b in succeeded[i + 1 :]:
            entity_ids_b = _entity_ids(res_b.entities)
            edge_keys_b = _edge_keys(res_b.edges)
            overlaps.append(
                OverlapMetrics(
                    a=cfg_a.label,
                    b=cfg_b.label,
                    entity_jaccard=_jaccard(entity_ids_a, entity_ids_b),
                    edge_jaccard=_jaccard(edge_keys_a, edge_keys_b),
                    shared_entities=len(entity_ids_a & entity_ids_b),
                    shared_edges=len(edge_keys_a & edge_keys_b),
                    total_entities=len(entity_ids_a | entity_ids_b),
                    total_edges=len(edge_keys_a | edge_keys_b),
                )
            )
    return overlaps


def _entity_ids(entities: list[Entity]) -> set[str]:
    return {e.node_id for e in entities}


def _edge_keys(edges: list[Edge]) -> set[tuple[str, str, str]]:
    return {(e.subject_id, e.relation.value, e.object_id) for e in edges}


def _jaccard(a: set, b: set) -> float:
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def save_report(report: BenchmarkReport, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2, default=str))
    return path
