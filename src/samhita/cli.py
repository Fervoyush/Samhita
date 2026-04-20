"""Samhita command-line interface."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import typer
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(
    name="samhita",
    help="The compendium-builder for biomedical knowledge.",
    no_args_is_help=True,
)

_console = Console()


def _load_drivers() -> None:
    """Import driver modules to trigger registry side-effects."""
    import samhita.orchestrators.custom_driver  # noqa: F401
    import samhita.orchestrators.langgraph_driver  # noqa: F401


@app.command()
def version() -> None:
    """Print Samhita version."""
    from samhita import __version__

    rprint(f"[bold cyan]Samhita[/bold cyan] v{__version__}")


@app.command(name="list-orchestrators")
def list_orchestrators_cmd() -> None:
    """List registered agent-orchestrator drivers."""
    _load_drivers()
    from samhita.orchestrators.registry import list_orchestrators

    rprint("[bold]Registered orchestrators:[/bold]")
    for name in list_orchestrators():
        rprint(f"  - {name}")


@app.command(name="list-recipes")
def list_recipes_cmd() -> None:
    """List v1 schema recipes."""
    from samhita.core.recipes import list_recipes

    rprint("[bold]V1 recipes:[/bold]")
    for recipe in list_recipes():
        rprint(f"  - [cyan]{recipe['name'].value}[/cyan]: {recipe['description']}")


@app.command()
def plan(
    request: str = typer.Argument(..., help="Natural-language KG build request."),
    framework: str = typer.Option(
        "langgraph", "--framework", "-f", help="Orchestrator driver to use."
    ),
    provider: str = typer.Option(
        "anthropic", "--provider", "-p", help="LLM provider for the planner."
    ),
    model: str = typer.Option(
        "claude-sonnet-4-6", "--model", "-m", help="Model for the planner."
    ),
) -> None:
    """Parse an NL request into a KGSpec and print it — no fetch / extract."""
    _check_secrets(provider)
    orch = _build_orchestrator(framework, provider, model)
    try:
        spec = asyncio.run(orch.plan(request))
    except Exception as exc:  # noqa: BLE001
        _console.print(f"[red]Planner failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    _console.print(
        Panel.fit(
            f"[bold]Recipe:[/bold] {spec.recipe.value}\n"
            f"[bold]Sources:[/bold] {', '.join(s.value for s in spec.sources)}\n"
            f"[bold]Entity types:[/bold] {', '.join(e.value for e in spec.entity_types)}\n"
            f"[bold]Relation types:[/bold] {', '.join(r.value for r in spec.relation_types)}\n"
            f"[bold]Max papers:[/bold] {spec.max_papers}\n"
            f"[bold]Seeds:[/bold] {json.dumps(spec.seeds, indent=2)}\n"
            f"[bold]Planner notes:[/bold] {spec.planner_notes or '-'}",
            title="Planned KGSpec",
            border_style="cyan",
        )
    )


@app.command()
def build(
    request: str = typer.Argument(..., help="Natural-language KG build request."),
    framework: str = typer.Option(
        "langgraph", "--framework", "-f", help="Orchestrator driver to use."
    ),
    provider: str = typer.Option(
        "anthropic", "--provider", "-p", help="LLM provider."
    ),
    model: str = typer.Option(
        "claude-sonnet-4-6", "--model", "-m", help="LLM model id."
    ),
    max_papers: int = typer.Option(
        -1,
        "--max-papers",
        help="Cap literature-source paper count (-1 = let the planner decide).",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Plan only, do not fetch / extract / write."
    ),
) -> None:
    """Run an end-to-end NL → KG build."""
    _check_secrets(provider)
    orch = _build_orchestrator(framework, provider, model)

    _console.print(f"[cyan]Planning[/cyan] via {provider}:{model} …")
    try:
        spec = asyncio.run(orch.plan(request))
    except Exception as exc:  # noqa: BLE001
        _console.print(f"[red]Planner failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    if max_papers > 0:
        spec = spec.model_copy(update={"max_papers": max_papers})

    _console.print(
        f"[green]Planned[/green] recipe=[bold]{spec.recipe.value}[/bold] · "
        f"{len(spec.sources)} sources · "
        f"{sum(len(v) for v in spec.seeds.values())} seeds · "
        f"max_papers={spec.max_papers}"
    )

    if dry_run:
        _console.print("[yellow]--dry-run[/yellow]: skipping execute.")
        return

    _console.print("[cyan]Executing[/cyan] pipeline …")
    try:
        result = asyncio.run(orch.execute(spec))
    except Exception as exc:  # noqa: BLE001
        _console.print(f"[red]Execution failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    _render_summary(result)


def _build_orchestrator(framework: str, provider: str, model: str):  # noqa: ANN202
    _load_drivers()
    from samhita.orchestrators.registry import get_orchestrator, list_orchestrators

    if framework not in list_orchestrators():
        _console.print(
            f"[red]Unknown framework {framework!r}.[/red] "
            f"Available: {', '.join(list_orchestrators())}"
        )
        raise typer.Exit(code=1)

    try:
        return get_orchestrator(framework, llm_provider=provider, llm_model=model)
    except TypeError:
        # Fallback for drivers that take no kwargs yet (e.g. CustomOrchestrator)
        return get_orchestrator(framework)


def _check_secrets(provider: str) -> None:
    if provider == "anthropic" and not os.getenv("ANTHROPIC_API_KEY"):
        _console.print(
            "[red]ANTHROPIC_API_KEY is not set.[/red] "
            "Export it before running plan / build, or pass --provider to use a different LLM."
        )
        raise typer.Exit(code=2)


def _render_summary(result) -> None:  # noqa: ANN001
    state = result.state

    table = Table(title="Run summary", show_header=False, border_style="cyan")
    table.add_row("Status", state.status.value)
    table.add_row("Duration (s)", f"{result.build_duration_seconds:.1f}")
    table.add_row("Fetched documents", str(state.fetched_documents))
    table.add_row("Extracted entities (raw)", str(state.extracted_entities))
    table.add_row("Extracted edges (raw)", str(state.extracted_edges))
    table.add_row("Normalized entities", str(state.normalized_entities))
    table.add_row("Final edges", str(len(result.edges)))
    table.add_row("Flagged conflicts", str(state.flagged_conflicts))
    table.add_row("Total cost (USD)", f"${state.total_cost_usd:.4f}")
    if state.cache_hit_rate:
        table.add_row("Cache hit rate", f"{state.cache_hit_rate:.1%}")
    _console.print(table)

    if state.errors:
        tail = state.errors[-6:]
        _console.print(f"[bold]Log (last {len(tail)}):[/bold]")
        for msg in tail:
            _console.print(f"  · {msg}")

    if state.output_path:
        _console.print(
            f"[green]Output:[/green] {state.output_path} "
            f"(open nodes.csv / edges.csv / schema.json or the Biocypher import files)"
        )


if __name__ == "__main__":
    app()
