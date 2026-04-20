"""Samhita command-line interface."""

from __future__ import annotations

import typer
from rich import print as rprint

app = typer.Typer(
    name="samhita",
    help="The compendium-builder for biomedical knowledge.",
    no_args_is_help=True,
)


def _load_drivers() -> None:
    """Import driver modules to trigger registry side-effects."""
    # Each import registers itself in the orchestrator registry
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
def build(
    request: str = typer.Argument(..., help="Natural-language KG build request."),
    framework: str = typer.Option(
        "langgraph",
        "--framework",
        "-f",
        help="Orchestrator driver to use (e.g. 'langgraph', 'custom').",
    ),
) -> None:
    """Build a KG from a natural-language request (scaffolding stub)."""
    _load_drivers()
    from samhita.orchestrators.registry import get_orchestrator, list_orchestrators

    if framework not in list_orchestrators():
        rprint(
            f"[red]Unknown framework {framework!r}.[/red] "
            f"Available: {', '.join(list_orchestrators())}"
        )
        raise typer.Exit(code=1)

    rprint(
        f"[yellow]Scaffold:[/yellow] would build KG for [cyan]{request!r}[/cyan] "
        f"using [cyan]{framework}[/cyan]"
    )
    rprint("[dim]Pipeline implementation in progress — Week 1.[/dim]")
    # Touch the orchestrator factory so the registry path is exercised
    _ = get_orchestrator(framework)


if __name__ == "__main__":
    app()
