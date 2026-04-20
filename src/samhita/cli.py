"""Samhita command-line interface.

Placeholder — commands will be wired up in Week 1 scaffolding.
"""

from __future__ import annotations

import typer
from rich import print as rprint

app = typer.Typer(
    name="samhita",
    help="The compendium-builder for biomedical knowledge.",
    no_args_is_help=True,
)


@app.command()
def version() -> None:
    """Print Samhita version."""
    from samhita import __version__

    rprint(f"[bold cyan]Samhita[/bold cyan] v{__version__}")


if __name__ == "__main__":
    app()
