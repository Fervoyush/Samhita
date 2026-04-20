"""One-shot registration helpers.

`bootstrap_tools()` imports every tool implementation module and
registers it in the central registry. Orchestrator drivers call this
during `__init__` so the registry is populated before any plan/execute
runs.
"""

from __future__ import annotations


def bootstrap_tools() -> None:
    """Register every built-in fetch/extract/normalize/write tool."""
    # Fetch (implemented)
    from samhita.core.tools.fetch import register_fetch_tools

    register_fetch_tools()

    # Extract / normalize / write land in the next scaffolding steps.
    # When they gain `register_*_tools` functions, import + call here.


def bootstrap_llm_providers() -> None:
    """Import provider modules so their factories self-register."""
    import samhita.core.llm_clients.anthropic  # noqa: F401
