"""One-shot registration helpers.

`bootstrap_tools()` imports every tool implementation module and
registers it in the central registry. Orchestrator drivers call this
during `__init__` so the registry is populated before any plan/execute
runs.
"""

from __future__ import annotations


def bootstrap_tools() -> None:
    """Register every built-in fetch / normalize / write tool.

    Extraction is LLM-dependent and therefore registered separately by
    orchestrator drivers via :func:`register_extract_tools(llm)`; see
    :mod:`samhita.core.tools.extract`.
    """
    from samhita.core.tools.fetch import register_fetch_tools
    from samhita.core.tools.normalize import register_normalize_tools
    from samhita.core.tools.write import register_write_tools

    register_fetch_tools()
    register_normalize_tools()
    register_write_tools()


def bootstrap_llm_providers() -> None:
    """Import provider modules so their factories self-register."""
    import samhita.core.llm_clients.anthropic  # noqa: F401
