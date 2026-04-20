"""One-shot registration helpers.

`bootstrap_tools()` imports every tool implementation module and
registers it in the central registry. Orchestrator drivers call this
during `__init__` so the registry is populated before any plan/execute
runs. Both bootstraps are idempotent and guarded so repeated calls
(e.g. CLI plan + build sharing one process) don't redo the work.
"""

from __future__ import annotations

_BOOTSTRAPPED_TOOLS = False
_BOOTSTRAPPED_LLM = False


def bootstrap_tools(*, force: bool = False) -> None:
    """Register every built-in fetch / normalize / write tool.

    Extraction is LLM-dependent and therefore registered separately by
    orchestrator drivers via :func:`register_extract_tools(llm)`; see
    :mod:`samhita.core.tools.extract`.
    """
    global _BOOTSTRAPPED_TOOLS
    if _BOOTSTRAPPED_TOOLS and not force:
        return
    from samhita.core.tools.fetch import register_fetch_tools
    from samhita.core.tools.normalize import register_normalize_tools
    from samhita.core.tools.write import register_write_tools

    register_fetch_tools()
    register_normalize_tools()
    register_write_tools()
    _BOOTSTRAPPED_TOOLS = True


def bootstrap_llm_providers(*, force: bool = False) -> None:
    """Import provider modules so their factories self-register."""
    global _BOOTSTRAPPED_LLM
    if _BOOTSTRAPPED_LLM and not force:
        return
    import samhita.core.llm_clients.anthropic  # noqa: F401
    _BOOTSTRAPPED_LLM = True


def _reset_bootstrap_flags() -> None:
    """Test helper: force re-bootstrap on the next call."""
    global _BOOTSTRAPPED_TOOLS, _BOOTSTRAPPED_LLM
    _BOOTSTRAPPED_TOOLS = False
    _BOOTSTRAPPED_LLM = False
