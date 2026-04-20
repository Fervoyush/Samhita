"""Runtime configuration for Samhita."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field


class Settings(BaseModel):
    """Samhita runtime settings — override via env vars or CLI flags."""

    # Orchestrator / framework selection
    orchestrator: str = Field(default="langgraph")

    # LLM provider + model
    llm_provider: str = Field(
        default="anthropic", description="anthropic | openai | moonshot | local"
    )
    llm_model: str = Field(default="claude-sonnet-4-6")

    # Paths
    data_dir: Path = Field(default=Path("./data"))
    biocypher_output_dir: Path = Field(default=Path("./biocypher-out"))

    # Neo4j
    neo4j_uri: str = Field(default="bolt://localhost:7687")
    neo4j_user: str = Field(default="neo4j")
    neo4j_password: str = Field(default="")

    # Cost guardrails
    max_cost_usd_per_build: float = Field(default=10.0)


def load_settings() -> Settings:
    """Load settings from env vars with sensible defaults."""
    return Settings(
        orchestrator=os.getenv("SAMHITA_ORCHESTRATOR", "langgraph"),
        llm_provider=os.getenv("SAMHITA_LLM_PROVIDER", "anthropic"),
        llm_model=os.getenv("SAMHITA_LLM_MODEL", "claude-sonnet-4-6"),
        neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
        neo4j_password=os.getenv("NEO4J_PASSWORD", ""),
        max_cost_usd_per_build=float(
            os.getenv("SAMHITA_MAX_COST_USD", "10.0")
        ),
    )
