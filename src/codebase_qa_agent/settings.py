"""Load configuration from environment. Secrets live only in `.env` (gitignored)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_env() -> None:
    """Load `.env` from project root if present. Safe to call multiple times."""
    env_path = PROJECT_ROOT / ".env"
    # `.env` is source of truth for local dev (wins over stale shell exports).
    load_dotenv(env_path, override=True)


@dataclass(frozen=True, slots=True)
class Settings:
    """Non-secret defaults and secret handles (values are not logged)."""

    project_root: Path
    llm_provider: str
    openai_api_key: str | None = field(repr=False)
    openai_model: str
    anthropic_api_key: str | None = field(repr=False)
    anthropic_model: str
    chroma_path: Path
    embedding_model: str
    retrieval_mode: str
    rerank_enabled: bool
    rerank_model: str

    @property
    def has_llm(self) -> bool:
        if self.llm_provider == "anthropic":
            return bool(self.anthropic_api_key)
        if self.llm_provider == "openai":
            return bool(self.openai_api_key)
        return bool(self.openai_api_key or self.anthropic_api_key)


def get_settings() -> Settings:
    load_env()
    provider = (os.getenv("LLM_PROVIDER") or "openai").strip().lower()
    return Settings(
        project_root=PROJECT_ROOT,
        llm_provider=provider,
        openai_api_key=os.getenv("OPENAI_API_KEY") or None,
        openai_model=os.getenv("OPENAI_MODEL") or "gpt-4o-mini",
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY") or None,
        anthropic_model=os.getenv("ANTHROPIC_MODEL") or "claude-sonnet-4-20250514",
        chroma_path=Path(os.getenv("CHROMA_PATH") or (PROJECT_ROOT / "data" / "chroma")),
        embedding_model=os.getenv("EMBEDDING_MODEL")
        or "sentence-transformers/all-MiniLM-L6-v2",
        retrieval_mode=(os.getenv("RETRIEVAL_MODE") or "hybrid").strip().lower(),
        rerank_enabled=(os.getenv("RERANK_ENABLED") or "false").strip().lower()
        in ("1", "true", "yes", "on"),
        rerank_model=os.getenv("RERANK_MODEL") or "cross-encoder/ms-marco-MiniLM-L-6-v2",
    )