"""Application configuration and project paths."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_LLM_MODEL = "gemini-2.5-flash"


@dataclass(frozen=True)
class Settings:
    """Runtime settings loaded from `.env` and environment variables."""

    project_root: Path
    api_key: str
    llm_model: str

    @property
    def documents_dir(self) -> Path:
        return self.project_root / "data" / "cards"

    @property
    def output_dir(self) -> Path:
        return self.project_root / "output"

    @property
    def index_path(self) -> Path:
        return self.output_dir / "index.json"

    @property
    def trace_path(self) -> Path:
        return self.output_dir / "traces.jsonl"


def load_settings(project_root: Path | None = None) -> Settings:
    """Load settings and fail early with an actionable missing-key message."""
    root = project_root or Path(__file__).resolve().parents[2]
    load_dotenv(root / ".env")
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key or api_key == "your_gemini_api_key_here":
        raise SystemExit(
            "Missing GEMINI_API_KEY. Copy .env.example to .env and add your Gemini API key."
        )
    return Settings(root, api_key, os.getenv("GEMINI_MODEL", DEFAULT_LLM_MODEL))
