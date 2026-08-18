"""Central configuration for development and deployment environments.

All tuneable parameters live here. No other module should read os.environ directly.

Usage
-----
    from src.config import config
    print(config.gemini_model)
"""

import os
import logging
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Resolve local dotenv files from the project root; hosted variables take precedence.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve_app_environment() -> str:
    """Return the explicit runtime environment used to select local dotenv files."""
    configured = os.environ.get("APP_ENV", "").strip().lower()
    if configured:
        return configured
    if os.environ.get("VERCEL") or os.environ.get("VERCEL_ENV"):
        return "deployment"
    return "development"


_APP_ENV = _resolve_app_environment()
_ENV_FILE = (
    _PROJECT_ROOT / ".env.deployment"
    if _APP_ENV in {"deployment", "production", "preview"}
    else _PROJECT_ROOT / ".env.development"
)
load_dotenv(_ENV_FILE, override=False)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Category constants — used throughout the codebase
# ---------------------------------------------------------------------------

CATEGORY_TREATMENT = "treatment"
CATEGORY_PREVENTION = "prevention"
CATEGORY_NUTRITION = "nutrition"
CATEGORY_GENERAL = "general"
CATEGORY_ALL = "all"

ALL_CATEGORIES = [CATEGORY_TREATMENT, CATEGORY_PREVENTION, CATEGORY_NUTRITION]

# ---------------------------------------------------------------------------
# Document → category mapping
# Covers the 12 PDFs in data/rew_data/books/.
# Keys are partial filename matches (case-insensitive substring).
# Override via DATA_CATEGORY_OVERRIDES env var (JSON dict) if needed.
# ---------------------------------------------------------------------------

DOCUMENT_CATEGORY_MAP: dict[str, str] = {
    # Treatment
    "who recommendations on care for women": CATEGORY_TREATMENT,
    "story_case_1": CATEGORY_TREATMENT,
    "story_case_2": CATEGORY_TREATMENT,
    "availability, price and affordability": CATEGORY_TREATMENT,
    "ejos_volume 4": CATEGORY_TREATMENT,
    "who_test": CATEGORY_TREATMENT,
    # Prevention
    "an overview of diabetes mellitus in egypt": CATEGORY_PREVENTION,
    "prevention and control of diabetes mellitus": CATEGORY_PREVENTION,
    "guidance on global monitoring for diabetes prevention": CATEGORY_PREVENTION,
    "the global diabetes compact": CATEGORY_PREVENTION,
    "regional commemoration of world diabetes day": CATEGORY_PREVENTION,
    # General (will be split across all collections by section at chunk level)
    "idf_diabetes_atlas": CATEGORY_GENERAL,
}


@dataclass
class AppConfig:
    """Immutable application configuration loaded from environment variables."""

    # ── Gemini ─────────────────────────────────────────────────────────────
    gemini_api_key: str = field(default_factory=lambda: os.environ.get("GEMINI_API_KEY", ""))
    gemini_model: str = field(default_factory=lambda: os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"))
    generation_provider: str = field(
        default_factory=lambda: os.environ.get("GENERATION_PROVIDER", "gemini").lower()
    )
    ai_gateway_model: str = field(
        default_factory=lambda: os.environ.get("AI_GATEWAY_MODEL", "google/gemini-2.5-flash")
    )
    ai_gateway_api_key: str = field(
        default_factory=lambda: os.environ.get("AI_GATEWAY_API_KEY", "")
    )
    vercel_oidc_token: str = field(
        default_factory=lambda: os.environ.get("VERCEL_OIDC_TOKEN", "")
    )

    # ── Embedding ──────────────────────────────────────────────────────────
    embedding_provider: str = field(
        default_factory=lambda: os.environ.get("EMBEDDING_PROVIDER", "local").lower()
    )
    embedding_model: str = field(
        default_factory=lambda: os.environ.get(
            "EMBEDDING_MODEL",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        )
    )
    online_embedding_model: str = field(
        default_factory=lambda: os.environ.get(
            "ONLINE_EMBEDDING_MODEL",
            "gemini-embedding-2",
        )
    )
    embedding_dimension: int = field(
        default_factory=lambda: int(os.environ.get("EMBEDDING_DIMENSION", "384"))
    )
    embedding_namespace: str = field(
        default_factory=lambda: os.environ.get("EMBEDDING_NAMESPACE", "")
    )
    online_embedding_batch_size: int = field(
        default_factory=lambda: int(os.environ.get("ONLINE_EMBEDDING_BATCH_SIZE", "16"))
    )
    online_embedding_rpm: int = field(
        default_factory=lambda: int(os.environ.get("ONLINE_EMBEDDING_RPM", "90"))
    )

    # ── Retrieval ──────────────────────────────────────────────────────────
    top_k: int = field(default_factory=lambda: int(os.environ.get("TOP_K", "5")))
    similarity_threshold: float = field(
        default_factory=lambda: float(os.environ.get("SIMILARITY_THRESHOLD", "0.30"))
    )

    # ── Chunking ───────────────────────────────────────────────────────────
    chunk_size: int = field(default_factory=lambda: int(os.environ.get("CHUNK_SIZE", "2000")))
    chunk_overlap: int = field(default_factory=lambda: int(os.environ.get("CHUNK_OVERLAP", "200")))
    min_chunk_size: int = field(default_factory=lambda: int(os.environ.get("MIN_CHUNK_SIZE", "100")))
    min_quality_score: float = field(
        default_factory=lambda: float(os.environ.get("MIN_QUALITY_SCORE", "0.1"))
    )
    ocr_language: str = field(default_factory=lambda: os.environ.get("OCR_LANGUAGE", "eng"))
    ocr_dpi: int = field(default_factory=lambda: int(os.environ.get("OCR_DPI", "150")))

    # ── Data / Storage ─────────────────────────────────────────────────────
    data_dir: Path = field(
        default_factory=lambda: _PROJECT_ROOT / os.environ.get("DATA_DIR", "data/rew_data/books")
    )
    database_url: str = field(
        default_factory=lambda: os.environ.get(
            "DATABASE_URL",
            "postgresql://creativa:creativa-local@localhost:5432/creativa_diabetes",
        )
    )
    database_url_unpooled: str = field(
        default_factory=lambda: os.environ.get("DATABASE_URL_UNPOOLED", "")
    )

    # ── Application ────────────────────────────────────────────────────────
    debug: bool = field(default_factory=lambda: os.environ.get("DEBUG", "false").lower() == "true")
    app_env: str = field(default_factory=lambda: _APP_ENV)
    auto_create_schema: bool = field(
        default_factory=lambda: os.environ.get(
            "AUTO_CREATE_SCHEMA",
            "false" if _APP_ENV in {"deployment", "production", "preview"} else "true",
        ).lower() == "true"
    )
    max_memory_turns: int = field(
        default_factory=lambda: int(os.environ.get("MAX_MEMORY_TURNS", "6"))
    )

    # ── Derived ────────────────────────────────────────────────────────────
    project_root: Path = field(default_factory=lambda: _PROJECT_ROOT)

    def validate(self) -> None:
        """Log warnings for missing or suspicious configuration values."""
        if self.embedding_provider == "gemini" and not self.gemini_api_key:
            logger.warning(
                "GEMINI_API_KEY is not set. Hosted embeddings will fail. "
                "Configure the key in .env.development or the deployment environment."
            )
        if self.generation_provider == "gemini" and not self.gemini_api_key:
            logger.warning("Direct Gemini generation is selected without GEMINI_API_KEY")
        if self.generation_provider == "vercel_gateway" and not self.generation_configured:
            logger.warning("AI Gateway credentials are unavailable outside the Vercel runtime")
        if self.embedding_provider not in {"local", "gemini"}:
            raise ValueError("EMBEDDING_PROVIDER must be 'local' or 'gemini'")
        if self.generation_provider not in {"extractive", "gemini", "vercel_gateway"}:
            raise ValueError(
                "GENERATION_PROVIDER must be 'extractive', 'gemini', or 'vercel_gateway'"
            )
        if self.generation_provider == "vercel_gateway" and "/" not in self.ai_gateway_model:
            raise ValueError("AI_GATEWAY_MODEL must use provider/model format")
        if self.is_deployment and self.embedding_provider != "gemini":
            raise ValueError(
                "Deployment requires EMBEDDING_PROVIDER=gemini; the local provider "
                "depends on Torch and a persistent model cache."
            )
        if not 1 <= self.embedding_dimension <= 2000:
            raise ValueError("EMBEDDING_DIMENSION must be between 1 and 2000")
        if self.online_embedding_batch_size < 1:
            raise ValueError("ONLINE_EMBEDDING_BATCH_SIZE must be at least 1")
        if self.online_embedding_rpm < 1:
            raise ValueError("ONLINE_EMBEDDING_RPM must be at least 1")
        if not self.database_url.startswith(("postgresql://", "postgres://")):
            raise ValueError("DATABASE_URL must be a PostgreSQL connection URL")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"CHUNK_OVERLAP ({self.chunk_overlap}) must be less than "
                f"CHUNK_SIZE ({self.chunk_size})"
            )
        if not self.ocr_language.strip():
            raise ValueError("OCR_LANGUAGE must not be empty")
        if not 72 <= self.ocr_dpi <= 600:
            raise ValueError("OCR_DPI must be between 72 and 600")
        if not self.data_dir.exists():
            logger.warning(f"DATA_DIR does not exist: {self.data_dir}")
        logger.debug("Configuration loaded: model=%s, top_k=%d, threshold=%.2f",
                     self.gemini_model, self.top_k, self.similarity_threshold)

    @property
    def resolved_embedding_namespace(self) -> str:
        """Return an explicit or provider-derived database namespace."""
        if self.embedding_namespace:
            return self.embedding_namespace
        return f"{self.embedding_provider}_{self.embedding_dimension}"

    @property
    def is_deployment(self) -> bool:
        return self.app_env in {"deployment", "production", "preview"}

    @property
    def schema_database_url(self) -> str:
        """Prefer Neon's direct connection for schema operations and migrations."""
        return self.database_url_unpooled or self.database_url

    @property
    def generation_configured(self) -> bool:
        """Return whether the active generation provider has runtime credentials."""
        if self.generation_provider == "vercel_gateway":
            return bool(self.ai_gateway_api_key or self.vercel_oidc_token)
        if self.generation_provider == "extractive":
            return True
        return bool(self.gemini_api_key)


# ---------------------------------------------------------------------------
# Singleton — import this everywhere
# ---------------------------------------------------------------------------
config = AppConfig()
