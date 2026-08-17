"""Central configuration — loads all settings from .env / environment variables.

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

# Load .env from the project root (two levels above this file: src/ → project root)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env", override=False)

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

# ChromaDB collection names (prefixed to avoid conflicts)
COLLECTION_PREFIX = "diabetes"


def _collection_name(category: str) -> str:
    """Return the ChromaDB collection name for a given category."""
    return f"{COLLECTION_PREFIX}_{category}"


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
    gemini_model: str = field(default_factory=lambda: os.environ.get("GEMINI_MODEL", "gemini-2.0-flash"))

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

    # ── Data / Storage ─────────────────────────────────────────────────────
    data_dir: Path = field(
        default_factory=lambda: _PROJECT_ROOT / os.environ.get("DATA_DIR", "data/rew_data/books")
    )
    chroma_db_dir: Path = field(
        default_factory=lambda: _PROJECT_ROOT / os.environ.get("CHROMA_DB_DIR", "chroma_db")
    )

    # ── Application ────────────────────────────────────────────────────────
    debug: bool = field(default_factory=lambda: os.environ.get("DEBUG", "false").lower() == "true")
    max_memory_turns: int = field(
        default_factory=lambda: int(os.environ.get("MAX_MEMORY_TURNS", "6"))
    )

    # ── Derived ────────────────────────────────────────────────────────────
    project_root: Path = field(default_factory=lambda: _PROJECT_ROOT)

    def validate(self) -> None:
        """Log warnings for missing or suspicious configuration values."""
        if not self.gemini_api_key:
            logger.warning(
                "GEMINI_API_KEY is not set. Generation will fail. "
                "Copy .env.example to .env and add your key."
            )
        if self.embedding_provider not in {"local", "gemini"}:
            raise ValueError("EMBEDDING_PROVIDER must be 'local' or 'gemini'")
        if not 1 <= self.embedding_dimension <= 2000:
            raise ValueError("EMBEDDING_DIMENSION must be between 1 and 2000")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"CHUNK_OVERLAP ({self.chunk_overlap}) must be less than "
                f"CHUNK_SIZE ({self.chunk_size})"
            )
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


# ---------------------------------------------------------------------------
# Singleton — import this everywhere
# ---------------------------------------------------------------------------
config = AppConfig()
