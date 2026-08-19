"""Central configuration for development and deployment environments.

All tuneable parameters live here. No other module should read os.environ directly.

Usage
-----
    from src.config import config
    print(config.gemini_model)
"""

import json
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


def _pricing_map_from_environment() -> dict[str, dict[str, dict[str, float]]]:
    """Read provider/model token pricing without embedding volatile prices in code.

    The optional value is JSON shaped as
    ``{"provider": {"model": {"input": 1.25, "output": 5.0}}}``, with
    values expressed in USD per million tokens.  A ``"*"`` model can provide
    a provider-wide default.  Invalid configuration must fail at startup
    rather than silently producing an incorrect cost figure.
    """
    raw = os.environ.get("GENERATION_PRICING_USD_PER_MILLION_JSON", "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("GENERATION_PRICING_USD_PER_MILLION_JSON must be valid JSON") from exc
    if not isinstance(parsed, dict):
        raise ValueError("GENERATION_PRICING_USD_PER_MILLION_JSON must be an object")
    pricing: dict[str, dict[str, dict[str, float]]] = {}
    for provider, models in parsed.items():
        if not isinstance(provider, str) or not isinstance(models, dict):
            raise ValueError("pricing configuration must map providers to model objects")
        pricing[provider.lower()] = {}
        for model, prices in models.items():
            if not isinstance(model, str) or not isinstance(prices, dict):
                raise ValueError("pricing configuration must map models to price objects")
            if set(prices) - {"input", "output"} or "input" not in prices or "output" not in prices:
                raise ValueError("each pricing entry requires numeric input and output prices")
            try:
                input_price, output_price = float(prices["input"]), float(prices["output"])
            except (TypeError, ValueError) as exc:
                raise ValueError("pricing values must be numeric") from exc
            if input_price < 0 or output_price < 0:
                raise ValueError("pricing values must be zero or positive")
            pricing[provider.lower()][model] = {"input": input_price, "output": output_price}
    return pricing


# ---------------------------------------------------------------------------
# Category constants — used throughout the codebase
# ---------------------------------------------------------------------------

CATEGORY_TREATMENT = "treatment"
CATEGORY_PREVENTION = "prevention"
CATEGORY_NUTRITION = "nutrition"
CATEGORY_GENERAL = "general"
CATEGORY_ALL = "all"

ALL_CATEGORIES = [CATEGORY_TREATMENT, CATEGORY_PREVENTION, CATEGORY_NUTRITION]

CHUNK_PROFILES: dict[str, tuple[int, int]] = {
    "small": (1200, 0),
    "balanced": (2000, 200),
    "large": (3000, 300),
}

CHUNK_PROFILE_DEFAULT = "balanced"

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
        default_factory=lambda: os.environ.get("GENERATION_PROVIDER", "auto").lower()
    )
    generation_primary_provider: str = field(
        default_factory=lambda: os.environ.get("GENERATION_PRIMARY_PROVIDER", "gemini").lower()
    )
    generation_fallback_provider: str = field(
        default_factory=lambda: os.environ.get("GENERATION_FALLBACK_PROVIDER", "groq").lower()
    )
    groq_api_key: str = field(default_factory=lambda: os.environ.get("GROQ_API_KEY", ""))
    groq_model: str = field(
        default_factory=lambda: os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
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
    generation_input_cost_per_million_usd: float = field(
        default_factory=lambda: float(os.environ.get("GENERATION_INPUT_COST_PER_MILLION_USD", "0"))
    )
    generation_output_cost_per_million_usd: float = field(
        default_factory=lambda: float(os.environ.get("GENERATION_OUTPUT_COST_PER_MILLION_USD", "0"))
    )
    generation_pricing_usd_per_million: dict[str, dict[str, dict[str, float]]] = field(
        default_factory=_pricing_map_from_environment
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
    active_index_namespace: str = field(
        default_factory=lambda: os.environ.get("ACTIVE_INDEX_NAMESPACE", "")
    )
    online_embedding_batch_size: int = field(
        default_factory=lambda: int(os.environ.get("ONLINE_EMBEDDING_BATCH_SIZE", "16"))
    )
    online_embedding_rpm: int = field(
        default_factory=lambda: int(os.environ.get("ONLINE_EMBEDDING_RPM", "90"))
    )

    # ── Retrieval ──────────────────────────────────────────────────────────
    top_k: int = field(default_factory=lambda: int(os.environ.get("TOP_K", "5")))
    retrieval_profile: str = field(
        default_factory=lambda: os.environ.get("RETRIEVAL_PROFILE", "balanced").lower()
    )
    similarity_threshold: float = field(
        default_factory=lambda: float(os.environ.get("SIMILARITY_THRESHOLD", "0.30"))
    )
    active_index_namespace: str = field(
        default_factory=lambda: os.environ.get("ACTIVE_INDEX_NAMESPACE", "")
    )
    retrieval_profile: str = field(
        default_factory=lambda: os.environ.get("RETRIEVAL_PROFILE", "default")
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
            "postgresql://creativa:creativa-local@localhost:5433/creativa_diabetes",
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
        if self.generation_provider == "groq" and not self.groq_api_key:
            logger.warning("Direct Groq generation is selected without GROQ_API_KEY")
        if self.generation_provider == "vercel_gateway" and not self.generation_configured:
            logger.warning("AI Gateway credentials are unavailable outside the Vercel runtime")
        if self.embedding_provider not in {"local", "gemini"}:
            raise ValueError("EMBEDDING_PROVIDER must be 'local' or 'gemini'")
        generation_providers = {"extractive", "gemini", "groq", "vercel_gateway", "auto"}
        if self.generation_provider not in generation_providers:
            raise ValueError(
                "GENERATION_PROVIDER must be 'extractive', 'gemini', 'groq', "
                "'vercel_gateway', or 'auto'"
            )
        if self.generation_primary_provider not in generation_providers - {"auto"}:
            raise ValueError("GENERATION_PRIMARY_PROVIDER must name a concrete provider")
        if self.generation_fallback_provider and self.generation_fallback_provider not in generation_providers - {"auto", "extractive"}:
            raise ValueError("GENERATION_FALLBACK_PROVIDER must be blank, 'gemini', 'groq', or 'vercel_gateway'")
        if self.generation_fallback_provider == self.generation_primary_provider:
            raise ValueError("GENERATION_FALLBACK_PROVIDER must differ from GENERATION_PRIMARY_PROVIDER")
        if self.generation_provider == "vercel_gateway" and "/" not in self.ai_gateway_model:
            raise ValueError("AI_GATEWAY_MODEL must use provider/model format")
        if self.generation_input_cost_per_million_usd < 0 or self.generation_output_cost_per_million_usd < 0:
            raise ValueError("Generation token prices must be zero or positive")
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
        if self.retrieval_profile not in CHUNK_PROFILES:
            raise ValueError(
                "RETRIEVAL_PROFILE must be one of "
                + ", ".join(sorted(CHUNK_PROFILES))
            )
        if self.active_index_namespace and self.embedding_namespace and (
            self.active_index_namespace != self.embedding_namespace
        ):
            raise ValueError(
                "ACTIVE_INDEX_NAMESPACE and EMBEDDING_NAMESPACE conflict; "
                "use ACTIVE_INDEX_NAMESPACE only"
            )
        if self.top_k not in {3, 4, 5}:
            raise ValueError("TOP_K must be one of 3, 4, or 5")
        if not self.database_url.startswith(("postgresql://", "postgres://")):
            raise ValueError("DATABASE_URL must be a PostgreSQL connection URL")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"CHUNK_OVERLAP ({self.chunk_overlap}) must be less than "
                f"CHUNK_SIZE ({self.chunk_size})"
            )
        if self.top_k <= 0:
            raise ValueError("TOP_K must be a positive integer")
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
        if self.active_index_namespace:
            return self.active_index_namespace
        if self.embedding_namespace:  # Backward-compatible migration fallback.
            return self.embedding_namespace
        return f"{self.embedding_provider}_{self.embedding_dimension}"

    @property
    def selected_chunk_profile(self) -> tuple[int, int]:
        """Return the fixed character settings for the selected named profile."""
        return CHUNK_PROFILES[self.retrieval_profile]

    @property
    def is_deployment(self) -> bool:
        return self.app_env in {"deployment", "production", "preview"}

    @property
    def schema_database_url(self) -> str:
        """Prefer Neon's direct connection for schema operations and migrations."""
        return self.database_url_unpooled or self.database_url

    @property
    def generation_configured(self) -> bool:
        """The deterministic evidence fallback keeps grounded answers available."""
        return True

    def provider_configured(self, provider: str) -> bool:
        """Return whether one concrete generation provider has credentials."""
        if provider == "vercel_gateway":
            return bool(self.ai_gateway_api_key or self.vercel_oidc_token)
        if provider == "groq":
            return bool(self.groq_api_key)
        if provider == "extractive":
            return True
        return bool(self.gemini_api_key)

    def generation_pricing(self, provider: str, model: str) -> tuple[float, float] | None:
        """Return configured USD-per-million input/output prices for a request.

        The legacy pair of environment variables remains a backwards-compatible
        generic fallback.  A configured map takes precedence and may contain a
        provider-wide ``*`` model entry.
        """
        provider_prices = self.generation_pricing_usd_per_million.get(provider.lower(), {})
        price = provider_prices.get(model) or provider_prices.get("*")
        if price is not None:
            return float(price["input"]), float(price["output"])
        if self.generation_input_cost_per_million_usd or self.generation_output_cost_per_million_usd:
            return self.generation_input_cost_per_million_usd, self.generation_output_cost_per_million_usd
        return None

    @property
    def configured_generation_provider_label(self) -> str:
        """Human-readable configured generation route before a request selects a fallback."""
        labels = {
            "gemini": "Gemini",
            "groq": "Groq",
            "vercel_gateway": "Vercel AI Gateway",
            "extractive": "Evidence excerpts",
        }
        if self.generation_provider == "extractive":
            return labels["extractive"]
        if self.generation_provider == "auto":
            providers = [self.generation_primary_provider, self.generation_fallback_provider]
        else:
            providers = [self.generation_provider, self.generation_fallback_provider]
        chain = [labels.get(provider, provider) for provider in providers if provider]
        chain.append(labels["extractive"])
        return " → ".join(dict.fromkeys(chain)) + " (automatic)"


# ---------------------------------------------------------------------------
# Singleton — import this everywhere
# ---------------------------------------------------------------------------
config = AppConfig()
