"""Configurable local and Gemini API embedding providers."""

import logging
import math
import re
import time
from collections import deque
from typing import Optional

from src.config import config
from src.embedding_quota import EmbeddingQuotaController, embedding_quota, estimate_input_tokens

logger = logging.getLogger(__name__)


def _retry_delay_seconds(message: str, fallback: float) -> float:
    """Return the provider-requested retry delay with a small safety margin."""
    hints: list[float] = []
    for pattern in (
        r"retry in\s+([0-9]+(?:\.[0-9]+)?)s",
        r"retryDelay['\"]?\s*:\s*['\"]([0-9]+(?:\.[0-9]+)?)s",
    ):
        hints.extend(float(value) for value in re.findall(pattern, message, re.I))
    return max(fallback, (max(hints) + 1.0) if hints else 0.0)


def _normalize(values: list[float]) -> list[float]:
    """Return a unit-length vector for reliable cosine search."""
    magnitude = math.sqrt(sum(value * value for value in values))
    if magnitude == 0:
        raise ValueError("Embedding provider returned a zero vector")
    return [value / magnitude for value in values]


def _local_dimension(model) -> int:
    """Read dimensions from current or older sentence-transformers versions."""
    getter = getattr(model, "get_embedding_dimension", None)
    if getter is None:
        getter = model.get_sentence_embedding_dimension
    return int(getter())


class EmbeddingModel:
    """Small facade over local sentence-transformers and Gemini embeddings."""

    def __init__(
        self,
        provider: Optional[str] = None,
        dimension: Optional[int] = None,
        local_model_name: Optional[str] = None,
        online_model_name: Optional[str] = None,
        online_client=None,
        quota_controller: EmbeddingQuotaController | None = None,
    ) -> None:
        self._provider = (provider or config.embedding_provider).lower()
        if self._provider not in {"local", "gemini"}:
            raise ValueError("Embedding provider must be 'local' or 'gemini'")

        self._dimension = dimension or config.embedding_dimension
        self._local_model_name = local_model_name or config.embedding_model
        self._online_model_name = online_model_name or config.online_embedding_model
        self._local_model = None
        self._online_client = online_client
        self._online_usage: deque[tuple[float, int]] = deque()
        self._quota = quota_controller or embedding_quota

    def _wait_for_online_quota(self, item_count: int) -> None:
        """Keep successful embedded items below the configured rolling RPM cap."""
        if self._quota.enabled:
            return
        window_seconds = 60.0
        while True:
            now = time.monotonic()
            while self._online_usage and now - self._online_usage[0][0] >= window_seconds:
                self._online_usage.popleft()
            used = sum(count for _, count in self._online_usage)
            if used + item_count <= config.online_embedding_rpm:
                return
            wait = max(0.25, window_seconds - (now - self._online_usage[0][0]) + 0.25)
            logger.info(
                "Gemini free-tier pacing: waiting %.1fs before embedding %d items",
                wait,
                item_count,
            )
            time.sleep(wait)

    def _load_local(self) -> None:
        if self._local_model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Local embeddings require `uv sync --extra local`."
            ) from exc

        logger.info("Loading local embedding model: %s", self._local_model_name)
        self._local_model = SentenceTransformer(self._local_model_name)
        actual = _local_dimension(self._local_model)
        if actual != self._dimension:
            raise ValueError(
                f"Local model outputs {actual} dimensions, but "
                f"EMBEDDING_DIMENSION is {self._dimension}."
            )

    def _get_online_client(self):
        if self._online_client is None:
            if not config.gemini_api_key:
                raise RuntimeError("GEMINI_API_KEY is required for Gemini embeddings")
            from google import genai

            self._online_client = genai.Client(api_key=config.gemini_api_key)
        return self._online_client

    @staticmethod
    def _online_text(text: str, task: str) -> str:
        if task == "query":
            return f"Task: retrieve relevant diabetes reference passages\nQuery: {text}"
        return f"Task: represent a diabetes reference passage for retrieval\nDocument: {text}"

    def _embed_online_request(self, texts: list[str], task: str) -> list[list[float]]:
        """Send one bounded Gemini embedding request with transient-error retries."""
        from google.genai import types

        client = self._get_online_client()
        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=self._online_text(text, task))],
            )
            for text in texts
        ]
        response = None
        last_error: Exception | None = None
        prepared_texts = [self._online_text(text, task) for text in texts]
        # Interactive retrieval must fail fast so the database lexical fallback
        # can answer within a serverless request. Corpus ingestion can wait and retry.
        fallback_delays = () if task == "query" else (2.0, 5.0, 10.0, 30.0)
        for attempt in range(len(fallback_delays) + 1):
            self._wait_for_online_quota(len(texts))
            self._quota.acquire(prepared_texts, task, interactive=task == "query")
            try:
                response = client.models.embed_content(
                    model=self._online_model_name,
                    contents=contents,
                    config=types.EmbedContentConfig(
                        output_dimensionality=self._dimension,
                    ),
                )
                break
            except Exception as exc:
                last_error = exc
                message = str(exc).lower()
                retryable = any(
                    marker in message
                    for marker in ("429", "quota", "rate", "resource_exhausted", "503", "unavailable")
                )
                if not retryable or attempt == len(fallback_delays):
                    self._quota.event(
                        "failed", task, embedded_items=len(texts),
                        input_tokens=estimate_input_tokens(prepared_texts),
                        error_code="provider_error",
                    )
                    raise
                fallback_delay = (
                    self._quota.retry_delay(attempt)
                    if self._quota.enabled
                    else fallback_delays[attempt]
                )
                delay = _retry_delay_seconds(message, fallback_delay)
                self._quota.event(
                    "rate_limited" if "429" in message or "resource_exhausted" in message else "retry",
                    task,
                    embedded_items=len(texts),
                    input_tokens=estimate_input_tokens(prepared_texts),
                    retry_delay_seconds=delay,
                    error_code="resource_exhausted" if "429" in message or "resource_exhausted" in message else "unavailable",
                )
                logger.warning(
                    "Gemini embedding request throttled/unavailable; retrying in %.1fs",
                    delay,
                )
                time.sleep(delay)
                continue

        if response is None:
            raise RuntimeError(f"Gemini embedding request failed: {last_error}")
        self._online_usage.append((time.monotonic(), len(texts)))
        self._quota.event(
            "succeeded", task, embedded_items=len(texts),
            input_tokens=estimate_input_tokens(prepared_texts),
        )
        vectors = [_normalize(list(item.values)) for item in response.embeddings]
        if len(vectors) != len(texts):
            raise RuntimeError(
                f"Gemini returned {len(vectors)} embeddings for {len(texts)} inputs"
            )
        return vectors

    def _embed_online(self, texts: list[str], task: str) -> list[list[float]]:
        """Embed inputs in bounded batches so corpus ingestion fits API limits."""
        vectors: list[list[float]] = []
        batch_size = config.online_embedding_batch_size
        for start in range(0, len(texts), batch_size):
            vectors.extend(self._embed_online_request(texts[start : start + batch_size], task))
        return vectors

    @property
    def dimension(self) -> int:
        if self._provider == "local":
            self._load_local()
        return self._dimension

    def embed_query(self, text: str) -> list[float]:
        """Embed one retrieval query."""
        text = text.strip()
        if not text:
            raise ValueError("Cannot embed an empty query")

        if self._provider == "gemini":
            return self._embed_online([text], task="query")[0]

        self._load_local()
        vector = self._local_model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vector.tolist()

    def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> list[list[float]]:
        """Embed document passages while preserving their order."""
        if not texts:
            return []
        clean_texts = [text.strip() if text and text.strip() else "." for text in texts]

        if self._provider == "gemini":
            return self._embed_online(clean_texts, task="document")

        self._load_local()
        vectors = self._local_model.encode(
            clean_texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vectors.tolist()

    @property
    def model_name(self) -> str:
        if self._provider == "gemini":
            return self._online_model_name
        return self._local_model_name

    @property
    def provider(self) -> str:
        return self._provider


embedder = EmbeddingModel()
