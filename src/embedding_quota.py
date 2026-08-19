"""Persistent Gemini embedding quota accounting and ingestion checkpoints."""

from __future__ import annotations

import contextvars
import math
import random
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator, Protocol
from zoneinfo import ZoneInfo

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from src.config import config


PACIFIC = ZoneInfo("America/Los_Angeles")
SCHEMA_PATH = Path(__file__).resolve().parent.parent / "database" / "embedding_quota_schema.sql"
_run_id: contextvars.ContextVar[str] = contextvars.ContextVar("embedding_run_id", default="")


class EmbeddingQuotaExceeded(RuntimeError):
    """Raised before a provider request would exceed a configured budget."""

    def __init__(self, reason: str, retry_after: float | None = None) -> None:
        super().__init__(f"Gemini embedding {reason} budget is exhausted")
        self.reason = reason
        self.retry_after = retry_after
        self.resumable = reason == "rpd"


@dataclass(frozen=True)
class QuotaLimits:
    rpm: int
    tpm: int
    rpd: int
    safety_factor: float

    def effective(self, name: str) -> int:
        factor = 0.95 if name == "rpd" else self.safety_factor
        return max(1, math.floor(getattr(self, name) * factor))


@dataclass(frozen=True)
class QuotaSnapshot:
    rpm_used: int
    tpm_used: int
    rpd_used: int
    rpm_limit: int
    tpm_limit: int
    rpd_limit: int
    safety_factor: float
    allowed: bool = True
    blocked_by: str = ""
    retry_after_seconds: float = 0.0

    @staticmethod
    def _utilization(used: int, limit: int) -> float:
        return round(used / limit, 4) if limit else 0.0

    @property
    def utilization(self) -> dict[str, float]:
        return {
            "rpm": self._utilization(self.rpm_used, self.rpm_limit),
            "tpm": self._utilization(self.tpm_used, self.tpm_limit),
            "rpd": self._utilization(self.rpd_used, self.rpd_limit),
        }

    @property
    def alert(self) -> str:
        highest = max(self.utilization.values(), default=0.0)
        if highest >= 0.95:
            return "hard_stop"
        if highest >= 0.85:
            return "critical"
        if highest >= 0.70:
            return "warning"
        return "normal"

    def serializable(self) -> dict[str, Any]:
        value = asdict(self)
        value["utilization"] = self.utilization
        value["remaining"] = {
            "rpm": max(0, self.rpm_limit - self.rpm_used),
            "tpm": max(0, self.tpm_limit - self.tpm_used),
            "rpd": max(0, self.rpd_limit - self.rpd_used),
        }
        value["alert"] = self.alert
        return value


class QuotaRepository(Protocol):
    def reserve(self, *, limits: QuotaLimits, input_tokens: int, embedded_items: int,
                operation: str, run_id: str, now: datetime) -> QuotaSnapshot: ...
    def record_event(self, *, event_type: str, operation: str, run_id: str,
                     embedded_items: int = 0, input_tokens: int = 0,
                     retry_delay_seconds: float = 0.0, error_code: str = "") -> None: ...
    def quota_snapshot(self, limits: QuotaLimits, now: datetime) -> QuotaSnapshot: ...
    def start_run(self, **values: Any) -> str: ...
    def checkpoint_run(self, run_id: str, **values: Any) -> None: ...
    def recent_runs(self, limit: int = 20) -> list[dict[str, Any]]: ...
    def recent_events(self, limit: int = 50) -> list[dict[str, Any]]: ...


def pacific_day_start(now: datetime) -> datetime:
    """Return midnight in Gemini's America/Los_Angeles quota day, as UTC."""
    local = now.astimezone(PACIFIC)
    return local.replace(hour=0, minute=0, second=0, microsecond=0).astimezone(timezone.utc)


def estimate_input_tokens(texts: list[str]) -> int:
    """Conservative provider-independent estimate used only for quota pacing."""
    return sum(max(len(text.split()), math.ceil(len(text) / 3)) for text in texts)


def quota_block_reason(usage: dict[str, int], limits: QuotaLimits, input_tokens: int) -> str:
    """Return the first exhausted budget in daily, request, then token order."""
    proposed = {
        "rpm": int(usage.get("rpm_used", 0)) + 1,
        "tpm": int(usage.get("tpm_used", 0)) + input_tokens,
        "rpd": int(usage.get("rpd_used", 0)) + 1,
    }
    return next(
        (name for name in ("rpd", "rpm", "tpm") if proposed[name] > limits.effective(name)),
        "",
    )


class PostgresQuotaRepository:
    """Atomic quota reservations and run checkpoints stored in Postgres."""

    def __init__(self, database_url: str | None = None) -> None:
        self.database_url = database_url or config.database_url
        self._schema_ready = False

    def ensure_schema(self) -> None:
        if self._schema_ready:
            return
        with psycopg.connect(config.schema_database_url, autocommit=True) as connection:
            connection.execute(SCHEMA_PATH.read_text(encoding="utf-8"))
        self._schema_ready = True

    @staticmethod
    def _snapshot(row: dict[str, Any], limits: QuotaLimits, **extra: Any) -> QuotaSnapshot:
        return QuotaSnapshot(
            rpm_used=int(row.get("rpm_used") or 0),
            tpm_used=int(row.get("tpm_used") or 0),
            rpd_used=int(row.get("rpd_used") or 0),
            rpm_limit=limits.rpm,
            tpm_limit=limits.tpm,
            rpd_limit=limits.rpd,
            safety_factor=limits.safety_factor,
            **extra,
        )

    @staticmethod
    def _usage(connection: Any, now: datetime) -> dict[str, Any]:
        return connection.execute(
            """
            SELECT
                COALESCE(sum(request_count) FILTER (WHERE recorded_at > %s), 0) AS rpm_used,
                COALESCE(sum(input_tokens) FILTER (WHERE recorded_at > %s), 0) AS tpm_used,
                COALESCE(sum(request_count) FILTER (WHERE recorded_at >= %s), 0) AS rpd_used
            FROM rag_embedding_events
            WHERE event_type = 'reserved'
            """,
            (now - timedelta(seconds=60), now - timedelta(seconds=60), pacific_day_start(now)),
        ).fetchone()

    def reserve(self, *, limits: QuotaLimits, input_tokens: int, embedded_items: int,
                operation: str, run_id: str, now: datetime) -> QuotaSnapshot:
        self.ensure_schema()
        with psycopg.connect(self.database_url, row_factory=dict_row) as connection:
            connection.execute("SELECT pg_advisory_xact_lock(hashtext('creativa_gemini_embedding_quota'))")
            usage = self._usage(connection, now)
            proposed = {
                "rpm": int(usage["rpm_used"]) + 1,
                "tpm": int(usage["tpm_used"]) + input_tokens,
                "rpd": int(usage["rpd_used"]) + 1,
            }
            blocked_by = quota_block_reason(usage, limits, input_tokens)
            if blocked_by:
                retry = 0.0 if blocked_by == "rpd" else 60.0
                return self._snapshot(usage, limits, allowed=False, blocked_by=blocked_by,
                                      retry_after_seconds=retry)
            connection.execute(
                """
                INSERT INTO rag_embedding_events (
                    run_id, event_type, operation, provider, model, dimension,
                    namespace, table_family, request_count, input_tokens, embedded_items, recorded_at
                ) VALUES (NULLIF(%s, '')::uuid, 'reserved', %s, 'gemini', %s, %s, %s, %s, 1, %s, %s, %s)
                """,
                (run_id, operation, config.online_embedding_model, config.embedding_dimension,
                 config.resolved_embedding_namespace, config.embedding_table_family,
                 input_tokens, embedded_items, now),
            )
            usage = {"rpm_used": proposed["rpm"], "tpm_used": proposed["tpm"], "rpd_used": proposed["rpd"]}
            return self._snapshot(usage, limits)

    def record_event(self, *, event_type: str, operation: str, run_id: str,
                     embedded_items: int = 0, input_tokens: int = 0,
                     retry_delay_seconds: float = 0.0, error_code: str = "") -> None:
        self.ensure_schema()
        with psycopg.connect(self.database_url) as connection:
            connection.execute(
                """
                INSERT INTO rag_embedding_events (
                    run_id, event_type, operation, provider, model, dimension, namespace,
                    table_family, input_tokens, embedded_items, retry_delay_ms, error_code
                ) VALUES (NULLIF(%s, '')::uuid, %s, %s, 'gemini', %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (run_id, event_type, operation, config.online_embedding_model,
                 config.embedding_dimension, config.resolved_embedding_namespace,
                 config.embedding_table_family, input_tokens, embedded_items,
                 round(retry_delay_seconds * 1000), error_code[:80]),
            )

    def quota_snapshot(self, limits: QuotaLimits, now: datetime) -> QuotaSnapshot:
        self.ensure_schema()
        with psycopg.connect(self.database_url, row_factory=dict_row) as connection:
            return self._snapshot(self._usage(connection, now), limits)

    def start_run(self, **values: Any) -> str:
        self.ensure_schema()
        run_id = str(uuid.uuid4())
        with psycopg.connect(self.database_url) as connection:
            connection.execute(
                """
                INSERT INTO rag_embedding_runs (
                    run_id, namespace, table_family, dimension, model, corpus_hash,
                    total_documents, status, checkpoint
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, 'running', %s::jsonb)
                """,
                (run_id, values["namespace"], values["table_family"], values["dimension"],
                 values["model"], values.get("corpus_hash", ""),
                 values.get("total_documents", 0), Jsonb(values.get("checkpoint", {}))),
            )
        return run_id

    def checkpoint_run(self, run_id: str, **values: Any) -> None:
        self.ensure_schema()
        with psycopg.connect(self.database_url) as connection:
            connection.execute(
                """
                UPDATE rag_embedding_runs SET
                    status = COALESCE(%s, status),
                    completed_documents = COALESCE(%s, completed_documents),
                    current_document = COALESCE(%s, current_document),
                    checkpoint = COALESCE(%s::jsonb, checkpoint),
                    last_error = COALESCE(%s, last_error),
                    updated_at = now(),
                    completed_at = CASE WHEN %s IN ('completed', 'failed', 'paused_quota') THEN now() ELSE completed_at END
                WHERE run_id = %s
                """,
                (values.get("status"), values.get("completed_documents"),
                 values.get("current_document"),
                 Jsonb(values["checkpoint"]) if "checkpoint" in values else None,
                 values.get("last_error"), values.get("status"), run_id),
            )

    def recent_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        self.ensure_schema()
        with psycopg.connect(self.database_url, row_factory=dict_row) as connection:
            return list(connection.execute(
                "SELECT * FROM rag_embedding_runs ORDER BY created_at DESC LIMIT %s", (limit,)
            ).fetchall())

    def recent_events(self, limit: int = 50) -> list[dict[str, Any]]:
        self.ensure_schema()
        with psycopg.connect(self.database_url, row_factory=dict_row) as connection:
            return list(connection.execute(
                """SELECT event_id, run_id, recorded_at, event_type, operation, embedded_items,
                          retry_delay_ms, error_code FROM rag_embedding_events
                   WHERE event_type <> 'reserved' ORDER BY recorded_at DESC LIMIT %s""",
                (limit,),
            ).fetchall())


class EmbeddingQuotaController:
    def __init__(self, repository: QuotaRepository | None = None,
                 limits: QuotaLimits | None = None) -> None:
        self.limits = limits or QuotaLimits(
            config.gemini_embedding_rpm_limit,
            config.gemini_embedding_tpm_limit,
            config.gemini_embedding_rpd_limit,
            config.gemini_embedding_safety_factor,
        )
        self.enabled = all((self.limits.rpm, self.limits.tpm, self.limits.rpd))
        self.repository = repository or PostgresQuotaRepository()

    def acquire(self, texts: list[str], operation: str, *, interactive: bool) -> QuotaSnapshot | None:
        if not self.enabled:
            return None
        input_tokens = estimate_input_tokens(texts)
        while True:
            snapshot = self.repository.reserve(
                limits=self.limits, input_tokens=input_tokens, embedded_items=len(texts),
                operation=operation, run_id=_run_id.get(), now=datetime.now(timezone.utc),
            )
            if snapshot.allowed:
                return snapshot
            if interactive or snapshot.blocked_by == "rpd":
                self.repository.record_event(event_type="quota_blocked", operation=operation,
                                             run_id=_run_id.get(), error_code=snapshot.blocked_by)
                raise EmbeddingQuotaExceeded(snapshot.blocked_by, snapshot.retry_after_seconds)
            time.sleep(max(0.25, snapshot.retry_after_seconds))

    def event(self, event_type: str, operation: str, **values: Any) -> None:
        if self.enabled:
            self.repository.record_event(event_type=event_type, operation=operation,
                                         run_id=_run_id.get(), **values)

    def retry_delay(self, attempt: int, provider_delay: float | None = None) -> float:
        if provider_delay is not None:
            return provider_delay
        return min(60.0, (2 ** attempt) + random.uniform(0.0, 1.0))

    @contextmanager
    def run_scope(self, run_id: str) -> Iterator[None]:
        token = _run_id.set(run_id)
        try:
            yield
        finally:
            _run_id.reset(token)

    def dashboard(self) -> dict[str, Any]:
        if not self.enabled:
            return {"configured": False, "active_dimension": config.embedding_dimension,
                    "namespace": config.resolved_embedding_namespace,
                    "table_family": config.embedding_table_family}
        snapshot = self.repository.quota_snapshot(self.limits, datetime.now(timezone.utc))
        return {
            "configured": True,
            "quota": snapshot.serializable(),
            "active_dimension": config.embedding_dimension,
            "namespace": config.resolved_embedding_namespace,
            "table_family": config.embedding_table_family,
            "model": config.online_embedding_model,
            "runs": self.repository.recent_runs(),
            "events": self.repository.recent_events(),
        }


embedding_quota = EmbeddingQuotaController()
