"""Foundational RAG telemetry with inspectable JSON and durable PostgreSQL storage."""

from __future__ import annotations

import json
import logging
import statistics
import tempfile
import threading
import time
import uuid
import re
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import psycopg
from psycopg.rows import dict_row

from src.config import CATEGORY_PREVENTION, config
from src.quality_metrics import answer_metrics, count_tokens, retrieval_metrics

logger = logging.getLogger(__name__)
RUNTIME_DIR = Path(tempfile.gettempdir()) / "creativa-diabetes" if config.is_deployment else config.project_root / ".runtime"
TRACE_FILE = RUNTIME_DIR / "request_traces.jsonl"
METRICS_SNAPSHOT_FILE = RUNTIME_DIR / "metrics_snapshot.json"
BENCHMARK_FILE = RUNTIME_DIR / "benchmark_history.jsonl"
METRICS_SCHEMA_PATH = config.project_root / "database" / "metrics_schema.sql"

FOUNDATIONAL_QUALITY_KEYS = (
    "hit_rate_at_k", "precision_at_k", "recall_at_k", "reciprocal_rank",
    "average_precision", "ndcg_at_k", "exact_match", "token_precision",
    "token_recall", "token_f1", "task_success",
)


@dataclass
class RequestTrace:
    query: str
    requested_category: str
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    conversation_id: str = ""
    turn_index: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    stages_ms: dict[str, float] = field(default_factory=dict)
    routed_category: str = ""
    language: str = ""
    risk_tier: str = ""
    namespace: str = ""
    index_manifest_hash: str = ""
    embedding_model: str = field(default_factory=lambda: config.embedding_model)
    retrieval_profile: str = field(default_factory=lambda: config.retrieval_profile)
    retrieval_count: int = 0
    best_score: float = 0.0
    top_k: int = 0
    retrieved_chunks: list[dict[str, Any]] = field(default_factory=list)
    answer: str = ""
    citations: str = ""
    generation_provider: str = ""
    generation_model: str = ""
    generation_attempts: list[dict[str, str]] = field(default_factory=list)
    provider_failure_count: int = 0
    fallback_count: int = 0
    input_tokens: int = 0
    context_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    token_count_method: str = "lexical_approximation"
    estimated_cost_usd: float | None = None
    cost_status: str = "pricing_not_configured"
    quality_metrics: dict[str, float | None] = field(default_factory=dict)
    quality_basis: dict[str, str] = field(default_factory=dict)
    label_case_id: str = ""
    status: str = "running"
    error: str = ""
    total_ms: float = 0.0
    _started: float = field(default_factory=time.perf_counter, repr=False)
    _gold_case: dict[str, Any] | None = field(default=None, repr=False)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            self.stages_ms[name] = round((time.perf_counter() - started) * 1000, 2)

    def capture_retrieval(self, envelope: Any) -> None:
        self.routed_category = envelope.routed_category
        self.language = "ar" if re.search(r"[\u0600-\u06FF]", self.query) else "en"
        from src.safety import classify_safety
        self.risk_tier = classify_safety(self.query).value
        self.namespace = envelope.namespace
        self.index_manifest_hash = envelope.index_manifest_hash
        self.retrieval_count = len(envelope.chunks)
        self.best_score = round(envelope.chunks[0].score, 4) if envelope.chunks else 0.0
        self.top_k = config.top_k
        self.retrieved_chunks = [
            {
                "rank": rank, "chunk_id": item.chunk_id, "document_name": item.document_name,
                "page_number": item.page_number, "section_title": item.section_title,
                "source_id": item.source_id, "source_url": item.source_url,
                "publisher": item.publisher, "publication_date": item.publication_date,
                "category": item.category, "language": item.language,
                "score": item.score, "distance": item.distance, "text": item.text,
            }
            for rank, item in enumerate(envelope.chunks, start=1)
        ]
        metrics, case = retrieval_metrics(self.query, envelope.chunks, self.top_k)
        self.quality_metrics.update(metrics)
        self._gold_case = case
        self.label_case_id = str(case.get("case_id", "")) if case else ""
        if case:
            self.quality_basis = {
                "retrieval": "curated_expected_provenance_and_text_anchor",
                "answer": "atomic_evidence_anchor_proxy" if case.get("expect_evidence") else "labeled_rejection_outcome",
            }
        self.context_tokens = sum(count_tokens(item.text) for item in envelope.chunks)

    def capture_generation(
        self, answer: str, provider: str, model: str, usage: dict[str, int] | None = None,
        attempts: list[dict[str, str]] | None = None,
        citations: str = "",
    ) -> None:
        self.answer = answer
        self.citations = citations
        self.generation_provider = provider
        self.generation_model = model
        self.generation_attempts = list(attempts or [])
        self.provider_failure_count = sum(item.get("status") == "error" for item in self.generation_attempts)
        self.fallback_count = int(self.provider_failure_count > 0 and provider in {"groq", "vercel_gateway", "extractive"})
        self.input_tokens = count_tokens(self.query) + self.context_tokens
        self.output_tokens = count_tokens(answer)
        self.total_tokens = self.input_tokens + self.output_tokens
        if usage and usage.get("total_tokens"):
            self.input_tokens = int(usage.get("input_tokens", 0))
            self.output_tokens = int(usage.get("output_tokens", 0))
            self.total_tokens = int(usage["total_tokens"])
            self.token_count_method = "provider_reported"
        if provider == "extractive":
            self.estimated_cost_usd = 0.0
            self.cost_status = "no_provider_cost"
        elif config.generation_input_cost_per_million_usd or config.generation_output_cost_per_million_usd:
            self.estimated_cost_usd = round(
                (self.input_tokens / 1_000_000) * config.generation_input_cost_per_million_usd
                + (self.output_tokens / 1_000_000) * config.generation_output_cost_per_million_usd,
                8,
            )
            self.cost_status = "configured_estimate"

    def finish(self, status: str = "ok", error: str = "") -> None:
        self.status = status
        self.error = error[:300]
        self.total_ms = round((time.perf_counter() - self._started) * 1000, 2)
        self.quality_metrics.update(answer_metrics(self.answer, self._gold_case, status))
        for key in FOUNDATIONAL_QUALITY_KEYS:
            self.quality_metrics.setdefault(key, None)

    def serializable(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("_started", None)
        value.pop("_gold_case", None)
        return value

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "RequestTrace":
        allowed = {name for name in cls.__dataclass_fields__ if not name.startswith("_")}
        trace = cls(**{key: value for key, value in record.items() if key in allowed})
        trace._started = time.perf_counter() - (float(record.get("total_ms", 0.0)) / 1000)
        from src.quality_metrics import find_case
        trace._gold_case = find_case(trace.query)
        return trace


class JsonlStore:
    """Thread-safe bounded JSON-lines storage."""

    def __init__(self, path: Path, max_records: int = 2000) -> None:
        self.path, self.max_records, self._lock = path, max_records, threading.Lock()

    def append(self, record: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            records = self.read()[-(self.max_records - 1):] + [record]
            temporary = self.path.with_suffix(self.path.suffix + ".tmp")
            temporary.write_text(
                "\n".join(json.dumps(item, ensure_ascii=False, sort_keys=True) for item in records) + "\n",
                encoding="utf-8",
            )
            temporary.replace(self.path)

    def read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        records = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return records


class MetricsRepository:
    """Dual-write repository; DB failure never blocks the answer path."""

    def __init__(self, json_store: JsonlStore | None = None, database_url: str | None = None) -> None:
        self.json_store = json_store or JsonlStore(TRACE_FILE)
        self.database_url = database_url or config.database_url
        self._schema_ready = False
        self._schema_lock = threading.Lock()

    def ensure_schema(self) -> None:
        if self._schema_ready:
            return
        with self._schema_lock:
            if not self._schema_ready:
                with psycopg.connect(config.schema_database_url, autocommit=True) as connection:
                    connection.execute(METRICS_SCHEMA_PATH.read_text(encoding="utf-8"))
                self._schema_ready = True

    def _deduplicated_json(self) -> list[dict[str, Any]]:
        latest: dict[str, dict[str, Any]] = {}
        for index, record in enumerate(self.json_store.read()):
            latest[str(record.get("trace_id") or index)] = record
        return sorted(latest.values(), key=lambda item: item.get("timestamp", ""))

    def save(self, record: dict[str, Any]) -> None:
        self.json_store.append(record)
        try:
            METRICS_SNAPSHOT_FILE.parent.mkdir(parents=True, exist_ok=True)
            temporary = METRICS_SNAPSHOT_FILE.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(build_metrics_report(self._deduplicated_json()), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            temporary.replace(METRICS_SNAPSHOT_FILE)
        except OSError:
            logger.exception("Could not refresh the metrics JSON snapshot")
        try:
            self.ensure_schema()
            params = {
                **record,
                "conversation_id": record.get("conversation_id", ""),
                "turn_index": record.get("turn_index", 0),
                "retrieval_ms": record.get("stages_ms", {}).get("retrieval"),
                "generation_ms": record.get("stages_ms", {}).get("generation"),
                "payload": json.dumps(record, ensure_ascii=False),
            }
            with psycopg.connect(self.database_url) as connection:
                connection.execute(
                    """
                    INSERT INTO rag_metric_events (
                        trace_id, conversation_id, turn_index, recorded_at, status, total_ms,
                        retrieval_ms, generation_ms, total_tokens, estimated_cost_usd, payload
                    ) VALUES (
                        %(trace_id)s::uuid, %(conversation_id)s, %(turn_index)s, %(timestamp)s::timestamptz,
                        %(status)s, %(total_ms)s, %(retrieval_ms)s, %(generation_ms)s,
                        %(total_tokens)s, %(estimated_cost_usd)s, %(payload)s::jsonb
                    ) ON CONFLICT (trace_id) DO UPDATE SET
                        conversation_id=EXCLUDED.conversation_id, turn_index=EXCLUDED.turn_index,
                        recorded_at=EXCLUDED.recorded_at, status=EXCLUDED.status, total_ms=EXCLUDED.total_ms,
                        retrieval_ms=EXCLUDED.retrieval_ms, generation_ms=EXCLUDED.generation_ms,
                        total_tokens=EXCLUDED.total_tokens, estimated_cost_usd=EXCLUDED.estimated_cost_usd,
                        payload=EXCLUDED.payload, updated_at=now()
                    """,
                    params,
                )
        except Exception:
            logger.exception("PostgreSQL metrics persistence failed; JSON trace remains available")

    def read(self, limit: int = 200, conversation_id: str = "") -> list[dict[str, Any]]:
        try:
            self.ensure_schema()
            where = "WHERE conversation_id = %s" if conversation_id else ""
            params: tuple[Any, ...] = (conversation_id, limit) if conversation_id else (limit,)
            with psycopg.connect(self.database_url, row_factory=dict_row) as connection:
                rows = connection.execute(
                    f"SELECT payload FROM rag_metric_events {where} ORDER BY recorded_at DESC LIMIT %s", params
                ).fetchall()
            database_records = [row["payload"] for row in reversed(rows)]
            merged = {
                str(item.get("trace_id")): item
                for item in [*self._deduplicated_json(), *database_records]
            }
            records = sorted(merged.values(), key=lambda item: item.get("timestamp", ""))
            return records[-limit:]
        except Exception:
            records = self._deduplicated_json()
            if conversation_id:
                records = [item for item in records if item.get("conversation_id") == conversation_id]
            return records[-limit:]

    def get(self, trace_id: str) -> dict[str, Any] | None:
        return next((item for item in reversed(self.read(limit=2000)) if item.get("trace_id") == trace_id), None)


trace_store = JsonlStore(TRACE_FILE)
metrics_repository = MetricsRepository(trace_store)
benchmark_store = JsonlStore(BENCHMARK_FILE)


def record_trace(trace: RequestTrace) -> None:
    metrics_repository.save(trace.serializable())


def load_trace(trace_id: str) -> RequestTrace | None:
    record = metrics_repository.get(trace_id)
    return RequestTrace.from_record(record) if record else None


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[round((len(ordered) - 1) * percentile)], 2)


def build_metrics_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [item for item in records if item.get("status") != "running"]
    total = len(completed)
    operational_failures = {"infrastructure_failure", "generation_error", "http_error", "timeout"}
    successful = sum(item.get("status") not in operational_failures for item in completed)
    def stage_values(name: str) -> list[float]:
        return [
            float(item.get("stages_ms", {}).get(name)) for item in completed
            if item.get("stages_ms", {}).get(name) is not None
        ]
    latencies = [float(item.get("total_ms", 0)) for item in completed]
    def distribution(values: list[float]) -> dict[str, float | None]:
        return {"p50": _percentile(values, .5), "p95": _percentile(values, .95), "p99": _percentile(values, .99)}
    quality: dict[str, Any] = {}
    for key in FOUNDATIONAL_QUALITY_KEYS:
        values = [
            float(item.get("quality_metrics", {}).get(key)) for item in completed
            if item.get("quality_metrics", {}).get(key) is not None
        ]
        quality[key] = {"mean": round(statistics.fmean(values), 6) if values else None, "measured_count": len(values)}
    costs = [float(item["estimated_cost_usd"]) for item in completed if item.get("estimated_cost_usd") is not None]
    tokens = [float(item.get("total_tokens", 0)) for item in completed]
    return {
        "summary": {
            "requests": total,
            "conversations": len({item.get("conversation_id") for item in completed if item.get("conversation_id")}),
            "availability": round(successful / total, 6) if total else None,
            "error_rate": round((total - successful) / total, 6) if total else None,
            "latency_ms": {
                "total": distribution(latencies),
                "retrieval": distribution(stage_values("retrieval")),
                "generation": distribution(stage_values("generation")),
                "reranking": {"p50": None, "p95": None, "p99": None, "status": "not_applicable"},
            },
            "errors_by_type": {
                status: sum(item.get("status") == status for item in completed)
                for status in sorted({str(item.get("status")) for item in completed if item.get("status") in operational_failures})
            },
            "provider_failures": sum(int(item.get("provider_failure_count", 0)) for item in completed),
            "fallbacks": sum(int(item.get("fallback_count", 0)) for item in completed),
            "mean_total_tokens": round(statistics.fmean(tokens), 2) if tokens else None,
            "cost_usd": {
                "mean": round(statistics.fmean(costs), 8) if costs else None,
                **distribution(costs),
            },
            "quality": quality,
        },
        "traces": completed,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def metrics_report(limit: int = 200, conversation_id: str = "") -> dict[str, Any]:
    return build_metrics_report(metrics_repository.read(limit=limit, conversation_id=conversation_id))


BENCHMARK_QUERIES = (
    "How can diabetes complications be prevented?",
    "What role does preventive cardiology have in diabetes care?",
    "What is known about diabetes epidemiology in Egypt?",
)


def run_retrieval_benchmark() -> dict[str, Any]:
    from src.retriever import retrieve
    from src.vector_store import vector_store
    latencies, best_scores, passed = [], [], 0
    for query in BENCHMARK_QUERIES:
        started = time.perf_counter()
        results = retrieve(query, category=CATEGORY_PREVENTION, top_k=3, similarity_threshold=0.0)
        latencies.append((time.perf_counter() - started) * 1000)
        if results and all(item.document_name and item.page_number > 0 for item in results):
            passed += 1
            best_scores.append(results[0].score)
    snapshot = {
        "timestamp": datetime.now(timezone.utc).isoformat(), "model": config.embedding_model,
        "namespace": config.resolved_embedding_namespace, "queries": len(BENCHMARK_QUERIES),
        "pass_rate": round(passed / len(BENCHMARK_QUERIES), 3),
        "mean_best_score": round(statistics.fmean(best_scores), 4) if best_scores else 0.0,
        "mean_latency_ms": round(statistics.fmean(latencies), 2), "max_latency_ms": round(max(latencies), 2),
        "indexed_chunks": sum(vector_store.collection_stats().values()),
    }
    previous = benchmark_store.read()[-1:] or [None]
    snapshot["score_delta"] = round(snapshot["mean_best_score"] - previous[0]["mean_best_score"], 4) if previous[0] else 0.0
    snapshot["latency_delta_ms"] = round(snapshot["mean_latency_ms"] - previous[0]["mean_latency_ms"], 2) if previous[0] else 0.0
    benchmark_store.append(snapshot)
    return snapshot


def diagnostics_markdown() -> str:
    # Keep these aliases patchable for lightweight developer/test diagnostics.
    traces, benchmarks = trace_store.read(), benchmark_store.read()
    latest_trace, latest_benchmark = (traces[-1] if traces else None), (benchmarks[-1] if benchmarks else None)
    lines = ["### System status"]
    if latest_trace:
        lines.extend([
            f"- Last request: **{latest_trace['status']}** in **{latest_trace['total_ms']:.0f} ms**",
            f"- Retrieval: **{latest_trace['retrieval_count']} chunks**, best score **{latest_trace['best_score']:.3f}**",
            f"- Route: `{latest_trace['routed_category'] or 'n/a'}`",
            "- Stages: " + ", ".join(f"`{name}` {duration:.0f} ms" for name, duration in latest_trace["stages_ms"].items()),
        ])
    else:
        lines.append("- No patient request has been traced yet.")
    if latest_benchmark:
        lines.extend([
            "\n### Latest retrieval benchmark",
            f"- Pass rate: **{latest_benchmark['pass_rate']:.0%}** ({latest_benchmark['queries']} fixed queries)",
            f"- Mean best score: **{latest_benchmark['mean_best_score']:.3f}** `{latest_benchmark.get('score_delta', 0):+.3f}`",
            f"- Mean latency: **{latest_benchmark['mean_latency_ms']:.0f} ms** `{latest_benchmark.get('latency_delta_ms', 0):+.0f} ms`",
            f"- Indexed chunks: **{latest_benchmark['indexed_chunks']}**",
        ])
    else:
        lines.extend(["\n### Latest retrieval benchmark", "- No benchmark has been run yet."])
    return "\n".join(lines)
