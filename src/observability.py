"""Small, local-only tracing and benchmark utilities for developers."""

from __future__ import annotations

import json
import statistics
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from src.config import CATEGORY_PREVENTION, config


RUNTIME_DIR = (
    Path(tempfile.gettempdir()) / "creativa-diabetes"
    if config.is_deployment
    else config.project_root / ".runtime"
)
TRACE_FILE = RUNTIME_DIR / "request_traces.jsonl"
BENCHMARK_FILE = RUNTIME_DIR / "benchmark_history.jsonl"


@dataclass
class RequestTrace:
    query: str
    requested_category: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    stages_ms: dict[str, float] = field(default_factory=dict)
    routed_category: str = ""
    retrieval_count: int = 0
    best_score: float = 0.0
    status: str = "running"
    error: str = ""
    total_ms: float = 0.0
    _started: float = field(default_factory=time.perf_counter, repr=False)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            self.stages_ms[name] = round((time.perf_counter() - started) * 1000, 2)

    def finish(self, status: str = "ok", error: str = "") -> None:
        self.status = status
        self.error = error[:300]
        self.total_ms = round((time.perf_counter() - self._started) * 1000, 2)

    def serializable(self) -> dict:
        value = asdict(self)
        value.pop("_started", None)
        return value


class JsonlStore:
    """Thread-safe append/read store with a bounded history."""

    def __init__(self, path: Path, max_records: int = 200) -> None:
        self.path = path
        self.max_records = max_records
        self._lock = threading.Lock()

    def append(self, record: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            records = self.read()[-(self.max_records - 1):]
            records.append(record)
            payload = "\n".join(json.dumps(item, ensure_ascii=False, sort_keys=True) for item in records)
            self.path.write_text(payload + "\n", encoding="utf-8")

    def read(self) -> list[dict]:
        if not self.path.exists():
            return []
        records = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return records


trace_store = JsonlStore(TRACE_FILE)
benchmark_store = JsonlStore(BENCHMARK_FILE)


def record_trace(trace: RequestTrace) -> None:
    trace_store.append(trace.serializable())


BENCHMARK_QUERIES = (
    "How can diabetes complications be prevented?",
    "What role does preventive cardiology have in diabetes care?",
    "What is known about diabetes epidemiology in Egypt?",
)


def run_retrieval_benchmark() -> dict:
    """Run stable, non-generative retrieval probes and persist the result."""
    from src.retriever import retrieve
    from src.vector_store import vector_store

    latencies = []
    best_scores = []
    passed = 0
    for query in BENCHMARK_QUERIES:
        started = time.perf_counter()
        results = retrieve(query, category=CATEGORY_PREVENTION, top_k=3, similarity_threshold=0.0)
        latencies.append((time.perf_counter() - started) * 1000)
        if results and all(item.document_name and item.page_number > 0 for item in results):
            passed += 1
            best_scores.append(results[0].score)

    snapshot = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": config.embedding_model,
        "namespace": config.resolved_embedding_namespace,
        "queries": len(BENCHMARK_QUERIES),
        "pass_rate": round(passed / len(BENCHMARK_QUERIES), 3),
        "mean_best_score": round(statistics.fmean(best_scores), 4) if best_scores else 0.0,
        "mean_latency_ms": round(statistics.fmean(latencies), 2),
        "max_latency_ms": round(max(latencies), 2),
        "indexed_chunks": sum(vector_store.collection_stats().values()),
    }
    previous = benchmark_store.read()[-1:] or [None]
    if previous[0]:
        snapshot["score_delta"] = round(snapshot["mean_best_score"] - previous[0]["mean_best_score"], 4)
        snapshot["latency_delta_ms"] = round(snapshot["mean_latency_ms"] - previous[0]["mean_latency_ms"], 2)
    else:
        snapshot["score_delta"] = 0.0
        snapshot["latency_delta_ms"] = 0.0
    benchmark_store.append(snapshot)
    return snapshot


def diagnostics_markdown() -> str:
    traces = trace_store.read()
    benchmarks = benchmark_store.read()
    latest_trace = traces[-1] if traces else None
    latest_benchmark = benchmarks[-1] if benchmarks else None
    lines = ["### System status"]
    if latest_trace:
        lines.extend([
            f"- Last request: **{latest_trace['status']}** in **{latest_trace['total_ms']:.0f} ms**",
            f"- Retrieval: **{latest_trace['retrieval_count']} chunks**, best score **{latest_trace['best_score']:.3f}**",
            f"- Route: `{latest_trace['routed_category'] or 'n/a'}`",
            "- Stages: " + ", ".join(
                f"`{name}` {duration:.0f} ms" for name, duration in latest_trace["stages_ms"].items()
            ),
        ])
    else:
        lines.append("- No patient request has been traced yet.")
    if latest_benchmark:
        score_arrow = "↑" if latest_benchmark["score_delta"] > 0 else "↓" if latest_benchmark["score_delta"] < 0 else "→"
        latency_arrow = "↓" if latest_benchmark["latency_delta_ms"] < 0 else "↑" if latest_benchmark["latency_delta_ms"] > 0 else "→"
        lines.extend([
            "\n### Latest retrieval benchmark",
            f"- Pass rate: **{latest_benchmark['pass_rate']:.0%}** ({latest_benchmark['queries']} fixed queries)",
            f"- Mean best score: **{latest_benchmark['mean_best_score']:.3f}** {score_arrow} `{latest_benchmark['score_delta']:+.3f}`",
            f"- Mean latency: **{latest_benchmark['mean_latency_ms']:.0f} ms** {latency_arrow} `{latest_benchmark['latency_delta_ms']:+.0f} ms`",
            f"- Indexed chunks: **{latest_benchmark['indexed_chunks']}**",
            "\n| Run (UTC) | Pass | Score | Mean latency | Δ score | Δ latency |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for item in reversed(benchmarks[-5:]):
            lines.append(
                f"| {item['timestamp'][0:19]} | {item['pass_rate']:.0%} | "
                f"{item['mean_best_score']:.3f} | {item['mean_latency_ms']:.0f} ms | "
                f"{item.get('score_delta', 0):+.3f} | {item.get('latency_delta_ms', 0):+.0f} ms |"
            )
    else:
        lines.extend(["\n### Latest retrieval benchmark", "- No benchmark has been run yet."])
    return "\n".join(lines)
