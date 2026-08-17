from __future__ import annotations

import time
from types import SimpleNamespace

from src import observability


def test_request_trace_records_stage_and_total_duration():
    trace = observability.RequestTrace("question", "all")
    with trace.stage("retrieval"):
        time.sleep(0.001)
    trace.retrieval_count = 2
    trace.best_score = 0.75
    trace.finish()

    value = trace.serializable()
    assert value["status"] == "ok"
    assert value["stages_ms"]["retrieval"] > 0
    assert value["total_ms"] >= value["stages_ms"]["retrieval"]
    assert "_started" not in value


def test_jsonl_store_is_bounded_and_ignores_corrupt_lines(tmp_path):
    store = observability.JsonlStore(tmp_path / "history.jsonl", max_records=2)
    store.append({"value": 1})
    store.append({"value": 2})
    store.append({"value": 3})
    assert store.read() == [{"value": 2}, {"value": 3}]
    store.path.write_text('{"value": 4}\nnot-json\n', encoding="utf-8")
    assert store.read() == [{"value": 4}]


def test_retrieval_benchmark_tracks_score_and_latency_deltas(tmp_path, monkeypatch):
    history = observability.JsonlStore(tmp_path / "benchmarks.jsonl")
    monkeypatch.setattr(observability, "benchmark_store", history)
    monkeypatch.setattr(
        "src.retriever.retrieve",
        lambda *args, **kwargs: [SimpleNamespace(document_name="guide.pdf", page_number=1, score=0.8)],
    )
    monkeypatch.setattr("src.vector_store.vector_store.collection_stats", lambda: {"prevention": 10})

    first = observability.run_retrieval_benchmark()
    second = observability.run_retrieval_benchmark()

    assert first["pass_rate"] == second["pass_rate"] == 1.0
    assert first["mean_best_score"] == second["mean_best_score"] == 0.8
    assert first["indexed_chunks"] == 10
    assert len(history.read()) == 2


def test_diagnostics_markdown_reports_trace_and_benchmark(tmp_path, monkeypatch):
    traces = observability.JsonlStore(tmp_path / "traces.jsonl")
    benchmarks = observability.JsonlStore(tmp_path / "benchmarks.jsonl")
    traces.append({
        "status": "ok", "total_ms": 100, "retrieval_count": 3, "best_score": 0.7,
        "routed_category": "prevention", "stages_ms": {"retrieval": 40},
    })
    benchmarks.append({
        "timestamp": "2026-01-01T00:00:00+00:00", "pass_rate": 1.0, "queries": 3,
        "mean_best_score": 0.7, "mean_latency_ms": 40, "score_delta": 0.01,
        "latency_delta_ms": -5, "indexed_chunks": 70,
    })
    monkeypatch.setattr(observability, "trace_store", traces)
    monkeypatch.setattr(observability, "benchmark_store", benchmarks)

    rendered = observability.diagnostics_markdown()
    assert "Last request: **ok**" in rendered
    assert "Pass rate: **100%**" in rendered
    assert "+0.010" in rendered
    assert "-5 ms" in rendered
