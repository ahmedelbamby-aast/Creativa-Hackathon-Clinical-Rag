from __future__ import annotations

from types import SimpleNamespace

import pytest

from src import observability, quality_metrics
from backend import server


def _chunk(*, source_id: str = "wrong", page: int = 1, text: str = "noise"):
    return SimpleNamespace(
        chunk_id=f"{source_id}-{page}", source_id=source_id,
        document_name="other.pdf", page_number=page, section_title="",
        score=0.8, text=text,
    )


def test_token_overlap_and_exact_match_normalize_unicode_and_case():
    overlap = quality_metrics.token_overlap("The Answer is 589 million.", "589 million")
    assert overlap == {"token_precision": 0.4, "token_recall": 1.0, "token_f1": 0.571429}
    metrics = quality_metrics.answer_metrics(
        "589 MILLION", {"expect_evidence": True, "text_anchors": ["589 million"]}, "ok"
    )
    assert metrics["exact_match"] == 1.0
    assert metrics["task_success"] == 1.0


def test_retrieval_metrics_match_canonical_ranked_result(monkeypatch):
    case = {
        "case_id": "gold", "expect_evidence": True, "expected_source_id": "gold-source",
        "expected_document_name": "gold.pdf", "expected_page": 9, "text_anchors": ["gold fact"],
    }
    monkeypatch.setattr(quality_metrics, "find_case", lambda _: case)
    ranked = [_chunk(), _chunk(source_id="gold-source", page=9, text="gold fact"), _chunk()]
    metrics, matched = quality_metrics.retrieval_metrics("query", ranked, k=3)
    assert matched == case
    assert metrics["hit_rate_at_k"] == 1.0
    assert metrics["precision_at_k"] == pytest.approx(1 / 3, abs=1e-6)
    assert metrics["recall_at_k"] == 1.0
    assert metrics["reciprocal_rank"] == 0.5
    assert metrics["average_precision"] == 0.5
    assert metrics["ndcg_at_k"] == 0.63093


def test_unlabeled_quality_is_none_not_zero():
    retrieval, case = quality_metrics.retrieval_metrics("not in gold set", [], 5)
    assert case is None
    assert all(value is None for value in retrieval.values())
    assert all(value is None for value in quality_metrics.answer_metrics("answer", None, "ok").values())


def test_metrics_report_separates_policy_outcomes_from_operational_errors():
    records = [
        {"trace_id": "1", "status": "ok", "total_ms": 100, "total_tokens": 10, "conversation_id": "a", "quality_metrics": {"task_success": 1.0}},
        {"trace_id": "2", "status": "out_of_scope", "total_ms": 200, "total_tokens": 5, "conversation_id": "b", "quality_metrics": {"task_success": 1.0}},
        {"trace_id": "3", "status": "generation_error", "total_ms": 300, "total_tokens": 8, "conversation_id": "b", "quality_metrics": {"task_success": 0.0}},
    ]
    report = observability.build_metrics_report(records)["summary"]
    assert report["availability"] == pytest.approx(2 / 3, abs=1e-6)
    assert report["error_rate"] == pytest.approx(1 / 3, abs=1e-6)
    assert report["latency_ms"]["total"] == {"p50": 200.0, "p95": 300.0, "p99": 300.0}
    assert report["latency_ms"]["reranking"]["status"] == "not_applicable"
    assert report["quality"]["task_success"] == {"mean": 0.666667, "measured_count": 3}
    assert report["quality"]["exact_match"] == {"mean": None, "measured_count": 0}


def test_repository_preserves_json_when_database_is_unavailable(tmp_path, monkeypatch):
    store = observability.JsonlStore(tmp_path / "traces.jsonl")
    repository = observability.MetricsRepository(store, "postgresql://unavailable")
    monkeypatch.setattr(observability, "METRICS_SNAPSHOT_FILE", tmp_path / "snapshot.json")
    monkeypatch.setattr(repository, "ensure_schema", lambda: (_ for _ in ()).throw(RuntimeError("offline")))
    record = observability.RequestTrace("question", "all").serializable()
    repository.save(record)
    assert store.read() == [record]
    assert (tmp_path / "snapshot.json").exists()


def test_static_ui_exposes_history_dashboard_and_trace_identity():
    index = (observability.config.project_root / "backend" / "static" / "index.html").read_text(encoding="utf-8")
    assert 'id="metrics-heading"' in index
    assert "loadMetrics()" in index
    assert "conversation_id: state.conversationId" in index
    assert "trace_id: retrieved.trace_id" in index
    assert "Not measured" in index
    assert "Gemini → Groq → Evidence excerpts (automatic)" in index
    assert 'id="theme-toggle"' in index
    assert 'role="region" tabindex="0" aria-label="Scrollable recent metrics table"' in index


def test_database_schema_has_durable_jsonb_metric_events():
    schema = (observability.config.project_root / "database" / "metrics_schema.sql").read_text(encoding="utf-8")
    assert "CREATE TABLE IF NOT EXISTS rag_metric_events" in schema
    assert "payload jsonb NOT NULL" in schema
    assert "conversation_id" in schema


def test_metrics_api_bounds_history_and_forwards_conversation_filter(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        server,
        "metrics_report",
        lambda **kwargs: (captured.update(kwargs) or {"summary": {}, "traces": []}),
    )
    response = server.foundational_metrics(limit=50_000, conversation_id="chat-1")
    assert response == {"summary": {}, "traces": []}
    assert captured == {"limit": 1000, "conversation_id": "chat-1"}


def test_trace_prefers_provider_reported_usage_and_records_fallbacks():
    trace = observability.RequestTrace("question", "all")
    trace.capture_generation(
        "answer", "groq", "model",
        {"input_tokens": 20, "output_tokens": 5, "total_tokens": 25},
        [{"provider": "gemini", "status": "error", "error_code": "timeout"},
         {"provider": "groq", "status": "ok", "error_code": ""}],
    )
    trace.finish("ok_with_fallback")
    assert trace.token_count_method == "provider_reported"
    assert trace.total_tokens == 25
    assert trace.provider_failure_count == 1
    assert trace.fallback_count == 1
