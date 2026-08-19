from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from backend import server
from src import observability, quality_metrics
from scripts.backfill_metrics import backfill_records
from scripts import review_gold_cases


def reviewed_case(**overrides):
    value = {
        "case_id": "case-1", "query": "What is the value?", "query_variants": ["Tell me the value"],
        "language": "en", "category": "all", "expect_evidence": True, "expected_status": "ready",
        "relevant_items": [
            {"source_id": "source", "document_name": "guide.pdf", "page_number": 1, "chunk_id": "a", "relevance_grade": 3},
            {"source_id": "source", "document_name": "guide.pdf", "page_number": 2, "chunk_id": "b", "relevance_grade": 1},
        ],
        "reference_answers": ["The value is 589 million adults."], "required_claims": ["589 million"],
        "accepted_aliases": ["589 million adults"],
        "task_pass_rules": ["expected_status", "required_claims_present", "certified_citation_present"],
        "review": {"status": "reviewed", "reviewer_role": "domain_expert", "reviewed_at": "2026-01-01T00:00:00Z"},
    }
    value.update(overrides)
    return value


def chunk(chunk_id: str, page: int, *, source="source", text=""):
    return SimpleNamespace(chunk_id=chunk_id, source_id=source, document_name="guide.pdf", page_number=page, text=text,
                           source_url="https://example.test/source", publisher="Example", publication_date="2026",
                           category="all", language="en", score=0.9, distance=0.1, section_title="Section")


def test_match_order_is_explicit_then_canonical_then_variant(monkeypatch):
    case = reviewed_case()
    monkeypatch.setattr(quality_metrics, "gold_dataset", lambda: {"version": "v2", "cases": [case]})
    assert quality_metrics.match_case("other", "case-1")[1] == "explicit_case_id"
    assert quality_metrics.match_case("WHAT is the value?")[1] == "canonical_query"
    assert quality_metrics.match_case("tell me the value")[1] == "reviewed_variant"
    assert quality_metrics.match_case("similar value question")[1] == "unlabeled_query"


def test_retrieval_metrics_use_complete_pool_grades_and_ignore_duplicate_chunks():
    metrics, labels = quality_metrics.retrieval_metrics(reviewed_case(), [chunk("a", 1), chunk("a", 1), chunk("x", 9), chunk("b", 2)], 4)
    assert labels == [
        {"rank": 1, "chunk_id": "a", "relevance_grade": 3},
        {"rank": 2, "chunk_id": "x", "relevance_grade": 0},
        {"rank": 3, "chunk_id": "b", "relevance_grade": 1},
    ]
    assert metrics["hit_rate_at_k"]["value"] == 1.0
    assert metrics["precision_at_k"]["value"] == 0.5
    assert metrics["recall_at_k"]["value"] == 1.0
    assert metrics["reciprocal_rank"]["value"] == 1.0
    assert metrics["average_precision"]["value"] == pytest.approx(0.833333, abs=1e-6)
    assert 0 < metrics["ndcg_at_k"]["value"] < 1


def test_measured_zero_is_not_not_measured():
    metrics, _ = quality_metrics.retrieval_metrics(reviewed_case(), [chunk("x", 9)], 1)
    assert metrics["hit_rate_at_k"] == {"value": 0.0, "applicable": True, "reason": ""}
    missing, _ = quality_metrics.retrieval_metrics(None, [], 5)
    assert missing["hit_rate_at_k"] == {"value": None, "applicable": False, "reason": "unlabeled_query"}


def test_answer_metrics_use_accepted_aliases_and_language():
    values = quality_metrics.answer_metrics("589 million adults", reviewed_case(), "en")
    assert values["exact_match"]["value"] == 1.0
    assert values["token_f1"]["applicable"] is True
    arabic = quality_metrics.answer_metrics("589 مليون بالغ", reviewed_case(language="ar"), "en")
    assert arabic["exact_match"] == {"value": None, "applicable": False, "reason": "reference_language_mismatch"}


def test_task_rules_cover_positive_and_negative_outcomes():
    case = reviewed_case()
    value, rules = quality_metrics.task_success(case, {"status": "ok_with_fallback", "answer": "589 million adults", "citations": "Sources", "retrieved_chunks": [{"source_id": "source"}]})
    assert value["value"] == 1.0 and all(item["passed"] for item in rules)
    value, _ = quality_metrics.task_success(case, {"status": "ok", "answer": "589 million adults", "citations": "Sources", "retrieved_chunks": [{"source_id": "source"}]})
    assert value["value"] == 1.0
    negative = reviewed_case(expect_evidence=False, expected_status="needs_clarification", required_claims=[], relevant_items=[], reference_answers=[], task_pass_rules=["expected_status", "generation_not_called", "retrieval_not_called"])
    value, rules = quality_metrics.task_success(negative, {"status": "needs_clarification", "generation_provider": "not_called", "retrieval_count": 0, "stages_ms": {}})
    assert value["value"] == 1.0 and all(item["passed"] for item in rules)


def test_trace_persists_applicability_case_identity_and_pricing(monkeypatch):
    case = reviewed_case()
    monkeypatch.setattr("src.observability.match_case", lambda *_: (case, "explicit_case_id"))
    monkeypatch.setattr("src.observability.gold_dataset", lambda: {"version": "v2", "cases": [case]})
    trace = observability.RequestTrace("question", "all", requested_case_id="case-1")
    trace.capture_retrieval(SimpleNamespace(routed_category="all", namespace="local", index_manifest_hash="h", chunks=(chunk("a", 1),)))
    trace.capture_generation("The value is 589 million adults.", "extractive", "", citations="source")
    trace.finish("ok")
    record = trace.serializable()
    assert record["label_case_id"] == "case-1"
    assert record["case_match_method"] == "explicit_case_id"
    assert record["gold_dataset_version"] == "v2"
    assert record["quality_metrics"]["hit_rate_at_k"]["applicable"]
    assert record["operational_metrics"]["cost_usd"] == {"value": 0.0, "applicable": True, "reason": ""}


def test_configured_provider_model_price_and_missing_price_are_distinct(monkeypatch):
    monkeypatch.setattr(observability.config, "generation_pricing_usd_per_million", {"groq": {"model-a": {"input": 2.0, "output": 4.0}}})
    monkeypatch.setattr(observability.config, "generation_input_cost_per_million_usd", 0.0)
    monkeypatch.setattr(observability.config, "generation_output_cost_per_million_usd", 0.0)
    trace = observability.RequestTrace("one two", "all")
    trace.capture_generation("three", "groq", "model-a")
    assert trace.estimated_cost_usd == 0.000008
    missing = observability.RequestTrace("one", "all")
    missing.capture_generation("two", "gemini", "unpriced")
    missing.finish()
    assert missing.operational_metrics["cost_usd"] == {"value": None, "applicable": False, "reason": "pricing_not_configured"}


def test_legacy_backfill_preserves_trace_identity_and_does_not_fabricate_labels():
    saved = []
    legacy = {"trace_id": "00000000-0000-0000-0000-000000000111", "timestamp": "2026-01-01T00:00:00+00:00", "query": "unlabeled", "requested_category": "all", "status": "ok", "total_ms": 12}
    result = backfill_records([legacy], dry_run=False, save=saved.append)
    assert result["updated"] == 1 and result["unavailable_metrics"] > 0
    assert saved[0]["trace_id"] == legacy["trace_id"]
    assert saved[0]["timestamp"] == legacy["timestamp"]
    assert saved[0]["quality_metrics"]["exact_match"] == {"value": None, "applicable": False, "reason": "unlabeled_query"}


def test_rehydrated_running_trace_keeps_retrieval_time_in_total_latency():
    started = (datetime.now(timezone.utc) - timedelta(seconds=16)).isoformat()
    trace = observability.RequestTrace.from_record({"trace_id": "00000000-0000-0000-0000-000000000114", "timestamp": started, "query": "q", "requested_category": "all", "status": "running", "stages_ms": {"retrieval": 15_000}, "total_ms": 0})
    trace.finish("ok")
    assert trace.total_ms >= 15_000


def test_review_validator_detects_wrong_language_reference(monkeypatch):
    case = reviewed_case(language="en", reference_answers=["إجابة عربية"])
    monkeypatch.setattr(review_gold_cases, "gold_dataset", lambda: {"version": "v", "cases": [case]})
    monkeypatch.setattr(review_gold_cases, "load_source_catalog", lambda: {})
    assert any(issue["issue"] == "reference_language_content_mismatch" for issue in review_gold_cases.validate())


def test_json_failure_still_attempts_database_write(tmp_path, monkeypatch):
    store = observability.JsonlStore(tmp_path / "history.jsonl")
    repository = observability.MetricsRepository(store, "postgresql://unused")
    attempted = []
    monkeypatch.setattr(store, "append", lambda _: (_ for _ in ()).throw(OSError("disk unavailable")))
    monkeypatch.setattr(repository, "ensure_schema", lambda: attempted.append("schema"))
    repository.save({"trace_id": "00000000-0000-0000-0000-000000000112", "timestamp": "2026-01-01T00:00:00+00:00", "status": "ok", "total_ms": 1, "total_tokens": 0, "stages_ms": {}})
    assert attempted == ["schema"]


def test_postgres_metric_write_is_an_upsert(tmp_path, monkeypatch):
    calls = []

    class Connection:
        def __enter__(self): return self
        def __exit__(self, *_): return False
        def execute(self, query, params=None):
            calls.append((str(query), params))

    repository = observability.MetricsRepository(observability.JsonlStore(tmp_path / "history.jsonl"), "postgresql://test")
    monkeypatch.setattr(observability.psycopg, "connect", lambda *args, **kwargs: Connection())
    repository.save({"trace_id": "00000000-0000-0000-0000-000000000113", "timestamp": "2026-01-01T00:00:00+00:00", "status": "ok", "total_ms": 1, "total_tokens": 0, "stages_ms": {}})
    assert any("ON CONFLICT (trace_id) DO UPDATE" in query for query, _ in calls)


def test_metrics_report_aggregates_new_and_legacy_metric_shapes():
    records = [
        {"status": "ok", "total_ms": 10, "total_tokens": 3, "quality_metrics": {"task_success": {"value": 0.0, "applicable": True, "reason": ""}}},
        {"status": "ok", "total_ms": 20, "total_tokens": 4, "quality_metrics": {"task_success": {"value": None, "applicable": False, "reason": "unlabeled_query"}}},
    ]
    result = observability.build_metrics_report(records)["summary"]["quality"]["task_success"]
    assert result["mean"] == 0.0 and result["measured_count"] == 1 and result["eligible_count"] == 2
    assert result["unavailable_reasons"] == {"unlabeled_query": 1}


def test_public_metrics_api_redacts_chat_and_evidence_content(monkeypatch):
    monkeypatch.setattr(server, "metrics_report", lambda **_: {"summary": {}, "traces": [{"trace_id": "t", "query": "secret", "answer": "secret", "retrieved_chunks": ["secret"], "quality_metrics": {}}]})
    assert server.foundational_metrics()["traces"] == [{"trace_id": "t", "quality_metrics": {}}]
