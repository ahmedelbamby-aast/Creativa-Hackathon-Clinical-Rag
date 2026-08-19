from src.rollout_acceptance import evaluate_stage


def stage(dimension=768):
    return {
        "dimension": dimension,
        "expected_documents": 12,
        "verified_documents": 12,
        "checksums_match": True,
        "invalid_vector_count": 0,
        "quality": {"hit_rate_at_5": 0.8, "recall_at_5": 0.75, "ndcg_at_5": 0.7, "task_success": 0.9},
        "retrieval_p95_ms": 100,
        "provenance_pass": True,
        "citations_pass": True,
        "refusal_pass": True,
        "preview_logs_clean": True,
    }


def test_stage_accepts_exact_boundary_non_regression() -> None:
    previous = stage(384)
    candidate = stage(768)
    candidate["quality"] = {key: value - 0.02 for key, value in previous["quality"].items()}
    candidate["retrieval_p95_ms"] = 200
    result = evaluate_stage(previous, candidate)
    assert result["accepted"] is True
    assert result["action"] == "promote_exact_preview"


def test_stage_failure_stops_rollout_and_keeps_previous() -> None:
    previous = stage(768)
    candidate = stage(1024)
    candidate["quality"]["ndcg_at_5"] = previous["quality"]["ndcg_at_5"] - 0.021
    candidate["preview_logs_clean"] = False
    result = evaluate_stage(previous, candidate)
    assert result["accepted"] is False
    assert set(result["failed_checks"]) == {"ndcg_at_5", "preview_logs_clean"}
    assert result["action"] == "retain_previous_and_stop"
