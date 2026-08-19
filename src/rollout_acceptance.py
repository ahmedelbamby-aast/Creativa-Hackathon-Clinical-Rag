"""Deterministic acceptance gate for one sequential embedding dimension stage."""

from __future__ import annotations

from typing import Any


QUALITY_KEYS = ("hit_rate_at_5", "recall_at_5", "ndcg_at_5", "task_success")


def evaluate_stage(previous: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """Evaluate the documented non-regression and operational rollout gates."""
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, actual: Any, expected: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "actual": actual, "expected": expected})

    expected_documents = int(candidate.get("expected_documents", 12))
    check("document_completeness", candidate.get("verified_documents") == expected_documents,
          candidate.get("verified_documents"), f"exactly {expected_documents}")
    check("document_checksums", bool(candidate.get("checksums_match")),
          candidate.get("checksums_match"), "all expected checksums match")
    check("vector_dimensions", int(candidate.get("invalid_vector_count", -1)) == 0,
          candidate.get("invalid_vector_count"), "0 invalid vectors")

    previous_quality = previous.get("quality", {})
    candidate_quality = candidate.get("quality", {})
    for key in QUALITY_KEYS:
        before = previous_quality.get(key)
        after = candidate_quality.get(key)
        passed = before is not None and after is not None and float(after) >= float(before) - 0.02
        check(key, passed, after, f">= {float(before) - 0.02:.4f}" if before is not None else "previous value required")

    previous_p95 = previous.get("retrieval_p95_ms")
    candidate_p95 = candidate.get("retrieval_p95_ms")
    latency_pass = (
        previous_p95 is not None and candidate_p95 is not None
        and float(candidate_p95) <= 2 * float(previous_p95)
    )
    check("retrieval_p95", latency_pass, candidate_p95,
          f"<= {2 * float(previous_p95):.2f} ms" if previous_p95 is not None else "previous value required")

    for key in ("provenance_pass", "citations_pass", "refusal_pass", "preview_logs_clean"):
        check(key, candidate.get(key) is True, candidate.get(key), "true")

    failed = [item["name"] for item in checks if not item["passed"]]
    return {
        "accepted": not failed,
        "previous_dimension": previous.get("dimension"),
        "candidate_dimension": candidate.get("dimension"),
        "checks": checks,
        "failed_checks": failed,
        "action": "promote_exact_preview" if not failed else "retain_previous_and_stop",
    }
