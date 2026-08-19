"""Reviewed, deterministic Level-1 RAG evaluation labels and calculations.

This module deliberately does not use fuzzy matching, embeddings, or a judge
model. A production chat is linked to a gold case only by an explicit case ID,
an exact normalized canonical query, or an exact normalized reviewed variant.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter
from functools import lru_cache
from typing import Any, Iterable

from src.config import config

_TOKEN = re.compile(r"\w+", re.UNICODE)
_CASES_PATH = config.project_root / "data" / "retrieval_cases.json"
GOLD_DATASET_FALLBACK_VERSION = "legacy-1"
RETRIEVAL_METRICS = ("hit_rate_at_k", "precision_at_k", "recall_at_k", "reciprocal_rank", "average_precision", "ndcg_at_k")
ANSWER_METRICS = ("exact_match", "token_precision", "token_recall", "token_f1")
TASK_METRIC = "task_success"
ALL_QUALITY_METRICS = RETRIEVAL_METRICS + ANSWER_METRICS + (TASK_METRIC,)


def normalize_text(value: str) -> str:
    """NFKC + casefold + Unicode word-token normalization for EM and F1."""
    return " ".join(_TOKEN.findall(unicodedata.normalize("NFKC", value).casefold()))


def metric(value: float | None, applicable: bool, reason: str = "") -> dict[str, Any]:
    """One explicit shape for a measured zero and an unavailable value."""
    return {"value": value, "applicable": applicable, "reason": reason}


def unavailable_metrics(keys: Iterable[str], reason: str) -> dict[str, dict[str, Any]]:
    return {key: metric(None, False, reason) for key in keys}


def token_overlap(prediction: str, reference: str) -> dict[str, float]:
    """Return multiset token precision, recall, and F1 after normalization."""
    predicted, expected = normalize_text(prediction).split(), normalize_text(reference).split()
    if not predicted and not expected:
        return {"token_precision": 1.0, "token_recall": 1.0, "token_f1": 1.0}
    if not predicted or not expected:
        return {"token_precision": 0.0, "token_recall": 0.0, "token_f1": 0.0}
    expected_counts = {token: expected.count(token) for token in set(expected)}
    overlap = sum(min(count, expected_counts.get(token, 0)) for token, count in {token: predicted.count(token) for token in set(predicted)}.items())
    precision, recall = overlap / len(predicted), overlap / len(expected)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"token_precision": round(precision, 6), "token_recall": round(recall, 6), "token_f1": round(f1, 6)}


def _legacy_relevant_item(case: dict[str, Any]) -> list[dict[str, Any]]:
    if not case.get("expect_evidence") or not (case.get("expected_source_id") or case.get("expected_document_name")):
        return []
    return [{"source_id": case.get("expected_source_id", ""), "document_name": case.get("expected_document_name", ""), "page_number": case.get("expected_page", 0), "chunk_id": "", "relevance_grade": 3}]


def _normalise_case(case: dict[str, Any], dataset_version: str) -> dict[str, Any]:
    """Map old anchor-only records into an explicitly incomplete legacy shape."""
    value = dict(case)
    value["dataset_version"] = str(value.get("dataset_version") or dataset_version)
    value["query_variants"] = list(value.get("query_variants") or [])
    value["expected_status"] = value.get("expected_status") or ("ready" if value.get("expect_evidence") else "out_of_scope")
    value["relevant_items"] = list(value.get("relevant_items") or _legacy_relevant_item(value))
    value["reference_answers"] = list(value.get("reference_answers") or [])
    value["required_claims"] = list(value.get("required_claims") or [])
    value["accepted_aliases"] = list(value.get("accepted_aliases") or [])
    value["task_pass_rules"] = list(value.get("task_pass_rules") or ["expected_status"])
    value["review"] = dict(value.get("review") or {"status": "legacy_unreviewed"})
    return value


@lru_cache(maxsize=1)
def gold_dataset() -> dict[str, Any]:
    if not _CASES_PATH.exists():
        return {"version": GOLD_DATASET_FALLBACK_VERSION, "cases": []}
    raw = json.loads(_CASES_PATH.read_text(encoding="utf-8"))
    version = str(raw.get("dataset_version") or raw.get("version") or GOLD_DATASET_FALLBACK_VERSION)
    return {"version": version, "cases": [_normalise_case(case, version) for case in raw.get("cases", [])]}


def clear_gold_cache() -> None:
    gold_dataset.cache_clear()


def reviewed_case(case: dict[str, Any] | None) -> bool:
    return bool(case and case.get("review", {}).get("status") == "reviewed")


def match_case(query: str, case_id: str = "") -> tuple[dict[str, Any] | None, str]:
    """Conservatively match labels; never infer a near match."""
    cases = gold_dataset()["cases"]
    if case_id:
        found = next((case for case in cases if case.get("case_id") == case_id), None)
        return (found, "explicit_case_id") if found else (None, "unknown_case_id")
    normalized = normalize_text(query)
    found = next((case for case in cases if normalize_text(case.get("query", "")) == normalized), None)
    if found:
        return found, "canonical_query"
    found = next((case for case in cases if normalized in {normalize_text(item) for item in case.get("query_variants", [])}), None)
    return (found, "reviewed_variant") if found else (None, "unlabeled_query")


def find_case(query: str, case_id: str = "") -> dict[str, Any] | None:
    return match_case(query, case_id)[0]


def _chunk_value(chunk: Any, name: str, default: Any = "") -> Any:
    return chunk.get(name, default) if isinstance(chunk, dict) else getattr(chunk, name, default)


def _item_identity(item: dict[str, Any]) -> tuple[str, str, int, str]:
    return (str(item.get("chunk_id", "")), str(item.get("source_id", "")), int(item.get("page_number") or 0), str(item.get("document_name", "")))


def _chunk_identity(chunk: Any) -> tuple[str, str, int, str]:
    return (str(_chunk_value(chunk, "chunk_id")), str(_chunk_value(chunk, "source_id")), int(_chunk_value(chunk, "page_number", 0) or 0), str(_chunk_value(chunk, "document_name")))


def _relevance_grade(
    item: dict[str, Any],
    chunk: Any,
    text_anchors: Iterable[str] = (),
) -> int:
    chunk_id, source_id, page, document = _chunk_identity(chunk)
    expected_chunk = str(item.get("chunk_id", ""))
    grade = int(item.get("relevance_grade", 0) or 0)
    if expected_chunk and chunk_id == expected_chunk:
        return grade
    source_ok = not item.get("source_id") or str(item.get("source_id")) == source_id
    document_ok = not item.get("document_name") or str(item.get("document_name")) == document
    page_value = int(item.get("page_number", 0) or 0)
    location_ok = source_ok and document_ok and (not page_value or page_value == page)
    if not location_ok:
        return 0
    anchors = [normalize_text(anchor) for anchor in text_anchors if normalize_text(anchor)]
    if expected_chunk and anchors:
        text = normalize_text(str(_chunk_value(chunk, "text", "")))
        return grade if any(anchor in text for anchor in anchors) else 0
    return grade if not expected_chunk else 0


def retrieval_metrics(case: dict[str, Any] | None, chunks: Iterable[Any], k: int) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Rank metrics from the complete reviewed relevance pool, with no duplicate counting."""
    if not case:
        return unavailable_metrics(RETRIEVAL_METRICS, "unlabeled_query"), []
    if not case.get("expect_evidence"):
        return unavailable_metrics(RETRIEVAL_METRICS, "negative_case_not_applicable"), []
    if not reviewed_case(case):
        return unavailable_metrics(RETRIEVAL_METRICS, "missing_relevance_judgments"), []
    pool = [item for item in case.get("relevant_items", []) if int(item.get("relevance_grade", 0) or 0) > 0]
    if not pool:
        return unavailable_metrics(RETRIEVAL_METRICS, "missing_relevance_judgments"), []
    unique_pool = {_item_identity(item): item for item in pool}
    ranked, seen, labels = [], set(), []
    for chunk in list(chunks)[:k]:
        identity = _chunk_identity(chunk)
        if identity in seen:
            continue
        seen.add(identity)
        grade = max(
            (
                _relevance_grade(item, chunk, case.get("text_anchors", ()))
                for item in unique_pool.values()
            ),
            default=0,
        )
        ranked.append((chunk, grade))
        labels.append({"rank": len(ranked), "chunk_id": identity[0], "relevance_grade": grade})
    relevant = [grade > 0 for _, grade in ranked]
    denominator = len(unique_pool)
    relevant_count = sum(relevant)
    first = next((index + 1 for index, hit in enumerate(relevant) if hit), 0)
    average_precision = sum(sum(relevant[:index + 1]) / (index + 1) for index, hit in enumerate(relevant) if hit) / denominator
    gains = [grade for _, grade in ranked]
    dcg = sum(gain / math.log2(index + 2) for index, gain in enumerate(gains))
    ideal = sorted((int(item["relevance_grade"]) for item in unique_pool.values()), reverse=True)[:k]
    ideal_dcg = sum(gain / math.log2(index + 2) for index, gain in enumerate(ideal))
    return {
        "hit_rate_at_k": metric(float(bool(relevant_count)), True),
        "precision_at_k": metric(round(relevant_count / k, 6) if k else 0.0, True),
        "recall_at_k": metric(round(relevant_count / denominator, 6), True),
        "reciprocal_rank": metric(round(1 / first, 6) if first else 0.0, True),
        "average_precision": metric(round(average_precision, 6), True),
        "ndcg_at_k": metric(round(dcg / ideal_dcg, 6) if ideal_dcg else 0.0, True),
    }, labels


def answer_metrics(answer: str, case: dict[str, Any] | None, answer_language: str) -> dict[str, dict[str, Any]]:
    if not case:
        return unavailable_metrics(ANSWER_METRICS, "unlabeled_query")
    if not case.get("expect_evidence"):
        return unavailable_metrics(ANSWER_METRICS, "negative_case_not_applicable")
    if not reviewed_case(case):
        return unavailable_metrics(ANSWER_METRICS, "missing_reference_answer")
    if case.get("language") != answer_language:
        return unavailable_metrics(ANSWER_METRICS, "reference_language_mismatch")
    references = [str(item) for item in [*case.get("reference_answers", []), *case.get("accepted_aliases", [])] if str(item).strip()]
    if not references:
        return unavailable_metrics(ANSWER_METRICS, "missing_reference_answer")
    best = max((token_overlap(answer, reference) for reference in references), key=lambda value: value["token_f1"])
    exact = float(any(normalize_text(answer) == normalize_text(reference) for reference in references))
    return {"exact_match": metric(exact, True), **{name: metric(value, True) for name, value in best.items()}}


def task_success(case: dict[str, Any] | None, trace: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not case:
        return metric(None, False, "unlabeled_query"), []
    if not reviewed_case(case):
        return metric(None, False, "missing_required_claims"), []
    results = []
    for name in case.get("task_pass_rules") or ["expected_status"]:
        answer, status = str(trace.get("answer", "")), str(trace.get("status", ""))
        if name == "expected_status":
            # `ready` is the retrieval-envelope contract; a completed answer
            # is recorded as `ok` (or `ok_with_fallback`) in the trace.
            accepted_statuses = {str(case.get("expected_status", ""))}
            if case.get("expected_status") == "ready":
                accepted_statuses.update({"ok", "ok_with_fallback"})
            value = status in accepted_statuses
        elif name == "required_claims_present":
            answer_counts = Counter(normalize_text(answer).split())
            required_claims = list(case.get("required_claims") or [])
            value = bool(required_claims) and all(
                not (Counter(normalize_text(claim).split()) - answer_counts)
                for claim in required_claims
            )
        elif name == "certified_citation_present":
            # The formatted citation list intentionally contains only human
            # readable document/page labels.  Check the preserved structured
            # evidence metadata for the source identity instead of requiring
            # an internal source ID to appear in user-facing text.
            cited_source_ids = {
                str(item.get("source_id", ""))
                for item in trace.get("retrieved_chunks", [])
                if isinstance(item, dict) and item.get("source_id")
            }
            expected_source_ids = {
                str(item.get("source_id", ""))
                for item in case.get("relevant_items", [])
                if item.get("source_id")
            }
            value = bool(trace.get("citations")) and bool(cited_source_ids & expected_source_ids)
        elif name == "generation_not_called": value = trace.get("generation_provider") == "not_called"
        elif name == "retrieval_not_called": value = "retrieval" not in trace.get("stages_ms", {}) or trace.get("retrieval_count", 0) == 0
        elif name == "structured_fields_present": value = bool(answer.strip())
        elif name == "answer_contains_accepted_value": value = any(normalize_text(item) in normalize_text(answer) for item in [*case.get("reference_answers", []), *case.get("accepted_aliases", [])])
        else:
            results.append({"rule": name, "applicable": False, "passed": None, "reason": "unknown_task_pass_rule"}); continue
        results.append({"rule": name, "applicable": True, "passed": bool(value), "reason": ""})
    applicable = [item for item in results if item["applicable"]]
    return (metric(float(all(item["passed"] for item in applicable)), True), results) if applicable else (metric(None, False, "missing_required_claims"), results)


def count_tokens(text: str) -> int:
    return len(_TOKEN.findall(unicodedata.normalize("NFKC", text)))
