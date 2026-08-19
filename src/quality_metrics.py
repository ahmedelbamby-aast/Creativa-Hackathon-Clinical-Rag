"""Deterministic Level-1 RAG quality metrics for labeled queries.

Production chats normally have no gold answer or relevance judgments.  This
module deliberately returns ``None`` for gold-dependent metrics in that case;
zero is reserved for a measured failure.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from src.config import config


_TOKEN = re.compile(r"\w+", re.UNICODE)
_CASES_PATH = config.project_root / "data" / "retrieval_cases.json"


def normalize_text(value: str) -> str:
    """Apply the documented EM/token-overlap normalization."""
    value = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(_TOKEN.findall(value))


def token_overlap(prediction: str, reference: str) -> dict[str, float]:
    """Return multiset token precision, recall, and F1."""
    predicted = normalize_text(prediction).split()
    expected = normalize_text(reference).split()
    if not predicted and not expected:
        return {"token_precision": 1.0, "token_recall": 1.0, "token_f1": 1.0}
    if not predicted or not expected:
        return {"token_precision": 0.0, "token_recall": 0.0, "token_f1": 0.0}
    predicted_counts = {token: predicted.count(token) for token in set(predicted)}
    expected_counts = {token: expected.count(token) for token in set(expected)}
    overlap = sum(min(count, expected_counts.get(token, 0)) for token, count in predicted_counts.items())
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "token_precision": round(precision, 6),
        "token_recall": round(recall, 6),
        "token_f1": round(f1, 6),
    }


@lru_cache(maxsize=1)
def labeled_cases() -> dict[str, dict[str, Any]]:
    """Index the repository's gold retrieval cases by normalized query."""
    if not _CASES_PATH.exists():
        return {}
    payload = json.loads(_CASES_PATH.read_text(encoding="utf-8"))
    return {normalize_text(item["query"]): item for item in payload.get("cases", [])}


def find_case(query: str) -> dict[str, Any] | None:
    return labeled_cases().get(normalize_text(query))


def _chunk_value(chunk: Any, name: str, default: Any = "") -> Any:
    if isinstance(chunk, dict):
        return chunk.get(name, default)
    return getattr(chunk, name, default)


def _gain(case: dict[str, Any], chunk: Any) -> int:
    """Return graded relevance: exact provenance=3, same source/doc=2, anchor=1."""
    source_id = str(_chunk_value(chunk, "source_id"))
    document = str(_chunk_value(chunk, "document_name"))
    page = int(_chunk_value(chunk, "page_number", 0) or 0)
    text = normalize_text(str(_chunk_value(chunk, "text")))
    expected_source = str(case.get("expected_source_id", ""))
    expected_document = str(case.get("expected_document_name", ""))
    expected_page = int(case.get("expected_page", 0) or 0)
    source_match = bool(expected_source and source_id == expected_source)
    document_match = bool(expected_document and document == expected_document)
    page_match = bool(expected_page and page == expected_page)
    anchor_match = any(normalize_text(anchor) in text for anchor in case.get("text_anchors", []))
    if (source_match or document_match) and page_match:
        return 3
    if source_match or document_match:
        return 2
    return 1 if anchor_match else 0


def retrieval_metrics(query: str, chunks: Iterable[Any], k: int) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Calculate Hit Rate@k, P@k, R@k, RR/AP, and nDCG for a labeled query."""
    case = find_case(query)
    unavailable = {
        "hit_rate_at_k": None,
        "precision_at_k": None,
        "recall_at_k": None,
        "reciprocal_rank": None,
        "average_precision": None,
        "ndcg_at_k": None,
    }
    if case is None or not case.get("expect_evidence", False):
        return unavailable, case
    ranked = list(chunks)[:k]
    gains = [_gain(case, chunk) for chunk in ranked]
    relevant = [gain > 0 for gain in gains]
    relevant_count = sum(relevant)
    first_rank = next((index + 1 for index, value in enumerate(relevant) if value), 0)
    precisions = [
        sum(relevant[: index + 1]) / (index + 1)
        for index, value in enumerate(relevant)
        if value
    ]
    # Each curated case identifies one required target; chunking may yield more
    # than one relevant chunk for the same source/page, so keep AP bounded.
    known_relevant = max(1, relevant_count)
    dcg = sum(gain / math.log2(index + 2) for index, gain in enumerate(gains))
    ideal_dcg = sum(gain / math.log2(index + 2) for index, gain in enumerate(sorted(gains, reverse=True)))
    return {
        "hit_rate_at_k": float(bool(relevant_count)),
        "precision_at_k": round(relevant_count / k, 6) if k else 0.0,
        "recall_at_k": round(relevant_count / known_relevant, 6),
        "reciprocal_rank": round(1 / first_rank, 6) if first_rank else 0.0,
        "average_precision": round(sum(precisions) / known_relevant, 6) if precisions else 0.0,
        "ndcg_at_k": round(dcg / ideal_dcg, 6) if ideal_dcg else 0.0,
    }, case


def answer_metrics(answer: str, case: dict[str, Any] | None, status: str) -> dict[str, Any]:
    """Calculate EM, token overlap, and the task-specific end-to-end pass."""
    unavailable = {
        "exact_match": None,
        "token_precision": None,
        "token_recall": None,
        "token_f1": None,
        "task_success": None,
    }
    if case is None:
        return unavailable
    if not case.get("expect_evidence", False):
        return {**unavailable, "task_success": float(status != "ok")}
    references = [str(value) for value in case.get("text_anchors", []) if str(value).strip()]
    if not references:
        return unavailable
    overlaps = [token_overlap(answer, reference) for reference in references]
    best = max(overlaps, key=lambda value: value["token_f1"])
    normalized_answer = normalize_text(answer)
    exact = float(any(normalized_answer == normalize_text(reference) for reference in references))
    anchors_present = all(normalize_text(reference) in normalized_answer for reference in references)
    return {
        "exact_match": exact,
        **best,
        "task_success": float(status in {"ok", "ok_with_fallback"} and anchors_present),
    }


def count_tokens(text: str) -> int:
    """Return a dependency-free, deterministic token-volume approximation."""
    return len(_TOKEN.findall(unicodedata.normalize("NFKC", text)))
