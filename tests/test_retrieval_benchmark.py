"""Phase 2 metric, labeling, and model-selection tests."""

import pytest

from src.retrieval_benchmark import (
    CandidateResult,
    build_review_labels,
    calculate_metrics,
    estimate_embedding_cost_usd,
    load_retrieval_cases,
    require_cross_review,
    select_candidate,
)
from src.retrieval_contracts import EvidenceChunk, RetrievalCase


def _chunk(chunk_id: str = "chunk-1") -> EvidenceChunk:
    return EvidenceChunk(
        chunk_id=chunk_id,
        text="Stable evidence anchor",
        score=0.9,
        distance=0.1,
        document_name="guide.pdf",
        page_number=4,
        section_title="Care",
        subsection_title="",
        category="treatment",
        language="en",
        source_id="guide",
        source_url="https://example.test/guide",
    )


def _cases() -> list[RetrievalCase]:
    return [
        RetrievalCase(
            case_id="positive",
            query="Question",
            language="en",
            category="all",
            expect_evidence=True,
            expected_source_id="guide",
            expected_document_name="guide.pdf",
            expected_page=4,
            expected_section="Care",
            text_anchors=("Stable evidence anchor",),
        ),
        RetrievalCase(
            case_id="negative",
            query="No evidence",
            language="en",
            category="all",
            expect_evidence=False,
        ),
    ]


def test_cases_have_required_phase_2_shape() -> None:
    cases = load_retrieval_cases()

    assert len([case for case in cases if case.expect_evidence]) == 10
    assert len([case for case in cases if not case.expect_evidence]) == 2
    assert {case.language for case in cases} == {"en", "ar"}


def test_metrics_require_reviewed_top_five_labels() -> None:
    rankings = {"positive": [_chunk()], "negative": [_chunk("chunk-2")]}
    with pytest.raises(ValueError, match="unjudged"):
        calculate_metrics(rankings, _cases(), [])


def test_metrics_calculate_precision_hit_recall_and_refusal() -> None:
    rankings = {"positive": [_chunk()], "negative": [_chunk("chunk-2")]}
    labels = [
        {"case_id": "positive", "rank": 1, "relevance": "relevant"},
        {"case_id": "negative", "rank": 1, "relevance": "not_relevant"},
    ]

    metrics = calculate_metrics(rankings, _cases(), labels, k_values=(3,))

    assert metrics["macro_by_k"][3]["macro_precision_at_k"] == 0.3333
    assert metrics["macro_by_k"][3]["hit_at_k"] == 1.0
    assert metrics["macro_by_k"][3]["macro_recall_at_k"] == 1.0
    assert metrics["macro_by_k"][3]["no_evidence_refusal_pass_rate"] == 1.0


def test_review_labels_are_initially_unjudged_with_suggestions() -> None:
    labels = build_review_labels("run-1", {"positive": [_chunk()]}, _cases())

    assert labels[0]["relevance"] == "unjudged"
    assert labels[0]["suggested_relevance"] == "relevant"


def test_selection_uses_cost_then_local_on_near_precision_tie() -> None:
    per_k = {
        3: {"macro_precision_at_k": 0.6, "hit_at_k": 0.8},
        4: {"macro_precision_at_k": 0.6, "hit_at_k": 0.8},
        5: {"macro_precision_at_k": 0.5, "hit_at_k": 0.9},
    }
    local = CandidateResult("balanced", "local", "mini", 0.50, 0.90, 80, 0.0, 1.0, True, per_k)
    gemini = CandidateResult("large", "gemini", "gemini-embedding-2", 0.53, 0.92, 60, 0.01, 1.0, True, per_k)

    winner, runtime_k = select_candidate([gemini, local])

    assert winner.provider == "local"
    assert runtime_k == 3


def test_embedding_cost_is_explicit_and_provider_safe() -> None:
    assert estimate_embedding_cost_usd("local", 1_000_000) == 0.0
    assert estimate_embedding_cost_usd("gemini", 1_000_000) == 0.20
    with pytest.raises(ValueError, match="unsupported"):
        estimate_embedding_cost_usd("other", 10)


def test_cross_review_requires_matching_named_reviewers() -> None:
    label = {
        "case_id": "positive",
        "rank": 1,
        "relevance": "relevant",
        "reviewer_a": "Developer A",
        "reviewer_b": "Developer B",
        "reviewer_a_label": "relevant",
        "reviewer_b_label": "relevant",
    }
    require_cross_review([label])

    label["reviewer_b_label"] = "not_relevant"
    with pytest.raises(ValueError, match="cross-review"):
        require_cross_review([label])
