"""Phase 2 metric, labeling, and model-selection tests."""

import pytest
from pathlib import Path

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
from scripts.benchmark_retrieval import FIELDNAMES, experiment_namespace, finalize, run_grid


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


def test_benchmark_namespaces_are_provider_and_profile_isolated() -> None:
    assert experiment_namespace("small", "local") == "phase2_small_local_384"
    assert experiment_namespace("small", "gemini") == "phase2_small_gemini_384"


def test_finalize_persists_failed_report_for_unresolved_review(tmp_path) -> None:
    import csv
    import json

    with (tmp_path / "review-labels.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerow({"case_id": "case", "rank": 1, "relevance": "unjudged"})

    assert finalize(tmp_path) == 1
    result = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    assert result["selection"]["accepted"] is False
    assert "cross-review" in result["selection"]["error"]


def test_partial_grid_preserves_existing_other_provider_labels(tmp_path, monkeypatch) -> None:
    import csv
    import json
    from types import SimpleNamespace

    existing = {"run_id": "small-local-384", "case_id": "existing", "rank": 1, "relevance": "relevant"}
    with (tmp_path / "review-labels.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerow(existing)

    monkeypatch.setattr("scripts.benchmark_retrieval.load_retrieval_cases", lambda: _cases())
    monkeypatch.setattr("scripts.benchmark_retrieval.CHUNK_PROFILES", {"small": (1200, 0)})

    def fake_run(command, **kwargs):
        output = next(Path(value) for index, value in enumerate(command) if command[index - 1] == "--single-output")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({
            "run_id": "small-gemini-384",
            "rankings": {"positive": [__import__("dataclasses").asdict(_chunk())], "negative": []},
        }), encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("scripts.benchmark_retrieval.subprocess.run", fake_run)
    run_grid(tmp_path, ("gemini",))

    with (tmp_path / "review-labels.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert any(row["run_id"] == "small-local-384" for row in rows)
    assert any(row["run_id"] == "small-gemini-384" for row in rows)
