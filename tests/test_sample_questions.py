"""Regression tests for the UI's verified bilingual sample matrix."""

from src.sample_questions import flat_sample_questions, load_sample_questions


def test_sample_questions_cover_each_expected_behavior_equally() -> None:
    catalog = load_sample_questions()
    flattened = flat_sample_questions()

    assert catalog["version"] == 1
    assert len(flattened) == 24
    assert len({item["id"] for item in flattened}) == 24
    for scenario in {item["scenario"] for item in flattened}:
        questions = [item for item in flattened if item["scenario"] == scenario]
        assert sum(item["language"] == "en" for item in questions) == 3
        assert sum(item["language"] == "ar" for item in questions) == 3


def test_sample_questions_have_expected_pipeline_statuses() -> None:
    by_scenario = {}
    for item in flat_sample_questions():
        by_scenario.setdefault(item["scenario"], set()).add(item["expected_status"])

    assert by_scenario == {
        "direct_success": {"ready"},
        "clarification_failure": {"needs_clarification"},
        "unknown_unsupported": {"out_of_scope"},
        "derived_inference": {"ready"},
    }
