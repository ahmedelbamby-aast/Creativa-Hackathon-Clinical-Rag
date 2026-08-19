"""Tests for dependency-free RAG logic."""

import pytest

from src.config import CATEGORY_ALL, CATEGORY_NUTRITION, CATEGORY_TREATMENT
from src.ingestion.category_classifier import classify_chunk
from src.ingestion.core.chunker import SmartChunker
from src.rewriter import rewrite_query
from src.router import route_query
from src.safety import SafetyLevel, classify_safety
from src.scoring import cosine_distance_to_score


@pytest.mark.parametrize(
    ("distance", "expected"),
    [
        (0.0, 1.0),
        (0.25, 0.75),
        (1.0, 0.0),
        (2.0, 0.0),
        (-0.2, 1.0),
    ],
)
def test_cosine_distance_conversion(distance: float, expected: float) -> None:
    assert cosine_distance_to_score(distance) == expected


def test_manual_category_wins() -> None:
    assert route_query("What should I eat?", CATEGORY_TREATMENT) == CATEGORY_TREATMENT


def test_clear_nutrition_query_is_routed() -> None:
    query = "What food, diet, and meal choices help with diabetes?"
    assert route_query(query) == CATEGORY_NUTRITION


def test_general_query_searches_all_collections() -> None:
    assert route_query("What is diabetes?") == CATEGORY_ALL


def test_short_query_gets_diabetes_context() -> None:
    rewritten = rewrite_query("What about fruit?")
    assert "diabetes" in rewritten.lower()


def test_follow_up_uses_previous_question() -> None:
    history = [
        {"role": "user", "content": "Which foods have a low glycemic index?"},
        {"role": "assistant", "content": "The sources list several foods."},
    ]
    rewritten = rewrite_query("What about bread?", history)
    assert "Which foods have a low glycemic index?" in rewritten
    assert "What about bread?" in rewritten


def test_rewriter_adds_bilingual_hints_for_arabic_source_figures() -> None:
    rewritten = rewrite_query(
        "ما نسبة الزيادة من 589 مليون مصاب بالسكري عام 2024 إلى 853 مليوناً عام 2050؟"
    )

    assert "853 million" in rewritten
    assert "percentage increase" in rewritten


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("I have chest pain and can't breathe", SafetyLevel.EMERGENCY),
        ("What exact dose of insulin should I take?", SafetyLevel.HIGH_RISK),
        ("Do I have diabetes?", SafetyLevel.DIAGNOSIS),
        ("What is type 2 diabetes?", SafetyLevel.INFORMATIONAL),
    ],
)
def test_safety_classification(query: str, expected: SafetyLevel) -> None:
    assert classify_safety(query) == expected


def test_chunker_respects_maximum_size() -> None:
    text = "Diabetes management includes regular monitoring. " * 30
    chunks = SmartChunker(max_chunk_size=180, overlap_size=20).chunk(text)
    assert len(chunks) > 1
    assert all(0 < len(chunk) <= 180 for chunk in chunks)


def test_nutrition_content_is_classified() -> None:
    category = classify_chunk(
        document_name="general.pdf",
        section_title="Nutrition and diet",
        subsection_title="Food and meal planning",
        content="Diet, food, meals, carbohydrates, and fibre are discussed.",
        content_type="text",
    )
    assert category == CATEGORY_NUTRITION
