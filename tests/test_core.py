"""Tests for dependency-free RAG logic."""

import unittest

from src.config import CATEGORY_ALL, CATEGORY_NUTRITION, CATEGORY_TREATMENT
from src.ingestion.category_classifier import classify_chunk
from src.ingestion.core.chunker import SmartChunker
from src.rewriter import rewrite_query
from src.router import route_query
from src.safety import SafetyLevel, classify_safety
from src.scoring import cosine_distance_to_score


class ScoringTests(unittest.TestCase):
    def test_cosine_distance_conversion(self) -> None:
        cases = {
            0.0: 1.0,
            0.25: 0.75,
            1.0: 0.0,
            2.0: 0.0,
            -0.2: 1.0,
        }
        for distance, expected in cases.items():
            with self.subTest(distance=distance):
                self.assertEqual(cosine_distance_to_score(distance), expected)


class RoutingTests(unittest.TestCase):
    def test_manual_category_wins(self) -> None:
        self.assertEqual(
            route_query("What should I eat?", CATEGORY_TREATMENT),
            CATEGORY_TREATMENT,
        )

    def test_clear_nutrition_query_is_routed(self) -> None:
        query = "What food, diet, and meal choices help with diabetes?"
        self.assertEqual(route_query(query), CATEGORY_NUTRITION)

    def test_general_query_searches_all_collections(self) -> None:
        self.assertEqual(route_query("What is diabetes?"), CATEGORY_ALL)


class RewriterTests(unittest.TestCase):
    def test_short_query_gets_diabetes_context(self) -> None:
        rewritten = rewrite_query("What about fruit?")
        self.assertIn("diabetes", rewritten.lower())

    def test_follow_up_uses_previous_question(self) -> None:
        history = [
            {"role": "user", "content": "Which foods have a low glycemic index?"},
            {"role": "assistant", "content": "The sources list several foods."},
        ]
        rewritten = rewrite_query("What about bread?", history)
        self.assertIn("Which foods have a low glycemic index?", rewritten)
        self.assertIn("What about bread?", rewritten)


class SafetyTests(unittest.TestCase):
    def test_emergency_query(self) -> None:
        self.assertEqual(
            classify_safety("I have chest pain and can't breathe"),
            SafetyLevel.EMERGENCY,
        )

    def test_dose_query_is_high_risk(self) -> None:
        self.assertEqual(
            classify_safety("What exact dose of insulin should I take?"),
            SafetyLevel.HIGH_RISK,
        )

    def test_personal_diagnosis_query(self) -> None:
        self.assertEqual(
            classify_safety("Do I have diabetes?"),
            SafetyLevel.DIAGNOSIS,
        )

    def test_informational_query(self) -> None:
        self.assertEqual(
            classify_safety("What is type 2 diabetes?"),
            SafetyLevel.INFORMATIONAL,
        )


class IngestionLogicTests(unittest.TestCase):
    def test_chunker_respects_maximum_size(self) -> None:
        text = "Diabetes management includes regular monitoring. " * 30
        chunks = SmartChunker(max_chunk_size=180, overlap_size=20).chunk(text)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(0 < len(chunk) <= 180 for chunk in chunks))

    def test_nutrition_content_is_classified(self) -> None:
        category = classify_chunk(
            document_name="general.pdf",
            section_title="Nutrition and diet",
            subsection_title="Food and meal planning",
            content="Diet, food, meals, carbohydrates, and fibre are discussed.",
            content_type="text",
        )
        self.assertEqual(category, CATEGORY_NUTRITION)


if __name__ == "__main__":
    unittest.main()

