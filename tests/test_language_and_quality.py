from __future__ import annotations

import pytest

from src.services.ingestion.language_detector import LanguageDetector
from src.services.ingestion.quality_filter import QualityFilter


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", "en"),
        ("1234 !?", "en"),
        ("Diabetes prevention and treatment", "en"),
        ("السكري والوقاية والعلاج", "ar"),
        ("diabetes مرض السكري care", "mixed"),
    ],
)
def test_detect_language(text, expected):
    assert LanguageDetector().detect(text) == expected


def test_detect_document_samples_beginning_middle_and_end():
    detector = LanguageDetector()
    text = ("English " * 30) + ("العربية " * 30) + ("English " * 30)
    assert detector.detect_document(text, sample_size=20) == "en"


def test_quality_score_is_bounded_and_rewards_domain_signals():
    quality = QualityFilter()
    plain = "short readable text"
    technical = "Analysis theorem equation process: glucose = 7.5 and HbA1c 8%. " * 10
    assert quality.calculate_quality_score("") == 0.0
    assert 0.0 <= quality.calculate_quality_score(plain) < quality.calculate_quality_score(technical) <= 1.0


def test_quality_filter_preserves_order_and_threshold():
    quality = QualityFilter()
    chunks = ["a", "Diabetes analysis is supported by equation 7.5% " * 5, ""]
    result = quality.filter(chunks, min_score=0.5)
    assert [text for text, _ in result] == [chunks[1]]
    assert all(score >= 0.5 for _, score in result)
