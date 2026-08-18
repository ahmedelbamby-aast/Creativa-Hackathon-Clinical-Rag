"""Gemini response classification and UI-safe-message matrix."""

import pytest

from src.gemini_errors import GeminiResponseError, classify_gemini_error, gemini_user_message


@pytest.mark.parametrize(
    ("error", "code", "english_fragment", "arabic_fragment"),
    [
        (RuntimeError("GEMINI_API_KEY is not set"), "missing_api_key", "not configured", "غير مُعدة"),
        (RuntimeError("GROQ_API_KEY is not set"), "missing_api_key", "not configured", "غير مُعدة"),
        (RuntimeError("400 API_KEY_INVALID"), "invalid_api_key", "temporarily unavailable", "غير متاحة"),
        (RuntimeError("401 UNAUTHENTICATED"), "authentication_failed", "temporarily unavailable", "غير متاحة"),
        (RuntimeError("403 PERMISSION_DENIED"), "permission_denied", "temporarily unavailable", "غير متاحة"),
        (RuntimeError("404 model not found"), "model_not_found", "model is unavailable", "نموذج الإجابة"),
        (RuntimeError("429 RESOURCE_EXHAUSTED quota"), "rate_limited", "busy", "مشغولة"),
        (GeminiResponseError("safety_blocked"), "safety_blocked", "can’t help", "لا يمكنني"),
        (RuntimeError("504 request timed out"), "timeout", "took too long", "وقتاً طويلاً"),
        (RuntimeError("503 service unavailable"), "service_unavailable", "temporarily unavailable", "مؤقتاً"),
        (RuntimeError("400 INVALID_ARGUMENT"), "invalid_request", "rephrase", "إعادة صياغته"),
        (GeminiResponseError("empty_response"), "empty_response", "returned no answer", "لم تُرجع"),
        (RuntimeError("unexpected provider failure"), "unknown_error", "temporary problem", "مؤقتة"),
    ],
)
def test_gemini_error_matrix_has_safe_bilingual_messages(error, code, english_fragment, arabic_fragment) -> None:
    assert classify_gemini_error(error).code == code
    english = gemini_user_message(error, is_arabic=False, scope="generation")
    arabic = gemini_user_message(error, is_arabic=True, scope="generation")

    assert english_fragment in english
    assert arabic_fragment in arabic
    assert "API_KEY" not in english and "401" not in english


def test_retrieval_messages_are_search_specific() -> None:
    message = gemini_user_message(
        RuntimeError("429 RESOURCE_EXHAUSTED"), is_arabic=False, scope="retrieval"
    )

    assert message == "The knowledge search is busy. Please try again in a minute."
