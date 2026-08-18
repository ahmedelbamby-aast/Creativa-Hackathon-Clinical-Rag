"""Classify Gemini failures into safe UI messages and compact trace codes."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal


GeminiScope = Literal["generation", "retrieval"]


@dataclass(frozen=True)
class GeminiErrorInfo:
    """A non-sensitive classification of a Gemini API failure."""

    code: str
    http_status: int | None
    retryable: bool


class GeminiResponseError(RuntimeError):
    """Provider response was structurally valid but cannot be used as an answer."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.gemini_code = code


def _status_from_error(error: BaseException, message: str) -> int | None:
    value = getattr(error, "status_code", None)
    if isinstance(value, int):
        return value
    match = re.search(r"\b(400|401|403|404|408|409|429|500|502|503|504)\b", message)
    return int(match.group(1)) if match else None


def classify_gemini_error(error: BaseException) -> GeminiErrorInfo:
    """Map Google API and transport errors without retaining raw provider details."""
    code = getattr(error, "gemini_code", "")
    if code:
        return GeminiErrorInfo(code=code, http_status=None, retryable=False)

    source = error.__cause__ or error
    message = str(source).lower()
    status = _status_from_error(source, message)
    if "api_key_invalid" in message or "api key not valid" in message:
        return GeminiErrorInfo("invalid_api_key", status or 400, False)
    if status == 401 or "unauthenticated" in message:
        return GeminiErrorInfo("authentication_failed", status or 401, False)
    if status == 403 or "permission_denied" in message or "permission denied" in message:
        return GeminiErrorInfo("permission_denied", status or 403, False)
    if status == 404 or "not_found" in message or "model not found" in message:
        return GeminiErrorInfo("model_not_found", status or 404, False)
    if status == 429 or any(token in message for token in ("resource_exhausted", "rate limit", "quota")):
        return GeminiErrorInfo("rate_limited", status or 429, True)
    if "safety" in message or "blocked" in message:
        return GeminiErrorInfo("safety_blocked", status, False)
    if status in {408, 504} or "timeout" in message or "timed out" in message:
        return GeminiErrorInfo("timeout", status, True)
    if status in {500, 502, 503} or "unavailable" in message or "connection" in message:
        return GeminiErrorInfo("service_unavailable", status, True)
    if status == 400 or "invalid_argument" in message or "bad request" in message:
        return GeminiErrorInfo("invalid_request", status or 400, False)
    if "gemini_api_key is not set" in message or "gemini_api_key is required" in message:
        return GeminiErrorInfo("missing_api_key", status, False)
    return GeminiErrorInfo("unknown_error", status, False)


def is_retryable_gemini_error(error: BaseException) -> bool:
    return classify_gemini_error(error).retryable


def gemini_user_message(error: BaseException, *, is_arabic: bool, scope: GeminiScope) -> str:
    """Return a concise bilingual, non-diagnostic user message for every known case."""
    code = classify_gemini_error(error).code
    messages = {
        "missing_api_key": (
            "خدمة الذكاء الاصطناعي غير مُعدة حالياً. يرجى المحاولة لاحقاً.",
            "The AI service is not configured yet. Please try again later.",
        ),
        "invalid_api_key": (
            "خدمة الذكاء الاصطناعي غير متاحة حالياً. يرجى المحاولة لاحقاً.",
            "The AI service is temporarily unavailable. Please try again later.",
        ),
        "authentication_failed": (
            "خدمة الذكاء الاصطناعي غير متاحة حالياً. يرجى المحاولة لاحقاً.",
            "The AI service is temporarily unavailable. Please try again later.",
        ),
        "permission_denied": (
            "خدمة الذكاء الاصطناعي غير متاحة حالياً. يرجى المحاولة لاحقاً.",
            "The AI service is temporarily unavailable. Please try again later.",
        ),
        "model_not_found": (
            "نموذج الإجابة غير متاح حالياً. يرجى المحاولة لاحقاً.",
            "The answer model is unavailable right now. Please try again later.",
        ),
        "rate_limited": (
            "خدمة الذكاء الاصطناعي مشغولة. يرجى المحاولة بعد دقيقة.",
            "The AI service is busy. Please try again in a minute.",
        ),
        "safety_blocked": (
            "لا يمكنني المساعدة في هذا الطلب.",
            "I can’t help with that request.",
        ),
        "timeout": (
            "استغرقت خدمة الذكاء الاصطناعي وقتاً طويلاً. يرجى المحاولة مجدداً.",
            "The AI service took too long. Please try again.",
        ),
        "service_unavailable": (
            "خدمة الذكاء الاصطناعي غير متاحة مؤقتاً. يرجى المحاولة لاحقاً.",
            "The AI service is temporarily unavailable. Please try again later.",
        ),
        "invalid_request": (
            "تعذر معالجة هذا الطلب. يرجى إعادة صياغته.",
            "I couldn’t process that request. Please rephrase it.",
        ),
        "empty_response": (
            "لم تُرجع خدمة الذكاء الاصطناعي إجابة. يرجى المحاولة مجدداً.",
            "The AI service returned no answer. Please try again.",
        ),
        "unknown_error": (
            "حدثت مشكلة مؤقتة. يرجى المحاولة لاحقاً.",
            "A temporary problem occurred. Please try again later.",
        ),
    }
    arabic, english = messages[code]
    if scope == "retrieval" and code not in {"safety_blocked", "invalid_request"}:
        arabic = arabic.replace("خدمة الذكاء الاصطناعي", "خدمة البحث")
        english = english.replace("AI service", "knowledge search")
    return arabic if is_arabic else english
