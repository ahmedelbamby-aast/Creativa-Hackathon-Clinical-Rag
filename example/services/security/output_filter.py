"""Output filter for sensitive content removal."""

from typing import Any


def strip_answers(question_content: dict[str, Any]) -> dict[str, Any]:
    """Strip answers, explanations, and scoring criteria from a question.

    Used for generating exam paper PDFs that must not contain any answers.

    Args:
        question_content: Full question content dict.

    Returns:
        Question content with only question-facing fields.
    """
    question_type = question_content.get("question_type", "")

    safe_fields = {
        "mcq": ["question_text", "options", "difficulty", "domain"],
        "true_false": ["statement", "difficulty", "domain"],
        "short_answer": ["question_text", "difficulty", "domain"],
        "essay": ["question_text", "difficulty", "domain"],
    }

    allowed = safe_fields.get(question_type, ["question_text", "difficulty", "domain"])
    return {k: v for k, v in question_content.items() if k in allowed}


def filter_sensitive_fields(data: dict[str, Any], fields_to_remove: list[str]) -> dict[str, Any]:
    """Remove specified sensitive fields from a dict.

    Args:
        data: Original data dict.
        fields_to_remove: List of field names to remove.

    Returns:
        Data dict with sensitive fields removed.
    """
    return {k: v for k, v in data.items() if k not in fields_to_remove}
