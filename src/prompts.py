"""Compatibility exports for the centralized grounded response policy."""

from src.response_policy import (
    NO_RESULTS_RESPONSE_AR,
    NO_RESULTS_RESPONSE_EN,
    SYSTEM_PROMPT,
    build_grounded_prompt,
    build_user_prompt,
)

__all__ = [
    "NO_RESULTS_RESPONSE_AR",
    "NO_RESULTS_RESPONSE_EN",
    "SYSTEM_PROMPT",
    "build_grounded_prompt",
    "build_user_prompt",
]
