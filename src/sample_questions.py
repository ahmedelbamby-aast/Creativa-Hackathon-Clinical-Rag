"""Validated sample-question catalog shared by the API and user interfaces."""

from __future__ import annotations

import json
from pathlib import Path

from src.config import config


DEFAULT_SAMPLE_PATH = config.project_root / "data" / "sample_questions.json"
VALID_SCENARIOS = {
    "direct_success": "ready",
    "clarification_failure": "needs_clarification",
    "unknown_unsupported": "out_of_scope",
    "derived_inference": "ready",
}


def load_sample_questions(path: Path = DEFAULT_SAMPLE_PATH) -> dict:
    """Load and validate the bilingual, balanced sample-question matrix."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    scenarios = payload.get("scenarios", [])
    if {item.get("id") for item in scenarios} != set(VALID_SCENARIOS):
        raise ValueError("sample catalog must contain the four required scenarios")

    seen: set[str] = set()
    for scenario in scenarios:
        if scenario.get("expected_status") != VALID_SCENARIOS[scenario["id"]]:
            raise ValueError(f"invalid expected status for {scenario['id']}")
        questions = scenario.get("questions", [])
        counts = {
            language: sum(item.get("language") == language for item in questions)
            for language in ("en", "ar")
        }
        if counts != {"en": 3, "ar": 3}:
            raise ValueError(f"{scenario['id']} must contain three English and three Arabic questions")
        for question in questions:
            question_id = str(question.get("id", ""))
            if not question_id or question_id in seen:
                raise ValueError("sample question IDs must be non-empty and unique")
            if question.get("category") not in {"all", "treatment", "prevention", "nutrition"}:
                raise ValueError(f"invalid category for {question_id}")
            if not str(question.get("text", "")).strip():
                raise ValueError(f"empty sample question: {question_id}")
            seen.add(question_id)
    return payload


def flat_sample_questions(path: Path = DEFAULT_SAMPLE_PATH) -> list[dict]:
    """Return every question with its scenario expectation attached."""
    flattened: list[dict] = []
    for scenario in load_sample_questions(path)["scenarios"]:
        for question in scenario["questions"]:
            flattened.append(
                {
                    **question,
                    "scenario": scenario["id"],
                    "expected_status": scenario["expected_status"],
                }
            )
    return flattened
