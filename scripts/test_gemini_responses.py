#!/usr/bin/env python
"""Verify Gemini response handling without exposing credentials or raw provider errors.

Usage:
    uv run python scripts/test_gemini_responses.py
    uv run python scripts/test_gemini_responses.py --live
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import config
from src.embeddings import EmbeddingModel
from src.gemini_errors import GeminiResponseError, classify_gemini_error, gemini_user_message
from src.generator import GeminiGenerator


SIMULATED_CASES = (
    (RuntimeError("GEMINI_API_KEY is not set"), "missing_api_key"),
    (RuntimeError("400 API_KEY_INVALID"), "invalid_api_key"),
    (RuntimeError("401 UNAUTHENTICATED"), "authentication_failed"),
    (RuntimeError("403 PERMISSION_DENIED"), "permission_denied"),
    (RuntimeError("404 model not found"), "model_not_found"),
    (RuntimeError("429 RESOURCE_EXHAUSTED quota"), "rate_limited"),
    (GeminiResponseError("safety_blocked"), "safety_blocked"),
    (RuntimeError("504 request timed out"), "timeout"),
    (RuntimeError("503 service unavailable"), "service_unavailable"),
    (RuntimeError("400 INVALID_ARGUMENT"), "invalid_request"),
    (GeminiResponseError("empty_response"), "empty_response"),
    (RuntimeError("unexpected provider failure"), "unknown_error"),
)


def run_simulated() -> bool:
    """Exercise every known provider response class deterministically."""
    passed = True
    for error, expected in SIMULATED_CASES:
        info = classify_gemini_error(error)
        message = gemini_user_message(error, is_arabic=False, scope="generation")
        ok = info.code == expected and bool(message)
        passed = passed and ok
        print(f"{'PASS' if ok else 'FAIL'} simulated {expected}: ui_message={message!r}")
    return passed


def run_live() -> bool:
    """Run safe, minimal generation and embedding probes against the configured Gemini key."""
    checks = []
    try:
        vector = EmbeddingModel(provider="gemini", dimension=384).embed_query(
            "Gemini response handling embedding probe"
        )
        checks.append(("embedding", len(vector) == 384, "ok"))
    except Exception as error:
        info = classify_gemini_error(error)
        checks.append(("embedding", False, f"{info.code}: {gemini_user_message(error, is_arabic=False, scope='retrieval')}"))

    try:
        answer = GeminiGenerator().generate("Reply with exactly: GEMINI_RESPONSE_TEST_OK")
        checks.append(("generation", bool(answer.strip()), "ok"))
    except Exception as error:
        info = classify_gemini_error(error)
        checks.append(("generation", False, f"{info.code}: {gemini_user_message(error, is_arabic=False, scope='generation')}"))

    for name, ok, detail in checks:
        print(f"{'PASS' if ok else 'FAIL'} live {name}: {detail}")
    return all(ok for _, ok, _ in checks)


def main() -> int:
    parser = argparse.ArgumentParser(description="Gemini response handling test script")
    parser.add_argument("--live", action="store_true", help="also call the configured Gemini API")
    args = parser.parse_args()

    passed = run_simulated()
    if args.live:
        passed = run_live() and passed
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
