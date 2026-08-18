#!/usr/bin/env python
"""Safely test configured Gemini and Groq generation providers.

Usage:
    uv run python scripts/test_llm_providers.py
    uv run python scripts/test_llm_providers.py --live --provider groq
    uv run python scripts/test_llm_providers.py --live --provider all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.gemini_errors import classify_gemini_error, gemini_user_message
from src.generator import GeminiGenerator


def run_simulated() -> bool:
    """Confirm provider-neutral missing-key handling remains safe for both services."""
    passed = True
    for provider in ("Gemini", "Groq"):
        error = RuntimeError(f"{provider.upper()}_API_KEY is not set")
        info = classify_gemini_error(error)
        message = gemini_user_message(error, is_arabic=False, scope="generation")
        ok = info.code == "missing_api_key" and bool(message)
        passed = passed and ok
        print(f"{'PASS' if ok else 'FAIL'} simulated {provider} missing key: {info.code}")
    return passed


def run_live(provider: str) -> bool:
    """Send a minimal request directly without printing secrets or raw provider errors."""
    generator = GeminiGenerator()
    try:
        generator._initialise(provider)
        answer = generator._generate_once("Reply with exactly: PROVIDER_TEST_OK", provider)
        ok = bool(answer.strip())
        print(f"{'PASS' if ok else 'FAIL'} live {provider}: {'ok' if ok else 'empty response'}")
        return ok
    except Exception as error:
        info = classify_gemini_error(error)
        message = gemini_user_message(error, is_arabic=False, scope="generation")
        print(f"FAIL live {provider}: {info.code}: {message}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Gemini/Groq provider test script")
    parser.add_argument("--live", action="store_true", help="send a minimal request to the selected provider")
    parser.add_argument(
        "--provider",
        choices=("gemini", "groq", "all"),
        default="all",
        help="provider to probe when --live is supplied",
    )
    args = parser.parse_args()

    passed = run_simulated()
    if args.live:
        providers = ("gemini", "groq") if args.provider == "all" else (args.provider,)
        live_results = [run_live(provider) for provider in providers]
        passed = all(live_results) and passed
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
