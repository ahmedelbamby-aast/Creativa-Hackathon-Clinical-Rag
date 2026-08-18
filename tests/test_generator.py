"""Generation provider selection without live external calls."""

from types import SimpleNamespace

import pytest

from src.generator import GeminiGenerator, _is_rate_limited, _is_retryable
from src.gemini_errors import GeminiResponseError


def test_rate_limit_detection_does_not_match_generate() -> None:
    assert _is_rate_limited("401 UNAUTHENTICATED GenerateContent") is False
    assert _is_rate_limited("429 RESOURCE_EXHAUSTED") is True


def test_retryability_covers_transient_gemini_responses_only() -> None:
    assert _is_retryable(RuntimeError("429 RESOURCE_EXHAUSTED")) is True
    assert _is_retryable(RuntimeError("503 unavailable")) is True
    assert _is_retryable(RuntimeError("401 UNAUTHENTICATED")) is False


def test_gemini_response_without_text_maps_safety_and_empty_cases(monkeypatch) -> None:
    generator = GeminiGenerator()
    generator._initialised = True
    monkeypatch.setattr("src.generator.config.generation_provider", "gemini")

    generator._client = SimpleNamespace(models=SimpleNamespace(generate_content=lambda **_: SimpleNamespace(
        text="", candidates=[SimpleNamespace(finish_reason="SAFETY")]
    )))
    with pytest.raises(GeminiResponseError, match="safety_blocked"):
        generator._generate_once("Evidence")

    generator._client = SimpleNamespace(models=SimpleNamespace(generate_content=lambda **_: SimpleNamespace(
        text="", candidates=[]
    )))
    with pytest.raises(GeminiResponseError, match="empty_response"):
        generator._generate_once("Evidence")


def test_exhausted_transient_retries_raise_a_sanitized_error(monkeypatch) -> None:
    generator = GeminiGenerator()
    generator._initialised = True
    monkeypatch.setattr(generator, "_generate_once", lambda _: (_ for _ in ()).throw(RuntimeError("429 RESOURCE_EXHAUSTED")))
    monkeypatch.setattr("src.generator.time.sleep", lambda _: None)

    with pytest.raises(GeminiResponseError, match="rate_limited"):
        generator.generate("Evidence")


def test_vercel_gateway_generation(monkeypatch) -> None:
    calls = []

    class Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="Grounded answer"))]
            )

    generator = GeminiGenerator()
    generator._client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    generator._initialised = True
    monkeypatch.setattr("src.generator.config.generation_provider", "vercel_gateway")
    monkeypatch.setattr("src.generator.config.ai_gateway_model", "google/gemini-2.5-flash")

    assert generator.generate("Evidence prompt") == "Grounded answer"
    assert calls[0]["model"] == "google/gemini-2.5-flash"
    assert [message["role"] for message in calls[0]["messages"]] == ["system", "user"]


def test_gateway_model_requires_provider_prefix() -> None:
    from src.config import AppConfig

    config = AppConfig(generation_provider="vercel_gateway", ai_gateway_model="bad-model")
    try:
        config.validate()
    except ValueError as exc:
        assert "provider/model" in str(exc)
    else:
        raise AssertionError("invalid gateway model was accepted")
