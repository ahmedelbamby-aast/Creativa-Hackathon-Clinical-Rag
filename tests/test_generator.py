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
    monkeypatch.setattr("src.generator.config.generation_provider", "gemini")
    monkeypatch.setattr("src.generator.config.generation_fallback_provider", "")
    monkeypatch.setattr(generator, "_generate_once", lambda *_: (_ for _ in ()).throw(RuntimeError("429 RESOURCE_EXHAUSTED")))
    monkeypatch.setattr("src.generator.time.sleep", lambda _: None)

    with pytest.raises(GeminiResponseError, match="rate_limited"):
        generator.generate("Evidence")


def test_interactive_generation_fails_over_without_retry_sleep(monkeypatch) -> None:
    generator = GeminiGenerator()
    calls = []

    def fake_generate(_prompt, provider=None):
        calls.append(provider)
        if provider == "gemini":
            raise RuntimeError("429 RESOURCE_EXHAUSTED")
        return "Grounded Groq answer"

    monkeypatch.setattr(generator, "_initialise", lambda provider: None)
    monkeypatch.setattr(generator, "_generate_once", fake_generate)
    monkeypatch.setattr("src.generator.config.generation_provider", "auto")
    monkeypatch.setattr("src.generator.config.generation_primary_provider", "gemini")
    monkeypatch.setattr("src.generator.config.generation_fallback_provider", "groq")
    monkeypatch.setattr("src.generator.time.sleep", lambda _: (_ for _ in ()).throw(
        AssertionError("interactive failover must not sleep")
    ))

    assert generator.generate("Evidence") == "Grounded Groq answer"
    assert calls == ["gemini", "groq"]


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


def test_groq_generation_uses_openai_compatible_chat_api(monkeypatch) -> None:
    calls = []

    class Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="Groq grounded answer"))]
            )

    generator = GeminiGenerator()
    generator._client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    generator._initialised = True
    monkeypatch.setattr("src.generator.config.generation_provider", "groq")
    monkeypatch.setattr("src.generator.config.groq_model", "openai/gpt-oss-120b")

    assert generator.generate("Evidence prompt") == "Groq grounded answer"
    assert generator.active_provider == "groq"
    assert generator.active_model == "openai/gpt-oss-120b"
    assert calls[0]["model"] == "openai/gpt-oss-120b"


def test_auto_routes_to_groq_when_gemini_fails(monkeypatch) -> None:
    generator = GeminiGenerator()
    calls = []

    def fake_generate(_prompt, provider):
        calls.append(provider)
        if provider == "gemini":
            raise RuntimeError("429 RESOURCE_EXHAUSTED")
        return "Groq fallback answer"

    monkeypatch.setattr(generator, "_generate_with_provider", fake_generate)
    monkeypatch.setattr("src.generator.config.generation_provider", "auto")
    monkeypatch.setattr("src.generator.config.generation_primary_provider", "gemini")
    monkeypatch.setattr("src.generator.config.generation_fallback_provider", "groq")

    assert generator.generate("Evidence prompt") == "Groq fallback answer"
    assert calls == ["gemini", "groq"]
    assert generator.active_provider == "groq"


def test_auto_attempts_gemini_then_groq_before_final_failure(monkeypatch) -> None:
    generator = GeminiGenerator()
    calls = []

    def fake_generate(_prompt, provider):
        calls.append(provider)
        raise RuntimeError("503 provider unavailable")

    monkeypatch.setattr(generator, "_generate_with_provider", fake_generate)
    monkeypatch.setattr("src.generator.config.generation_provider", "auto")
    monkeypatch.setattr("src.generator.config.generation_primary_provider", "gemini")
    monkeypatch.setattr("src.generator.config.generation_fallback_provider", "groq")

    with pytest.raises(RuntimeError, match="503 provider unavailable"):
        generator.generate("Evidence prompt")

    assert calls == ["gemini", "groq"]


def test_auto_does_not_fail_over_safety_block(monkeypatch) -> None:
    generator = GeminiGenerator()
    calls = []

    def fake_generate(_prompt, provider):
        calls.append(provider)
        raise GeminiResponseError("safety_blocked")

    monkeypatch.setattr(generator, "_generate_with_provider", fake_generate)
    monkeypatch.setattr("src.generator.config.generation_provider", "auto")
    monkeypatch.setattr("src.generator.config.generation_primary_provider", "gemini")
    monkeypatch.setattr("src.generator.config.generation_fallback_provider", "groq")

    with pytest.raises(GeminiResponseError, match="safety_blocked"):
        generator.generate("Evidence prompt")
    assert calls == ["gemini"]


def test_gateway_model_requires_provider_prefix() -> None:
    from src.config import AppConfig

    config = AppConfig(generation_provider="vercel_gateway", ai_gateway_model="bad-model")
    try:
        config.validate()
    except ValueError as exc:
        assert "provider/model" in str(exc)
    else:
        raise AssertionError("invalid gateway model was accepted")
