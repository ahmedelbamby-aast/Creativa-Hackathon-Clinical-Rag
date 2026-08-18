"""Generation provider selection without live external calls."""

from types import SimpleNamespace

from src.generator import GeminiGenerator, _is_rate_limited


def test_rate_limit_detection_does_not_match_generate() -> None:
    assert _is_rate_limited("401 UNAUTHENTICATED GenerateContent") is False
    assert _is_rate_limited("429 RESOURCE_EXHAUSTED") is True


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
