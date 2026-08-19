"""Tests for local and hosted embedding providers."""

from types import SimpleNamespace

import pytest

from src.embeddings import EmbeddingModel, _local_dimension, _retry_delay_seconds


class FakeOnlineModels:
    def __init__(self) -> None:
        self.calls = []

    def embed_content(self, **kwargs):
        self.calls.append(kwargs)
        count = len(kwargs["contents"])
        return SimpleNamespace(
            embeddings=[SimpleNamespace(values=[3.0, 4.0]) for _ in range(count)]
        )


def test_gemini_provider_normalizes_and_preserves_batch_order() -> None:
    models = FakeOnlineModels()
    client = SimpleNamespace(models=models)
    embedder = EmbeddingModel(
        provider="gemini",
        dimension=2,
        online_model_name="gemini-embedding-2",
        online_client=client,
    )

    vectors = embedder.embed_batch(["first", "second"])

    assert vectors == [[0.6, 0.8], [0.6, 0.8]]
    assert len(models.calls) == 1


def test_gemini_provider_splits_corpus_into_bounded_batches(monkeypatch) -> None:
    models = FakeOnlineModels()
    client = SimpleNamespace(models=models)
    embedder = EmbeddingModel(
        provider="gemini",
        dimension=2,
        online_client=client,
    )
    monkeypatch.setattr("src.embeddings.config.online_embedding_batch_size", 2)

    vectors = embedder.embed_batch(["one", "two", "three", "four", "five"])

    assert len(vectors) == 5
    assert [len(call["contents"]) for call in models.calls] == [2, 2, 1]


def test_gemini_query_uses_query_instruction() -> None:
    models = FakeOnlineModels()
    client = SimpleNamespace(models=models)
    embedder = EmbeddingModel(
        provider="gemini",
        dimension=2,
        online_client=client,
    )

    assert embedder.embed_query("What is HbA1c?") == [0.6, 0.8]
    content = models.calls[0]["contents"][0]
    assert "Query:" in content.parts[0].text


def test_unknown_provider_is_rejected() -> None:
    with pytest.raises(ValueError, match="provider"):
        EmbeddingModel(provider="unknown")


def test_current_local_dimension_api_is_preferred() -> None:
    class FakeModel:
        def get_embedding_dimension(self):
            return 384

        def get_sentence_embedding_dimension(self):
            raise AssertionError("deprecated method should not be called")

    assert _local_dimension(FakeModel()) == 384


def test_provider_retry_delay_is_honoured_with_safety_margin() -> None:
    message = "Please retry in 34.478s. retryDelay': '34s'"

    assert _retry_delay_seconds(message, fallback=2.0) == pytest.approx(35.478)
    assert _retry_delay_seconds("temporarily unavailable", fallback=5.0) == 5.0


def test_gemini_ingestion_retry_uses_provider_delay(monkeypatch) -> None:
    class ThrottledModels(FakeOnlineModels):
        def embed_content(self, **kwargs):
            if not self.calls:
                self.calls.append(kwargs)
                raise RuntimeError("429 RESOURCE_EXHAUSTED; retry in 12.5s")
            return super().embed_content(**kwargs)

    models = ThrottledModels()
    embedder = EmbeddingModel(
        provider="gemini",
        dimension=2,
        online_client=SimpleNamespace(models=models),
    )
    sleeps: list[float] = []
    monkeypatch.setattr("src.embeddings.time.sleep", sleeps.append)

    assert embedder.embed_batch(["HbA1c"]) == [[0.6, 0.8]]
    assert sleeps == [13.5]
    assert len(models.calls) == 2


def test_gemini_interactive_query_fails_fast_for_database_fallback(monkeypatch) -> None:
    class ThrottledModels(FakeOnlineModels):
        def embed_content(self, **kwargs):
            self.calls.append(kwargs)
            raise RuntimeError("429 RESOURCE_EXHAUSTED; retry in 60s")

    models = ThrottledModels()
    embedder = EmbeddingModel(
        provider="gemini",
        dimension=2,
        online_client=SimpleNamespace(models=models),
    )
    monkeypatch.setattr(
        "src.embeddings.time.sleep",
        lambda _: (_ for _ in ()).throw(AssertionError("query embedding must not sleep")),
    )

    with pytest.raises(RuntimeError, match="429"):
        embedder.embed_query("HbA1c")
    assert len(models.calls) == 1


def test_gemini_pacing_waits_for_rolling_window(monkeypatch) -> None:
    embedder = EmbeddingModel(
        provider="gemini",
        dimension=2,
        online_client=SimpleNamespace(models=FakeOnlineModels()),
    )
    embedder._online_usage.append((0.0, 2))
    moments = iter((30.0, 61.0))
    sleeps: list[float] = []
    monkeypatch.setattr("src.embeddings.config.online_embedding_rpm", 2)
    monkeypatch.setattr("src.embeddings.time.monotonic", lambda: next(moments))
    monkeypatch.setattr("src.embeddings.time.sleep", sleeps.append)

    embedder._wait_for_online_quota(1)

    assert sleeps == [pytest.approx(30.25)]
