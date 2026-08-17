"""Tests for local and hosted embedding providers."""

from types import SimpleNamespace

import pytest

from src.embeddings import EmbeddingModel, _local_dimension


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
