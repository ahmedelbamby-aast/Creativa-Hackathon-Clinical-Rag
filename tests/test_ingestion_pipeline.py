"""Tests for ingestion control flow without external services."""

from unittest.mock import Mock

from src.ingestion import pipeline


def test_existing_document_is_skipped_without_force(monkeypatch) -> None:
    store = Mock()
    store.has_document.return_value = True
    parse_document = Mock()

    monkeypatch.setattr(pipeline, "vector_store", store)
    monkeypatch.setattr(pipeline, "parse_document", parse_document)

    stats = pipeline.ingest_document("sample.pdf", force=False)

    assert stats["skipped"] is True
    parse_document.assert_not_called()
    store.delete_document.assert_not_called()
    store.add_chunks.assert_not_called()


def test_force_replaces_existing_document_after_processing(monkeypatch) -> None:
    store = Mock()
    store.has_document.return_value = True
    store.add_chunks.return_value = {
        "treatment": 1,
        "prevention": 1,
        "nutrition": 1,
    }
    fake_embedder = Mock()
    fake_embedder.embed_batch.return_value = [[0.1, 0.2]]
    element = {
        "document_name": "sample.pdf",
        "page_number": 1,
        "section_title": "",
        "subsection_title": "",
        "content": "Useful diabetes guidance.",
        "content_type": "text",
    }
    record = {"text": "Useful diabetes guidance.", "category": "general"}

    monkeypatch.setattr(pipeline, "vector_store", store)
    monkeypatch.setattr(pipeline, "embedder", fake_embedder)
    monkeypatch.setattr(pipeline, "parse_document", lambda _: [element])
    monkeypatch.setattr(pipeline, "propagate_section_titles", lambda value: value)
    monkeypatch.setattr(pipeline, "detect_document_language", lambda _: "en")
    monkeypatch.setattr(pipeline, "chunk_elements", lambda *args, **kwargs: [record])

    stats = pipeline.ingest_document("sample.pdf", force=True)

    assert stats["error"] is None
    assert stats["skipped"] is False
    store.delete_document.assert_called_once_with("sample.pdf")
    store.add_chunks.assert_called_once_with([record], [[0.1, 0.2]])
