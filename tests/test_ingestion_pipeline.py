"""Tests for ingestion control flow without loading external services."""

import sys
import types
import unittest
from unittest.mock import Mock, patch


def _install_chromadb_stub() -> None:
    """Provide import-only ChromaDB types when the package is unavailable."""
    try:
        __import__("chromadb")
        return
    except ImportError:
        pass

    chromadb = types.ModuleType("chromadb")
    chromadb.PersistentClient = object
    chromadb.Collection = object

    chromadb_config = types.ModuleType("chromadb.config")
    chromadb_config.Settings = object

    sys.modules["chromadb"] = chromadb
    sys.modules["chromadb.config"] = chromadb_config


_install_chromadb_stub()

from src.ingestion import pipeline  # noqa: E402


class IngestionPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = Mock()
        self.element = {
            "document_name": "sample.pdf",
            "page_number": 1,
            "section_title": "",
            "subsection_title": "",
            "content": "Useful diabetes guidance.",
            "content_type": "text",
        }
        self.record = {"text": "Useful diabetes guidance.", "category": "general"}

    def test_existing_document_is_skipped_without_force(self) -> None:
        self.store.has_document.return_value = True

        with (
            patch.object(pipeline, "vector_store", self.store),
            patch.object(pipeline, "parse_document") as parse_document,
        ):
            stats = pipeline.ingest_document("sample.pdf", force=False)

        self.assertTrue(stats["skipped"])
        parse_document.assert_not_called()
        self.store.delete_document.assert_not_called()
        self.store.add_chunks.assert_not_called()

    def test_force_replaces_existing_document_after_processing(self) -> None:
        self.store.has_document.return_value = True
        self.store.add_chunks.return_value = {
            "treatment": 1,
            "prevention": 1,
            "nutrition": 1,
        }
        fake_embedder = Mock()
        fake_embedder.embed_batch.return_value = [[0.1, 0.2]]

        with (
            patch.object(pipeline, "vector_store", self.store),
            patch.object(pipeline, "embedder", fake_embedder),
            patch.object(pipeline, "parse_document", return_value=[self.element]),
            patch.object(pipeline, "propagate_section_titles", side_effect=lambda value: value),
            patch.object(pipeline, "detect_document_language", return_value="en"),
            patch.object(pipeline, "chunk_elements", return_value=[self.record]),
        ):
            stats = pipeline.ingest_document("sample.pdf", force=True)

        self.assertIsNone(stats["error"])
        self.assertFalse(stats["skipped"])
        self.store.delete_document.assert_called_once_with("sample.pdf")
        self.store.add_chunks.assert_called_once_with([self.record], [[0.1, 0.2]])


if __name__ == "__main__":
    unittest.main()

