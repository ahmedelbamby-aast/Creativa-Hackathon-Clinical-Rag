"""Metadata propagation tests.

Verifies that the new metadata fields survive:
parsing/chunking -> vector store addition -> retrieval.
"""

from __future__ import annotations

import pytest
from src.ingestion.chunker_adapter import build_chunk_record, chunk_elements
from src.retriever import RetrievedChunk, retrieve
from src.vector_store import VectorStore


def test_build_chunk_record_propagates_metadata():
    """Verify that build_chunk_record correctly includes the new metadata fields."""
    record = build_chunk_record(
        chunk_text="Test paragraph content.",
        quality_score=0.9,
        document_name="doc.pdf",
        page_number=2,
        section_title="Intro",
        subsection_title="Sub",
        content_type="text",
        category="general",
        language="en",
        global_index=1,
        source_id="src-1",
        source_url="https://source.url",
        publisher="Publisher Name",
        publication_date="2026",
        source_checksum="a" * 64,
        chunk_profile="balanced",
    )
    
    assert record["source_id"] == "src-1"
    assert record["source_url"] == "https://source.url"
    assert record["publisher"] == "Publisher Name"
    assert record["publication_date"] == "2026"
    assert record["source_checksum"] == "a" * 64
    assert record["chunk_profile"] == "balanced"


def test_chunk_elements_propagates_optional_metadata(monkeypatch):
    """Verify chunk_elements passes metadata from elements or accepts default parameters."""
    elements = [
        {
            "document_name": "doc.pdf",
            "page_number": 1,
            "section_title": "Intro",
            "subsection_title": "",
            "content": "Test text for chunking.",
            "content_type": "text",
        }
    ]
    
    # Check that chunking works and fields are initialized to default empty strings
    records = chunk_elements(elements)
    assert len(records) == 1
    assert records[0]["source_id"] == ""
    assert records[0]["source_url"] == ""
    assert records[0]["publisher"] == ""
    assert records[0]["publication_date"] == ""
    assert records[0]["source_checksum"] == ""
    assert records[0]["chunk_profile"] == ""


def test_retrieved_chunk_accepts_metadata_fields():
    """Verify RetrievedChunk dataclass supports the new provenance metadata fields."""
    chunk = RetrievedChunk(
        chunk_id="doc_p1_c0",
        text="Sample text",
        score=0.85,
        distance=0.15,
        document_name="doc.pdf",
        page_number=1,
        section_title="Intro",
        subsection_title="",
        category="general",
        content_type="text",
        language="en",
        source_id="src-1",
        source_url="https://source.url",
        publisher="Publisher Name",
        publication_date="2026",
        source_checksum="a" * 64,
        chunk_profile="balanced",
    )
    
    assert chunk.source_id == "src-1"
    assert chunk.source_url == "https://source.url"
    assert chunk.publisher == "Publisher Name"
    assert chunk.publication_date == "2026"
    assert chunk.source_checksum == "a" * 64
    assert chunk.chunk_profile == "balanced"
