"""Tests for page-scoped semantic aggregation before embedding."""

from src.ingestion.chunker_adapter import chunk_elements


def _element(page: int, content: str, content_type: str = "text") -> dict:
    return {
        "document_name": "WHO_Test.pdf",
        "page_number": page,
        "section_title": "Diabetes care",
        "subsection_title": "Prevention",
        "content": content,
        "content_type": content_type,
    }


def test_adjacent_blocks_merge_within_page_but_not_across_pages() -> None:
    elements = [
        _element(1, "Diabetes prevention guidance includes healthy food choices."),
        _element(1, "Regular activity and clinical follow-up reduce diabetes risk."),
        _element(2, "Blood glucose monitoring supports safe diabetes care."),
    ]

    chunks = chunk_elements(
        elements,
        document_language="en",
        chunk_size=500,
        chunk_overlap=50,
        min_chunk_size=10,
        min_quality_score=0.0,
    )

    assert [chunk["page_number"] for chunk in chunks] == [1, 2]
    assert "healthy food choices" in chunks[0]["text"]
    assert "clinical follow-up" in chunks[0]["text"]


def test_table_is_a_standalone_semantic_unit() -> None:
    elements = [
        _element(1, "Diabetes treatment guidance before the table."),
        _element(1, "| Measure | Target |\n| --- | --- |\n| HbA1c | Individualized |", "table"),
        _element(1, "Diabetes treatment guidance after the table."),
    ]

    chunks = chunk_elements(
        elements,
        document_language="en",
        chunk_size=500,
        chunk_overlap=50,
        min_chunk_size=10,
        min_quality_score=0.0,
    )

    assert [chunk["content_type"] for chunk in chunks] == ["text", "table", "text"]
