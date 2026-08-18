"""Focused tests for PDF table extraction in the authoritative Ramez parser."""

from __future__ import annotations

import sys
from types import SimpleNamespace

from src.ingestion import parser


class FakeTable:
    bbox = (10, 50, 190, 120)

    def __init__(self, rows):
        self.rows = rows

    def extract(self):
        return self.rows


def test_table_markdown_normalizes_sparse_ragged_and_multiline_cells():
    table = FakeTable([
        [" Group\nname ", "HbA1c"],
        [None, None],
        ["Treatment | intensive", " 6.4 ", "p < 0.05"],
    ])
    assert parser._table_to_markdown(table) == (
        "| Group name | HbA1c |  |\n"
        "| --- | --- | --- |\n"
        r"| Treatment \| intensive | 6.4 | p < 0.05 |"
    )


def test_table_markdown_handles_empty_and_failed_extraction():
    assert parser._table_to_markdown(FakeTable([])) == ""

    class BrokenTable:
        def extract(self):
            raise RuntimeError("damaged table")

    assert parser._table_to_markdown(BrokenTable()) == ""


def test_pdf_parser_supports_tablefinder_and_avoids_duplicate_table_text(monkeypatch):
    table = FakeTable([["Group", "HbA1c"], ["Treatment", "6.4"]])
    table_block = {
        "type": 0,
        "bbox": (10, 50, 190, 120),
        "lines": [{"spans": [{"text": "Group HbA1c Treatment 6.4", "size": 10, "flags": 0}]}],
    }
    prose_block = {
        "type": 0,
        "bbox": (10, 150, 190, 180),
        "lines": [{"spans": [{"text": "Clinical follow-up text", "size": 10, "flags": 0}]}],
    }

    class Page:
        def get_text(self, mode="text"):
            assert mode == "dict"
            return {"blocks": [table_block, prose_block]}

        def find_tables(self):
            # Deliberately non-iterable: this is the modern PyMuPDF contract.
            return SimpleNamespace(tables=[table])

    class Document:
        page_count = 1

        def __getitem__(self, index):
            assert index == 0
            return Page()

        def close(self):
            pass

    monkeypatch.setitem(sys.modules, "pymupdf", SimpleNamespace(open=lambda _: Document()))
    elements = parser._extract_with_fitz("study.pdf")

    assert [element["content_type"] for element in elements] == ["table", "text"]
    assert elements[0]["content"] == "| Group | HbA1c |\n| --- | --- |\n| Treatment | 6.4 |"
    assert elements[1]["content"] == "Clinical follow-up text"
    assert sum("Treatment 6.4" in element["content"] for element in elements) == 0


def test_ocr_extracts_image_only_pages(monkeypatch):
    class Page:
        def get_textpage_ocr(self, **kwargs):
            assert kwargs == {"language": "eng", "dpi": 150, "full": True}
            return "ocr-text-page"

        def get_text(self, *, textpage):
            assert textpage == "ocr-text-page"
            return "OCR diabetes guidance"

    class Document:
        page_count = 1

        def __getitem__(self, index):
            assert index == 0
            return Page()

        def close(self):
            pass

    monkeypatch.setattr(parser, "_configure_tessdata", lambda: True)
    monkeypatch.setitem(sys.modules, "pymupdf", SimpleNamespace(open=lambda _: Document()))

    assert parser._extract_with_ocr("scan.pdf") == [
        {
            "document_name": "scan.pdf",
            "page_number": 1,
            "section_title": "",
            "subsection_title": "",
            "content": "OCR diabetes guidance",
            "content_type": "text",
        }
    ]


def test_pdf_parser_uses_ocr_before_plain_text_fallback(monkeypatch):
    ocr_elements = [{"content": "from OCR"}]
    fallback = SimpleNamespace(called=False)
    monkeypatch.setattr(parser, "_extract_with_fitz", lambda _: [])
    monkeypatch.setattr(parser, "_extract_with_ocr", lambda _: ocr_elements)
    monkeypatch.setattr(
        parser,
        "_extract_with_pypdf",
        lambda _: setattr(fallback, "called", True) or [],
    )

    assert parser.parse_document("scan.pdf") is ocr_elements
    assert fallback.called is False
