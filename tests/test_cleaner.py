from __future__ import annotations

from pathlib import Path

import pytest
from docx import Document

from src.services.ingestion import cleaner


@pytest.mark.parametrize(
    ("name", "content_type", "extractor"),
    [
        ("report.PDF", "application/pdf", "pypdf"),
        ("report.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "python-docx"),
        ("notes.txt", "text/plain", "plain-text"),
        ("unknown.bin", "text/plain", "plain-text"),
    ],
)
def test_content_type_and_extractor_detection(name, content_type, extractor):
    assert cleaner.detect_content_type(name) == content_type
    assert cleaner.get_extractor_label(name) == extractor


def test_extract_utf8_and_latin1_text(tmp_path):
    utf8 = tmp_path / "utf8.txt"
    utf8.write_text("  diabetes café  ", encoding="utf-8")
    latin1 = tmp_path / "latin1.txt"
    latin1.write_bytes("café".encode("latin-1"))
    assert cleaner.extract_pages(str(utf8)) == [{"page_number": 1, "text": "diabetes café"}]
    assert cleaner.extract_pages(str(latin1)) == [{"page_number": 1, "text": "café"}]


def test_extract_docx(tmp_path):
    path = tmp_path / "sample.docx"
    document = Document()
    document.add_paragraph("Diabetes care")
    document.add_paragraph("   ")
    document.add_paragraph("Second paragraph")
    document.save(path)
    assert cleaner.extract_pages(str(path)) == [
        {"page_number": 1, "text": "Diabetes care\nSecond paragraph"}
    ]


def test_extract_missing_or_empty_file_returns_no_pages(tmp_path):
    assert cleaner.extract_pages(str(tmp_path / "missing.txt")) == []
    empty = tmp_path / "empty.txt"
    empty.write_text("  ", encoding="utf-8")
    assert cleaner.extract_pages(str(empty)) == []


def test_clean_pages_removes_artifacts_and_tracks_statistics():
    pages = [
        {"page_number": index, "text": f"\ufeffCOMMON HEADER\npre-\nvention\u00a0text {index}\n{index}\nsite.example | {index}\n\n\n"}
        for index in range(1, 5)
    ]
    cleaned, stats = cleaner.clean_pages(pages)
    assert [page["page_number"] for page in cleaned] == [1, 2, 3, 4]
    assert [page["text"] for page in cleaned] == [f"prevention text {index}" for index in range(1, 5)]
    assert stats["page_number_lines_removed"] == 4
    assert stats["repeated_boilerplate_lines_removed"] == 4
    assert stats["footer_lines_removed"] == 4
    assert stats["chars_after"] < stats["chars_before"]


def test_extract_text_joins_cleaned_pages(monkeypatch):
    monkeypatch.setattr(cleaner, "extract_pages", lambda _: [
        {"page_number": 1, "text": "first"},
        {"page_number": 2, "text": "second"},
    ])
    assert cleaner.extract_text("ignored.pdf") == "first\n\nsecond"
