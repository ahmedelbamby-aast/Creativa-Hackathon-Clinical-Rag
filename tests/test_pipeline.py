from __future__ import annotations

import json
import sys
from datetime import datetime

import pytest


def test_build_chunk_records(pipeline_main):
    records = pipeline_main.build_chunk_records([("one two", 0.7)], "en", 3, start_order=5)
    assert records == [{
        "index": 5,
        "page_number": 3,
        "text": "one two",
        "char_count": 7,
        "word_count": 2,
        "quality_score": 0.7,
        "language": "en",
    }]


@pytest.mark.integration
def test_process_text_file_end_to_end(tmp_path, pipeline_main):
    source = tmp_path / "guide.txt"
    source.write_text(
        "Diabetes prevention and glucose analysis are important. " * 15,
        encoding="utf-8",
    )
    output_dir = tmp_path / "output"
    result = pipeline_main.process_file(
        str(source), str(output_dir), min_quality_score=0.0, max_chunk_size=120, overlap_size=20
    )
    output_path = output_dir / "guide.chunks.json"
    assert output_path.exists()
    assert json.loads(output_path.read_text(encoding="utf-8")) == result
    assert result["chunk_count"] == len(result["chunks"]) > 1
    assert [chunk["index"] for chunk in result["chunks"]] == list(range(result["chunk_count"]))
    assert all(chunk["page_number"] == 1 and chunk["char_count"] <= 120 for chunk in result["chunks"])
    assert datetime.fromisoformat(result["created_at"]).tzinfo is not None


def test_process_file_rejects_failed_extraction(monkeypatch, pipeline_main):
    monkeypatch.setattr(pipeline_main, "extract_pages", lambda _: [])
    with pytest.raises(ValueError, match="Failed to extract"):
        pipeline_main.process_file("missing.txt")


def test_process_file_explains_scanned_document(monkeypatch, pipeline_main):
    monkeypatch.setattr(pipeline_main, "extract_pages", lambda _: [{"page_number": 1, "text": ""}])
    monkeypatch.setattr(
        pipeline_main,
        "clean_pages",
        lambda _: ([], {"chars_before": 0, "chars_after": 0}),
    )
    with pytest.raises(ValueError, match="OCR required"):
        pipeline_main.process_file("scan.pdf")


def test_process_file_rejects_all_filtered_chunks(tmp_path, pipeline_main):
    source = tmp_path / "weak.txt"
    source.write_text("tiny", encoding="utf-8")
    with pytest.raises(ValueError, match="No valid chunks"):
        pipeline_main.process_file(str(source), str(tmp_path / "out"), min_quality_score=1.0)


def test_cli_reports_missing_explicit_file(monkeypatch, pipeline_main):
    monkeypatch.setattr(sys, "argv", ["main.py", "not-there.pdf"])
    with pytest.raises(SystemExit, match="File not found"):
        pipeline_main.main()


def test_cli_forwards_arguments(tmp_path, monkeypatch, pipeline_main):
    source = tmp_path / "input.txt"
    source.write_text("content", encoding="utf-8")
    captured = {}
    monkeypatch.setattr(pipeline_main, "process_file", lambda *args, **kwargs: captured.update(args=args, kwargs=kwargs))
    monkeypatch.setattr(sys, "argv", [
        "main.py", str(source), "--output-dir", str(tmp_path / "out"),
        "--max-chunk-size", "80", "--overlap-size", "10", "--min-quality-score", "0.2",
    ])
    pipeline_main.main()
    assert captured["kwargs"] == {
        "output_dir": str(tmp_path / "out"),
        "min_quality_score": 0.2,
        "max_chunk_size": 80,
        "overlap_size": 10,
    }
