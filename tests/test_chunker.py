from __future__ import annotations

import pytest

from src.services.ingestion.chunker import SmartChunker


def test_invalid_overlap_is_rejected():
    with pytest.raises(ValueError, match="overlap_size"):
        SmartChunker(max_chunk_size=50, overlap_size=50)


@pytest.mark.parametrize("text", ["", "   ", "\n\n"])
def test_empty_input_produces_no_chunks(text):
    assert SmartChunker().chunk(text) == []


def test_short_text_is_unchanged():
    assert SmartChunker(max_chunk_size=100, overlap_size=10).chunk("Short diabetes text") == ["Short diabetes text"]


def test_long_text_respects_hard_size_limit_and_has_overlap():
    words = [f"word{i}" for i in range(100)]
    chunks = SmartChunker(max_chunk_size=80, overlap_size=15, min_chunk_size=1).chunk(" ".join(words))
    assert len(chunks) > 2
    assert all(0 < len(chunk) <= 80 for chunk in chunks)
    assert any(set(left.split()) & set(right.split()) for left, right in zip(chunks, chunks[1:]))


def test_character_fallback_for_unbroken_text():
    chunks = SmartChunker(max_chunk_size=30, overlap_size=0, min_chunk_size=1).chunk("x" * 75)
    assert [len(chunk) for chunk in chunks] == [30, 30, 15]
    assert "".join(chunks) == "x" * 75


@pytest.mark.parametrize(
    "block",
    [
        "```python\nvalue = 42\n```",
        "$$x = y + z$$",
        "<table><tr><td>value</td></tr></table>",
        "| A | B |\n|---|---|\n| 1 | 2 |",
    ],
)
def test_semantic_blocks_are_restored_without_corruption(block):
    text = f"Introduction to diabetes. {block} Final clinical note."
    chunks = SmartChunker(max_chunk_size=200, overlap_size=10).chunk(text)
    assert block in " ".join(chunks)
    assert "BLOCK_" not in " ".join(chunks)


def test_tiny_tail_is_merged_when_space_allows():
    chunker = SmartChunker(max_chunk_size=50, overlap_size=0, min_chunk_size=10)
    assert chunker._merge_small_chunks(["a" * 20, "tail"]) == ["a" * 20 + " tail"]

