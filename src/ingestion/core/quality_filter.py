"""Quality filter for text chunks.

Scores each chunk and optionally drops low-quality fragments
(e.g. chunks that are mostly numbers, whitespace, or OCR garbage).
"""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Characters that are purely structural noise when they dominate a chunk
_NOISE_CHARS_RE = re.compile(r"[|\-=_•·~*#^]{3,}")
_DIGIT_DOMINANT_RE = re.compile(r"^\d[\d\s.,]+$")


def _score_chunk(text: str) -> float:
    """Heuristic quality score in [0.0, 1.0].

    Penalises:
    - Very short chunks (< 30 chars)
    - Chunks dominated by noise characters (table borders, dashes)
    - Chunks that are pure numbers
    - Chunks with very low word density

    Returns:
        Quality score: 1.0 = perfect, 0.0 = garbage.
    """
    if not text or not text.strip():
        return 0.0

    stripped = text.strip()
    length = len(stripped)

    # Very short
    if length < 20:
        return 0.1

    words = stripped.split()
    word_count = len(words)

    # Pure number lines
    if _DIGIT_DOMINANT_RE.match(stripped):
        return 0.15

    # Noise character density
    noise_matches = _NOISE_CHARS_RE.findall(stripped)
    noise_len = sum(len(m) for m in noise_matches)
    noise_ratio = noise_len / length
    if noise_ratio > 0.5:
        return 0.2

    # Low word density (many non-letter chars)
    alpha_chars = sum(1 for c in stripped if c.isalpha())
    alpha_ratio = alpha_chars / length if length > 0 else 0
    if alpha_ratio < 0.3:
        return 0.3 + alpha_ratio

    # Reasonable chunk: score based on word count (caps at 1.0)
    density_score = min(1.0, word_count / 50)
    return 0.5 + density_score * 0.5


def filter_chunks(
    chunks: list[str],
    min_score: float = 0.1,
) -> list[tuple[str, float]]:
    """Score and filter chunks by quality.

    Args:
        chunks: List of chunk texts.
        min_score: Minimum score to keep a chunk.

    Returns:
        List of (chunk_text, score) tuples for chunks that pass the filter.
    """
    results: list[tuple[str, float]] = []
    for chunk in chunks:
        score = _score_chunk(chunk)
        if score >= min_score:
            results.append((chunk, score))
        else:
            logger.debug("Dropped low-quality chunk (score=%.2f): %r...", score, chunk[:60])
    return results
