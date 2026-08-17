"""Smart Chunker — boundary-aware, language-safe text chunking.

Copied from the existing chunking pipeline (chunking/src/services/ingestion/chunker.py)
and adapted for the RAG ingestion layer. No logic changes — only import paths differ.

Splitting logic
---------------
Uses a greedy forward-merge strategy with a separator hierarchy:
1. Text is split at the largest natural boundaries first (paragraph breaks → line breaks
   → sentence-ending punctuation → clause separators → spaces → character fallback).
2. Segments are merged greedily into chunks <= max_chunk_size.
3. Tables, code fences, LaTeX math and HTML tables are protected from splitting.
4. Overlap is snapped backward to a word boundary (never mid-word).
"""

import re

# ---------------------------------------------------------------------------
# Separator hierarchy  (largest / most preferred -> smallest / last-resort)
# ---------------------------------------------------------------------------
_SEPARATORS: list[tuple[str, bool]] = [
    ("\n\n\n", False),   # Major section break
    ("\n\n",   False),   # Paragraph break
    ("\n",     False),   # Line break
    (". ",     True),    # English sentence end
    (".\n",    True),
    ("! ",     True),
    ("!\n",    True),
    ("? ",     True),
    ("?\n",    True),
    ("\u061f ", True),   # Arabic question mark  ؟
    ("\u061f\n", True),
    ("\u061b ", True),   # Arabic semicolon      ؛
    ("\u060c ", True),   # Arabic comma          ،
    (", ",     True),    # English comma
    ("; ",     True),
    (" ",      False),   # Word boundary (last safe level)
]


class SmartChunker:
    """Split text into clean, ordered, meaningful chunks.

    Never cuts a word in the middle; avoids cutting sentences whenever a
    coarser boundary is available. Handles English and Arabic text.

    Args:
        max_chunk_size: Hard upper limit on chunk length (characters).
        overlap_size:   Characters of context copied from the tail of the
                        previous chunk (must be < max_chunk_size).
        min_chunk_size: Chunks shorter than this are merged with a neighbour.
    """

    BLOCK_PATTERNS: list[tuple[str, str]] = [
        (r"```[\s\S]*?```",              "code"),
        (r"\$\$[\s\S]*?\$\$",            "math"),
        (r"<table[\s\S]*?</table>",      "table_html"),
        (r"\|[^\n]+\|(?:\n\|[^\n]+\|)+", "table_md"),
    ]

    def __init__(
        self,
        max_chunk_size: int = 2000,
        overlap_size: int = 200,
        min_chunk_size: int = 100,
    ) -> None:
        if overlap_size >= max_chunk_size:
            raise ValueError("overlap_size must be < max_chunk_size")
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size
        self.min_chunk_size = min_chunk_size

    # ------------------------------------------------------------------
    # Block protection
    # ------------------------------------------------------------------

    def _protect_blocks(self, text: str) -> tuple[str, list[tuple[str, str]]]:
        protected: list[tuple[str, str]] = []
        result = text
        for i, (pattern, block_type) in enumerate(self.BLOCK_PATTERNS):
            matches = list(re.finditer(pattern, result, re.DOTALL))
            for j, match in enumerate(reversed(matches)):
                placeholder = f"\x00BLOCK_{block_type}_{i}_{j}\x00"
                protected.append((placeholder, match.group()))
                result = result[: match.start()] + placeholder + result[match.end():]
        return result, protected

    def _restore_blocks(self, text: str, protected: list[tuple[str, str]]) -> str:
        for placeholder, original in protected:
            text = text.replace(placeholder, original)
        return text

    # ------------------------------------------------------------------
    # Core splitting
    # ------------------------------------------------------------------

    @staticmethod
    def _split_on(text: str, sep: str, sep_on_left: bool) -> list[str]:
        parts = text.split(sep)
        result: list[str] = []
        for k, part in enumerate(parts):
            if k < len(parts) - 1:
                result.append(part + sep if sep_on_left else part + " ")
            else:
                result.append(part)
        return result

    def _split_segments(self, text: str, seps: list[tuple[str, bool]]) -> list[str]:
        if len(text) <= self.max_chunk_size:
            return [text] if text else []

        if not seps:
            return [
                text[i: i + self.max_chunk_size]
                for i in range(0, len(text), self.max_chunk_size)
            ]

        sep, sep_on_left = seps[0]
        remaining_seps = seps[1:]
        raw_pieces = self._split_on(text, sep, sep_on_left)
        pieces = [p for p in raw_pieces if p]

        if len(pieces) <= 1:
            return self._split_segments(text, remaining_seps)

        result: list[str] = []
        for piece in pieces:
            if len(piece) <= self.max_chunk_size:
                result.append(piece)
            else:
                result.extend(self._split_segments(piece, remaining_seps))
        return result

    def _merge_segments(self, segments: list[str]) -> list[str]:
        if not segments:
            return []
        chunks: list[str] = []
        current = segments[0]
        for seg in segments[1:]:
            candidate = current + seg
            if len(candidate) <= self.max_chunk_size:
                current = candidate
            else:
                chunks.append(current)
                current = seg
        chunks.append(current)
        return chunks

    # ------------------------------------------------------------------
    # Overlap
    # ------------------------------------------------------------------

    def _overlap_tail(self, text: str) -> str:
        if self.overlap_size <= 0 or len(text) <= self.overlap_size:
            return text
        raw_start = len(text) - self.overlap_size
        space_before = text.rfind(" ", 0, raw_start + 1)
        if space_before != -1:
            return text[space_before + 1:]
        return text[raw_start:]

    def _apply_overlap(self, chunks: list[str]) -> list[str]:
        if self.overlap_size <= 0 or len(chunks) <= 1:
            return chunks
        result = [chunks[0]]
        for i in range(1, len(chunks)):
            overlap = self._overlap_tail(chunks[i - 1])
            chunk = chunks[i]
            if len(overlap) + len(chunk) > self.max_chunk_size:
                allowed = self.max_chunk_size - len(overlap)
                if allowed > 0:
                    trimmed = chunk[:allowed]
                    last_space = trimmed.rfind(" ")
                    if last_space > 0:
                        trimmed = trimmed[:last_space]
                    chunk = trimmed
                else:
                    overlap = ""
            result.append(overlap + chunk)
        return result

    # ------------------------------------------------------------------
    # Small-chunk merging
    # ------------------------------------------------------------------

    def _merge_small_chunks(self, chunks: list[str]) -> list[str]:
        if not chunks:
            return chunks
        merged: list[str] = []
        buffer = ""
        for chunk in chunks:
            if buffer:
                candidate = buffer + chunk
                if len(candidate) <= self.max_chunk_size:
                    buffer = candidate
                    continue
                else:
                    merged.append(buffer)
                    buffer = ""
            if len(chunk) < self.min_chunk_size:
                buffer = chunk
            else:
                merged.append(chunk)
        if buffer:
            if merged and len(merged[-1]) + len(buffer) <= self.max_chunk_size:
                merged[-1] = merged[-1] + buffer
            else:
                merged.append(buffer)
        return merged

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk(self, text: str) -> list[str]:
        """Split *text* into a list of clean, ordered, meaningful chunks."""
        if not text or not text.strip():
            return []

        protected_text, blocks = self._protect_blocks(text)
        segments = self._split_segments(protected_text, _SEPARATORS)
        chunks = self._merge_segments(segments)
        chunks = [self._restore_blocks(c, blocks) for c in chunks]
        chunks = self._apply_overlap(chunks)
        chunks = self._merge_small_chunks(chunks)
        return [c.strip() for c in chunks if c and c.strip()]
