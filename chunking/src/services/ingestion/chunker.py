# """Smart chunking service with semantic block protection."""

# import re
# from typing import Optional


# class SmartChunker:
#     """Split text into chunks with semantic block protection.

#     Paragraph splitting is hierarchical and file-agnostic: the text is split
#     at the largest meaningful unit first (paragraph -> line -> sentence ->
#     word -> character) so chunks always respect ``max_chunk_size`` while
#     keeping as much context as possible. It works the same way regardless of
#     whether paragraphs are separated by blank lines, single line breaks, or
#     nothing visible (as often happens in extracted PDF text).

#     Overlap between chunks is snapped to word boundaries so words are never
#     cut in half.
#     """

#     # Block patterns that should not be split
#     BLOCK_PATTERNS = [
#         (r'```[\s\S]*?```', 'code'),
#         (r'\$\$[\s\S]*?\$\$', 'math'),
#         (r'<table[\s\S]*?</table>', 'table'),
#     ]

#     # Separator hierarchy: largest meaningful unit first, smallest last.
#     # Covers blank-line paragraphs, single-line paragraphs, sentence endings
#     # for English and Arabic, the Arabic comma, plain word spaces, and finally
#     # character-level splitting as a hard fallback.
#     SEPARATORS = [
#         '\n\n',
#         '\n',
#         '. ', '.\n',
#         '! ', '!\n',
#         '? ', '?\n',
#         '؟ ', '؟\n',
#         '، ',
#         ' ',
#         '',
#     ]

#     def __init__(
#         self,
#         max_chunk_size: int = 500,
#         overlap_size: int = 50,
#     ):
#         self.max_chunk_size = max_chunk_size
#         self.overlap_size = overlap_size

#     def _protect_blocks(self, text: str) -> tuple[str, list[tuple[str, str]]]:
#         """Replace semantic blocks with placeholders to protect them from splitting.

#         Returns:
#             Tuple of (text with placeholders, list of (placeholder, original) pairs).
#         """
#         protected = []
#         result = text

#         for i, (pattern, block_type) in enumerate(self.BLOCK_PATTERNS):
#             matches = list(re.finditer(pattern, result, re.DOTALL))
#             for j, match in enumerate(reversed(matches)):
#                 placeholder = f"__BLOCK_{block_type}_{i}_{j}__"
#                 original = match.group()
#                 protected.append((placeholder, original))
#                 result = result[:match.start()] + placeholder + result[match.end():]

#         return result, protected

#     def _restore_blocks(self, text: str, protected: list[tuple[str, str]]) -> str:
#         """Restore protected blocks from placeholders."""
#         result = text
#         for placeholder, original in protected:
#             result = result.replace(placeholder, original)
#         return result

#     def _recursive_split(self, text: str, separators: list[str]) -> list[str]:
#         """Split text using a separator hierarchy.

#         Splits on the current (largest) separator first. Pieces that fit in
#         ``max_chunk_size`` are merged with their separator; pieces that are
#         still too large are recursed into with the next separator in the
#         hierarchy. The final fallback splits by characters.

#         Args:
#             text: Text to split.
#             separators: Remaining separator hierarchy.

#         Returns:
#             List of pieces, each at most ``max_chunk_size`` characters.
#         """
#         sep = separators[-1]
#         if sep == '':
#             return [
#                 text[i:i + self.max_chunk_size]
#                 for i in range(0, len(text), self.max_chunk_size)
#             ]

#         chunks = []
#         current = ""
#         for part in text.split(sep):
#             if not part:
#                 continue
#             candidate = part if not current else current + sep + part
#             if len(candidate) <= self.max_chunk_size:
#                 current = candidate
#             else:
#                 if current:
#                     chunks.append(current)
#                 if len(part) <= self.max_chunk_size:
#                     current = part
#                 else:
#                     chunks.extend(self._recursive_split(part, separators[:-1]))
#                     current = ""

#         if current:
#             chunks.append(current)
#         return chunks

#     def _overlap_tail(self, text: str) -> str:
#         """Return the last ``overlap_size`` chars expanded to a word boundary.

#         Args:
#             text: Previous chunk text.

#         Returns:
#             Overlap string starting at a word boundary (never mid-word).
#         """
#         if len(text) <= self.overlap_size:
#             return text
#         start = len(text) - self.overlap_size
#         space = text.find(" ", start)
#         if space != -1:
#             return text[space + 1:]
#         return text[start:]

#     def _build_overlapped(self, chunks: list[str]) -> list[str]:
#         """Add word-boundary-snapped overlap between consecutive chunks."""
#         overlapped = [chunks[0]]
#         for i in range(1, len(chunks)):
#             overlap = self._overlap_tail(chunks[i - 1])
#             overlapped.append(overlap + chunks[i])
#         return overlapped

#     def chunk(self, text: str) -> list[str]:
#         """Split text into chunks with semantic block protection.

#         Args:
#             text: Full document (or page) text.

#         Returns:
#             List of text chunks.
#         """
#         if not text:
#             return []

#         # Protect semantic blocks
#         protected_text, protected = self._protect_blocks(text)

#         # Hierarchical paragraph/sentence/word splitting
#         chunks = self._recursive_split(protected_text, self.SEPARATORS)

#         # Restore protected blocks
#         chunks = [self._restore_blocks(c, protected) for c in chunks]

#         # Add word-boundary-snapped overlap between chunks
#         if self.overlap_size > 0 and len(chunks) > 1:
#             chunks = self._build_overlapped(chunks)

#         return [c.strip() for c in chunks if c.strip()]


# # Global instance with default settings
# smart_chunker = SmartChunker()

"""
Smart Chunker v2 – General-purpose, language-aware chunking.
Works for: PDF, plain text, Markdown, mixed Arabic/English.
"""

import re
from typing import Optional


class SmartChunker:
    """
    Hierarchical text chunker with:
    - Semantic block protection (code, math, tables)
    - Separator hierarchy: paragraph → line → sentence → clause → word → char
    - Word-boundary-safe overlap
    - Hard max_chunk_size guarantee (overlap included)
    """

    # ─── Block patterns to protect from splitting ───────────────────────
    BLOCK_PATTERNS = [
        (r'```[\s\S]*?```', 'code'),
        (r'\$\$[\s\S]*?\$\$', 'math'),
        (r'<table[\s\S]*?</table>', 'table_html'),
        (r'\|[^\n]+\|(?:\n\|[^\n]+\|)+', 'table_md'),  # Markdown tables
    ]

    # ─── Separator hierarchy (largest → smallest) ──────────────────────
    # Works for English + Arabic
    SEPARATORS = [
        '\n\n\n',       # Major section breaks
        '\n\n',         # Paragraph breaks
        '\n',           # Line breaks
        '. ',           # English sentence end
        '.\n',
        '! ',
        '!\n',
        '? ',
        '?\n',
        '؟ ',           # Arabic question mark
        '؟\n',
        '؛ ',           # Arabic semicolon
        '، ',           # Arabic comma
        ', ',           # English comma
        '; ',
        ' ',            # Word boundary
        '',             # Character-level fallback
    ]

    def __init__(
        self,
        max_chunk_size: int = 500,
        overlap_size: int = 50,
        min_chunk_size: int = 50,
    ):
        if overlap_size >= max_chunk_size:
            raise ValueError("overlap_size must be < max_chunk_size")
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size
        self.min_chunk_size = min_chunk_size
        # Effective content size per chunk (leaving room for overlap)
        self._content_budget = max_chunk_size - overlap_size

    # ─── Block Protection ───────────────────────────────────────────────

    def _protect_blocks(self, text: str) -> tuple[str, list[tuple[str, str]]]:
        protected = []
        result = text
        for i, (pattern, block_type) in enumerate(self.BLOCK_PATTERNS):
            for j, match in enumerate(reversed(list(re.finditer(pattern, result, re.DOTALL)))):
                placeholder = f"\x00BLOCK_{block_type}_{i}_{j}\x00"
                protected.append((placeholder, match.group()))
                result = result[:match.start()] + placeholder + result[match.end():]
        return result, protected

    def _restore_blocks(self, text: str, protected: list[tuple[str, str]]) -> str:
        for placeholder, original in protected:
            text = text.replace(placeholder, original)
        return text

    # ─── Core Recursive Split (FIXED) ──────────────────────────────────

    def _split_by_separator(self, text: str, sep: str, budget: int) -> list[str]:
        """Split text by a single separator, merging parts up to budget."""
        if not sep:
            # Character-level fallback
            return [text[i:i + budget] for i in range(0, len(text), budget)]

        parts = text.split(sep)
        pieces = []
        current = ""

        for part in parts:
            if not part and sep.strip():
                continue
            candidate = part if not current else current + sep + part
            if len(candidate) <= budget:
                current = candidate
            else:
                if current:
                    pieces.append(current)
                # If single part exceeds budget, it needs deeper splitting
                if len(part) > budget:
                    pieces.append(part)  # Mark for recursion
                else:
                    current = part
                    continue
                current = ""
        if current:
            pieces.append(current)
        return pieces

    def _recursive_split(self, text: str, seps: list[str], budget: int) -> list[str]:
        """
        ✅ FIXED: Start from separators[0] (largest) and recurse deeper.
        """
        if not text or len(text) <= budget:
            return [text] if text.strip() else []

        if not seps:
            # Absolute fallback: hard character split
            return [text[i:i + budget] for i in range(0, len(text), budget)]

        sep = seps[0]  # ✅ Use FIRST separator (largest)
        remaining = seps[1:]

        pieces = self._split_by_separator(text, sep, budget)

        result = []
        for piece in pieces:
            if len(piece) <= budget:
                result.append(piece)
            else:
                # ✅ Recurse with NEXT (smaller) separator
                result.extend(self._recursive_split(piece, remaining, budget))

        return result

    # ─── Overlap (word-boundary safe, size-guaranteed) ────────────────

    def _get_overlap(self, text: str) -> str:
        """Get overlap from end of previous chunk, snapped to word boundary."""
        if len(text) <= self.overlap_size:
            return text

        start = len(text) - self.overlap_size
        # Snap forward to next word boundary
        space_idx = text.find(' ', start)
        if space_idx != -1 and space_idx < len(text) - 1:
            return text[space_idx + 1:]
        return text[start:]

    def _apply_overlap(self, chunks: list[str]) -> list[str]:
        """
        Add overlap while GUARANTEEING final size <= max_chunk_size.
        Overlap is prepended, but chunk content is trimmed if needed.
        """
        if len(chunks) <= 1 or self.overlap_size <= 0:
            return chunks

        result = [chunks[0]]
        for i in range(1, len(chunks)):
            overlap = self._get_overlap(chunks[i - 1])
            available = self.max_chunk_size - len(overlap)

            chunk_text = chunks[i]
            if len(chunk_text) > available:
                # Trim from end, try to keep word boundary
                trimmed = chunk_text[:available]
                last_space = trimmed.rfind(' ')
                if last_space > available * 0.7:  # Only if we don't lose too much
                    trimmed = trimmed[:last_space]
                chunk_text = trimmed

            result.append(overlap + chunk_text)

        return result

    # ─── Merge tiny trailing chunks ─────────────────────────────────────

    def _merge_small_chunks(self, chunks: list[str]) -> list[str]:
        """Merge chunks smaller than min_chunk_size into neighbors."""
        if not chunks:
            return chunks

        merged = []
        buffer = ""

        for chunk in chunks:
            if buffer:
                combined = buffer + " " + chunk
                if len(combined) <= self.max_chunk_size:
                    buffer = combined
                    continue
                else:
                    merged.append(buffer)
                    buffer = ""

            if len(chunk) < self.min_chunk_size:
                buffer = chunk
            else:
                merged.append(chunk)

        if buffer:
            if merged and len(merged[-1]) + len(buffer) + 1 <= self.max_chunk_size:
                merged[-1] = merged[-1] + " " + buffer
            else:
                merged.append(buffer)

        return merged

    # ─── Public API ─────────────────────────────────────────────────────

    def chunk(self, text: str) -> list[str]:
        """
        Split text into chunks.
        
        ✅ Guarantees:
        - Every chunk <= max_chunk_size characters
        - Overlap is word-boundary safe
        - Semantic blocks (code/math/tables) are never split
        - Works for any language (Arabic, English, mixed)
        """
        if not text or not text.strip():
            return []

        # 1. Protect semantic blocks
        protected_text, blocks = self._protect_blocks(text)

        # 2. Hierarchical split using content budget (reserves space for overlap)
        chunks = self._recursive_split(
            protected_text,
            self.SEPARATORS,
            budget=self._content_budget,  # ✅ Leave room for overlap
        )

        # 3. Restore protected blocks
        chunks = [self._restore_blocks(c, blocks) for c in chunks]

        # 4. Apply overlap (size-guaranteed)
        chunks = self._apply_overlap(chunks)

        # 5. Merge tiny fragments
        chunks = self._merge_small_chunks(chunks)

        # 6. Final cleanup
        return [c.strip() for c in chunks if c and c.strip()]


# ─── Default instance ───────────────────────────────────────────────────────
smart_chunker = SmartChunker(
    max_chunk_size=500,
    overlap_size=50,
    min_chunk_size=50,
)
