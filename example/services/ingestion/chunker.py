"""Smart chunking service with semantic block protection."""

import re
from typing import Optional


class SmartChunker:
    """Split text into chunks with semantic block protection."""

    # Block patterns that should not be split
    BLOCK_PATTERNS = [
        (r'```[\s\S]*?```', 'code'),
        (r'\$\$[\s\S]*?\$\$', 'math'),
        (r'<table[\s\S]*?</table>', 'table'),
    ]

    def __init__(
        self,
        max_chunk_size: int = 500,
        overlap_size: int = 50,
    ):
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size

    def _protect_blocks(self, text: str) -> tuple[str, list[tuple[str, str]]]:
        """Replace semantic blocks with placeholders to protect them from splitting.

        Returns:
            Tuple of (text with placeholders, list of (placeholder, original) pairs).
        """
        protected = []
        result = text

        for i, (pattern, block_type) in enumerate(self.BLOCK_PATTERNS):
            matches = list(re.finditer(pattern, result, re.DOTALL))
            for j, match in enumerate(reversed(matches)):
                placeholder = f"__BLOCK_{block_type}_{i}_{j}__"
                original = match.group()
                protected.append((placeholder, original))
                result = result[:match.start()] + placeholder + result[match.end():]

        return result, protected

    def _restore_blocks(self, text: str, protected: list[tuple[str, str]]) -> str:
        """Restore protected blocks from placeholders."""
        result = text
        for placeholder, original in protected:
            result = result.replace(placeholder, original)
        return result

    def chunk(self, text: str) -> list[str]:
        """Split text into chunks with semantic block protection.

        Args:
            text: Full document text.

        Returns:
            List of text chunks.
        """
        if not text:
            return []

        # Protect semantic blocks
        protected_text, protected = self._protect_blocks(text)

        # Split by paragraphs first
        paragraphs = re.split(r'\n\n+', protected_text)

        chunks = []
        current_chunk = ""

        for para in paragraphs:
            if len(current_chunk) + len(para) <= self.max_chunk_size:
                current_chunk += ("\n\n" if current_chunk else "") + para
            else:
                if current_chunk:
                    chunks.append(current_chunk)

                # If single paragraph is too long, split by sentences
                if len(para) > self.max_chunk_size:
                    sentences = re.split(r'(?<=[.!?؟])\s+', para)
                    current_chunk = ""
                    for sentence in sentences:
                        if len(current_chunk) + len(sentence) <= self.max_chunk_size:
                            current_chunk += (" " if current_chunk else "") + sentence
                        else:
                            if current_chunk:
                                chunks.append(current_chunk)
                            current_chunk = sentence
                else:
                    current_chunk = para

        if current_chunk:
            chunks.append(current_chunk)

        # Restore protected blocks
        chunks = [self._restore_blocks(c, protected) for c in chunks]

        # Add overlap between chunks
        if self.overlap_size > 0 and len(chunks) > 1:
            overlapped = [chunks[0]]
            for i in range(1, len(chunks)):
                # Take last N chars of previous chunk as overlap
                overlap = chunks[i-1][-self.overlap_size:]
                overlapped.append(overlap + chunks[i])
            chunks = overlapped

        return [c.strip() for c in chunks if c.strip()]


# Global instance with default settings
smart_chunker = SmartChunker()
