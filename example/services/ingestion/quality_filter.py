"""Quality filtering service for text chunks."""

import re


class QualityFilter:
    """Filter low-quality chunks based on multiple signals."""

    def __init__(
        self,
        min_length: int = 20,
        min_text_ratio: float = 0.3,
        min_domain_signals: int = 1,
    ):
        self.min_length = min_length
        self.min_text_ratio = min_text_ratio
        self.min_domain_signals = min_domain_signals

    def calculate_quality_score(self, text: str) -> float:
        """Calculate quality score for a chunk.

        Args:
            text: Chunk text.

        Returns:
            Quality score between 0.0 and 1.0.
        """
        if not text:
            return 0.0

        # Length score (word count)
        word_count = len(text.split())
        length_score = min(word_count / 100, 1.0)  # Normalize to 100 words

        # Text-to-symbol ratio
        text_chars = sum(1 for c in text if c.isalpha() or c.isspace())
        total_chars = len(text)
        text_ratio = text_chars / total_chars if total_chars > 0 else 0

        # Domain signals
        domain_signals = self._count_domain_signals(text)
        signal_score = min(domain_signals / 5, 1.0)  # Normalize to 5 signals

        # Combined score
        score = (length_score * 0.4) + (text_ratio * 0.4) + (signal_score * 0.2)
        return round(min(score, 1.0), 3)

    def _count_domain_signals(self, text: str) -> int:
        """Count domain-specific signals in text."""
        signals = 0
        # Technical terms
        signals += len(re.findall(r'\b(theorem|equation|process|analysis)\b', text, re.IGNORECASE))
        # Numbers and formulas
        signals += len(re.findall(r'\d+[.,]\d+|\d+%', text))
        # Special characters
        signals += len(re.findall(r'[=<>+\-*/]', text))
        return signals

    def filter(self, chunks: list[str], min_score: float = 0.1) -> list[tuple[str, float]]:
        """Filter chunks by quality score.

        Args:
            chunks: List of chunk texts.
            min_score: Minimum quality score to keep.

        Returns:
            List of (chunk, score) tuples for chunks above threshold.
        """
        scored = [(chunk, self.calculate_quality_score(chunk)) for chunk in chunks]
        return [(chunk, score) for chunk, score in scored if score >= min_score]


# Global instance
quality_filter = QualityFilter()
