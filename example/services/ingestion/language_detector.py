"""Language detection service."""

from typing import Optional


class LanguageDetector:
    """Detect document language using Unicode character distribution analysis."""

    def detect(self, text: str) -> str:
        """Detect the language of the given text.

        Args:
            text: Text to analyze.

        Returns:
            Language code: 'ar', 'en', or 'mixed'.
        """
        if not text:
            return "en"

        # Count Arabic vs Latin characters
        arabic_count = sum(1 for c in text if '\u0600' <= c <= '\u06FF' or '\u0750' <= c <= '\u077F')
        latin_count = sum(1 for c in text if 'a' <= c.lower() <= 'z')
        total = arabic_count + latin_count

        if total == 0:
            return "en"

        arabic_ratio = arabic_count / total

        # Thresholds for classification
        if arabic_ratio > 0.7:
            return "ar"
        elif arabic_ratio < 0.3:
            return "en"
        else:
            return "mixed"

    def detect_document(self, text: str, sample_size: int = 10000) -> str:
        """Detect language using a sample of the text for efficiency.

        Args:
            text: Full document text.
            sample_size: Number of characters to sample.

        Returns:
            Language code: 'ar', 'en', or 'mixed'.
        """
        # Sample from beginning, middle, and end
        samples = []
        if len(text) > sample_size * 3:
            samples.append(text[:sample_size])
            samples.append(text[len(text)//2:len(text)//2 + sample_size])
            samples.append(text[-sample_size:])
        else:
            samples.append(text)

        # Detect language for each sample and take majority
        results = [self.detect(s) for s in samples]
        return max(set(results), key=results.count)


# Global instance
language_detector = LanguageDetector()
