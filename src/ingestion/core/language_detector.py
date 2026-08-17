"""Language detection utility.

Thin wrapper around langdetect. Returns ISO 639-1 language codes.
Falls back to "en" when detection is ambiguous or fails.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Minimum text length for reliable detection
_MIN_DETECTION_LENGTH = 20


def detect_language(text: str) -> str:
    """Detect the dominant language of *text*.

    Returns:
        ISO 639-1 language code, e.g. "en", "ar". Falls back to "en".
    """
    if not text or len(text.strip()) < _MIN_DETECTION_LENGTH:
        return "en"
    try:
        from langdetect import detect, LangDetectException
        return detect(text.strip())
    except Exception:
        return "en"


def detect_document_language(text: str, sample_chars: int = 3000) -> str:
    """Detect the dominant language of a full document using a text sample.

    Uses the first *sample_chars* characters of cleaned text for speed.
    """
    sample = text[:sample_chars].strip() if text else ""
    return detect_language(sample)
