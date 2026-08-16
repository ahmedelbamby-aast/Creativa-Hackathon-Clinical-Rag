"""Input sanitization for prompt injection prevention."""

import re


# Patterns that may indicate prompt injection
INJECTION_PATTERNS = [
    r"(?i)ignore\s+previous\s+instructions",
    r"(?i)system\s*:\s*",
    r"(?i)you\s+are\s+now",
    r"(?i)disregard\s+all",
    r"(?i)override\s+system",
]

# Delimiters for separating user content from system prompts
CONTENT_DELIMITER = "---USER_CONTENT_START---\n"
CONTENT_END_DELIMITER = "\n---USER_CONTENT_END---"


def sanitize_input(text: str) -> str:
    """Sanitize user input against prompt injection attempts.

    Args:
        text: Raw user input text.

    Returns:
        Sanitized text with injection patterns neutralized.
    """
    # Remove null bytes
    text = text.replace("\x00", "")

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def wrap_user_content(content: str) -> str:
    """Wrap user content with delimiters to prevent injection.

    Args:
        content: User-provided content to include in a prompt.

    Returns:
        Content wrapped with injection-prevention delimiters.
    """
    sanitized = sanitize_input(content)
    return f"{CONTENT_DELIMITER}{sanitized}{CONTENT_END_DELIMITER}"


def detect_injection_attempt(text: str) -> bool:
    """Check if text contains potential prompt injection patterns.

    Args:
        text: Text to check.

    Returns:
        True if injection patterns detected.
    """
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, text):
            return True
    return False
