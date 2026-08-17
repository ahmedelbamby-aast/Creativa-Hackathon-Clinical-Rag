"""Gemini LLM generation wrapper — uses the new google-genai SDK.

Wraps the google.genai client with:
- API key validation at startup
- Configurable model (from .env)
- Retry logic for rate limits / transient errors
- Graceful error handling

The generator takes a pre-built prompt (from prompts.py) and returns
the model's text response. It does NOT retrieve chunks — that is the
retriever's responsibility.
"""

import logging
import time
from typing import Optional

from src.config import config

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_RETRY_BACKOFF = [2, 5, 10]  # seconds between retries


class GeminiGenerator:
    """Lazy-initialising Gemini API wrapper using the google.genai SDK."""

    def __init__(self) -> None:
        self._client = None
        self._initialised = False

    def _initialise(self) -> None:
        """Initialise the Gemini client on first use."""
        if self._initialised:
            return

        if not config.gemini_api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. "
                "Copy .env.example to .env and add your Gemini API key."
            )

        try:
            from google import genai
            self._client = genai.Client(api_key=config.gemini_api_key)
            self._initialised = True
            logger.info("Gemini client initialised for model: %s", config.gemini_model)
        except Exception as e:
            logger.error("Failed to initialise Gemini client: %s", e)
            raise

    def generate(
        self,
        user_prompt: str,
        history: Optional[list[dict]] = None,
    ) -> str:
        """Generate a grounded answer from the Gemini model.

        Args:
            user_prompt: The full formatted prompt (with context injected).
            history: Ignored (history is baked into the prompt via build_user_prompt).

        Returns:
            Generated text response.
        """
        self._initialise()

        from src.prompts import SYSTEM_PROMPT
        from google.genai import types

        # Build the full content including system instruction as a prepended message
        contents = user_prompt

        last_error: Optional[Exception] = None

        for attempt in range(_MAX_RETRIES):
            try:
                response = self._client.models.generate_content(
                    model=config.gemini_model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        temperature=0.1,
                        top_p=0.95,
                        max_output_tokens=2048,
                        safety_settings=[
                            types.SafetySetting(
                                category="HARM_CATEGORY_DANGEROUS_CONTENT",
                                threshold="BLOCK_NONE",
                            ),
                        ],
                    ),
                )

                # Extract text
                if response.text:
                    return response.text

                logger.warning("Gemini returned empty response (attempt %d)", attempt + 1)
                return ""

            except Exception as e:
                last_error = e
                error_str = str(e).lower()

                # Rate limit → wait and retry
                if any(k in error_str for k in ["rate", "quota", "429", "resource_exhausted"]):
                    wait = _RETRY_BACKOFF[min(attempt, len(_RETRY_BACKOFF) - 1)]
                    logger.warning(
                        "Rate limit hit. Waiting %ds before retry %d/%d...",
                        wait, attempt + 1, _MAX_RETRIES,
                    )
                    time.sleep(wait)
                    continue

                # Non-retryable errors
                logger.error("Gemini API error: %s", e)
                raise

        logger.error("All %d retries exhausted. Last error: %s", _MAX_RETRIES, last_error)
        raise RuntimeError(f"Gemini generation failed after {_MAX_RETRIES} retries: {last_error}")


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
generator = GeminiGenerator()
