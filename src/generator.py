"""Grounded text generation through Gemini or Vercel AI Gateway."""

import logging
import re
import time
from typing import Optional

from src.config import config
from src.gemini_errors import GeminiResponseError, classify_gemini_error, is_retryable_gemini_error

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_RETRY_BACKOFF = [2, 5, 10]


def _is_rate_limited(error_text: str) -> bool:
    """Recognise throttling without matching unrelated words such as 'generate'."""
    normalized = error_text.lower()
    return bool(
        re.search(r"\b429\b|resource_exhausted|rate[ _-]?limit|quota", normalized)
    )


def _is_retryable(error: BaseException) -> bool:
    """Retry Gemini throttling, transient service failures, and timeouts only."""
    return is_retryable_gemini_error(error)


class GeminiGenerator:
    """Lazy generation facade shared by the local and hosted runtimes."""

    def __init__(self) -> None:
        self._client = None
        self._initialised = False

    def _initialise(self) -> None:
        """Initialise only the configured provider on first use."""
        if self._initialised:
            return

        try:
            if config.generation_provider == "vercel_gateway":
                gateway_token = config.ai_gateway_api_key or config.vercel_oidc_token
                if not gateway_token:
                    raise RuntimeError(
                        "Vercel AI Gateway requires AI_GATEWAY_API_KEY or VERCEL_OIDC_TOKEN"
                    )
                from openai import OpenAI

                self._client = OpenAI(
                    api_key=gateway_token,
                    base_url="https://ai-gateway.vercel.sh/v1",
                )
                model = config.ai_gateway_model
            else:
                if not config.gemini_api_key:
                    raise RuntimeError(
                        "GEMINI_API_KEY is not set. Configure it in the active environment."
                    )
                from google import genai

                self._client = genai.Client(api_key=config.gemini_api_key)
                model = config.gemini_model

            self._initialised = True
            logger.info(
                "Generation client initialised: provider=%s model=%s",
                config.generation_provider,
                model,
            )
        except Exception as exc:
            logger.error("Failed to initialise generation client: %s", exc)
            raise

    def _generate_once(self, user_prompt: str) -> str:
        """Issue one provider request and return its text payload."""
        from src.prompts import SYSTEM_PROMPT

        if config.generation_provider == "vercel_gateway":
            response = self._client.chat.completions.create(
                model=config.ai_gateway_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=2048,
            )
            return response.choices[0].message.content or ""

        from google.genai import types

        response = self._client.models.generate_content(
            model=config.gemini_model,
            contents=user_prompt,
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
        text = response.text or ""
        if text:
            return text
        candidates = getattr(response, "candidates", []) or []
        reason = str(getattr(candidates[0], "finish_reason", "")) if candidates else ""
        if "safety" in reason.lower():
            raise GeminiResponseError("safety_blocked")
        raise GeminiResponseError("empty_response")

    def generate(
        self,
        user_prompt: str,
        history: Optional[list[dict]] = None,
    ) -> str:
        """Generate one grounded answer from a pre-built evidence prompt."""
        del history  # History is already incorporated by build_user_prompt().
        self._initialise()
        last_error: Optional[Exception] = None

        for attempt in range(_MAX_RETRIES):
            try:
                text = self._generate_once(user_prompt)
                if text:
                    return text
                logger.warning("Generation returned an empty response (attempt %d)", attempt + 1)
                return ""
            except Exception as exc:
                last_error = exc
                error_info = classify_gemini_error(exc)
                if _is_retryable(exc):
                    wait = _RETRY_BACKOFF[min(attempt, len(_RETRY_BACKOFF) - 1)]
                    logger.warning(
                        "Generation rate limit hit; waiting %ds before retry %d/%d",
                        wait,
                        attempt + 1,
                        _MAX_RETRIES,
                    )
                    time.sleep(wait)
                    continue
                logger.error(
                    "Gemini generation failed: code=%s status=%s type=%s",
                    error_info.code,
                    error_info.http_status,
                    type(exc).__name__,
                )
                raise

        error_info = classify_gemini_error(last_error or RuntimeError("unknown Gemini error"))
        logger.error(
            "Gemini generation retries exhausted: code=%s status=%s",
            error_info.code,
            error_info.http_status,
        )
        raise GeminiResponseError(error_info.code) from last_error


generator = GeminiGenerator()
