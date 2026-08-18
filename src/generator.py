"""Grounded text generation through Gemini or Vercel AI Gateway."""

import logging
import time
from typing import Optional

from src.config import config

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_RETRY_BACKOFF = [2, 5, 10]


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
        return response.text or ""

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
                error_text = str(exc).lower()
                if any(marker in error_text for marker in ("rate", "quota", "429", "resource_exhausted")):
                    wait = _RETRY_BACKOFF[min(attempt, len(_RETRY_BACKOFF) - 1)]
                    logger.warning(
                        "Generation rate limit hit; waiting %ds before retry %d/%d",
                        wait,
                        attempt + 1,
                        _MAX_RETRIES,
                    )
                    time.sleep(wait)
                    continue
                logger.error("Generation API error: %s", exc)
                raise

        logger.error("All generation retries exhausted. Last error: %s", last_error)
        raise RuntimeError(f"Generation failed after {_MAX_RETRIES} retries: {last_error}")


generator = GeminiGenerator()
