"""Grounded text generation through Gemini, Groq, or Vercel AI Gateway."""

import logging
import re
import time
from typing import Optional

from src.config import config
from src.gemini_errors import GeminiResponseError, classify_gemini_error, is_retryable_gemini_error

logger = logging.getLogger(__name__)

_MAX_RETRIES = 1
_RETRY_BACKOFF = [0]


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
    """Provider-aware generation facade with safe automatic failover."""

    def __init__(self) -> None:
        self._client = None
        self._initialised = False
        self._clients: dict[str, object] = {}
        self._active_provider = ""

    @property
    def active_provider(self) -> str:
        """Provider used by the latest successful call, or its configured first choice."""
        return self._active_provider or self._provider_order()[0]

    @property
    def active_model(self) -> str:
        """Model used by the latest successful call, or the configured first choice."""
        return self._provider_model(self.active_provider)

    def mark_extractive_fallback(self) -> None:
        """Expose deterministic evidence mode after every LLM route fails."""
        self._active_provider = "extractive"

    def _provider_order(self) -> list[str]:
        if config.generation_provider != "auto":
            providers = [config.generation_provider, config.generation_fallback_provider]
        else:
            providers = [config.generation_primary_provider, config.generation_fallback_provider]
        return list(dict.fromkeys(provider for provider in providers if provider))

    @staticmethod
    def _provider_model(provider: str) -> str:
        if provider == "gemini":
            return config.gemini_model
        if provider == "groq":
            return config.groq_model
        if provider == "vercel_gateway":
            return config.ai_gateway_model
        return "extractive"

    def _initialise(self, provider: str | None = None) -> None:
        """Initialise one selected provider lazily and cache its client."""
        provider = provider or config.generation_provider
        if provider in self._clients:
            self._client = self._clients[provider]
            self._initialised = True
            return

        # Preserves direct unit-test injection of self._client.
        if self._initialised and self._client is not None and not self._clients:
            return

        try:
            if provider == "vercel_gateway":
                gateway_token = config.ai_gateway_api_key or config.vercel_oidc_token
                if not gateway_token:
                    raise RuntimeError(
                        "Vercel AI Gateway requires AI_GATEWAY_API_KEY or VERCEL_OIDC_TOKEN"
                    )
                from openai import OpenAI

                self._client = OpenAI(
                    api_key=gateway_token,
                    base_url="https://ai-gateway.vercel.sh/v1",
                    max_retries=0,
                    timeout=20.0,
                )
                model = config.ai_gateway_model
            elif provider == "groq":
                if not config.groq_api_key:
                    raise RuntimeError("GROQ_API_KEY is not set. Configure it in the active environment.")
                from openai import OpenAI

                self._client = OpenAI(
                    api_key=config.groq_api_key,
                    base_url="https://api.groq.com/openai/v1",
                    max_retries=0,
                    timeout=20.0,
                )
                model = config.groq_model
            elif provider == "gemini":
                if not config.gemini_api_key:
                    raise RuntimeError(
                        "GEMINI_API_KEY is not set. Configure it in the active environment."
                    )
                from google import genai

                self._client = genai.Client(api_key=config.gemini_api_key)
                model = config.gemini_model
            else:
                raise RuntimeError(f"Unsupported generation provider: {provider}")

            self._initialised = True
            self._clients[provider] = self._client
            logger.info(
                "Generation client initialised: provider=%s model=%s",
                provider,
                model,
            )
        except Exception as exc:
            logger.error("Failed to initialise generation client: %s", exc)
            raise

    def _generate_once(self, user_prompt: str, provider: str | None = None) -> str:
        """Issue one provider request and return its text payload."""
        from src.prompts import SYSTEM_PROMPT

        provider = provider or config.generation_provider
        if provider in {"vercel_gateway", "groq"}:
            response = self._client.chat.completions.create(
                model=self._provider_model(provider),
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=2048,
            )
            text = response.choices[0].message.content or ""
            if text.strip():
                return text
            raise GeminiResponseError("empty_response")

        from google.genai import types

        response = self._client.models.generate_content(
            model=config.gemini_model,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.0,
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

    def _generate_with_provider(self, user_prompt: str, provider: str) -> str:
        self._initialise(provider)
        last_error: Optional[Exception] = None
        for attempt in range(_MAX_RETRIES):
            try:
                return self._generate_once(user_prompt, provider)
            except Exception as exc:
                last_error = exc
                error_info = classify_gemini_error(exc)
                if _is_retryable(exc):
                    if attempt < _MAX_RETRIES - 1:
                        wait = _RETRY_BACKOFF[min(attempt, len(_RETRY_BACKOFF) - 1)]
                        logger.warning(
                            "%s generation is temporarily unavailable; waiting %ds before retry %d/%d",
                            provider.title(), wait, attempt + 1, _MAX_RETRIES,
                        )
                        time.sleep(wait)
                        continue
                    break
                logger.error(
                    "%s generation failed: code=%s status=%s type=%s",
                    provider.title(), error_info.code, error_info.http_status, type(exc).__name__,
                )
                raise
        error_info = classify_gemini_error(last_error or RuntimeError("unknown provider error"))
        log = logger.warning if _is_retryable(last_error or RuntimeError()) else logger.error
        log(
            "%s generation attempt unavailable: code=%s status=%s",
            provider.title(), error_info.code, error_info.http_status,
        )
        raise GeminiResponseError(error_info.code) from last_error

    def generate(
        self,
        user_prompt: str,
        history: Optional[list[dict]] = None,
    ) -> str:
        """Generate one grounded answer from a pre-built evidence prompt."""
        del history  # History is already incorporated by build_user_prompt().
        providers = self._provider_order()
        if not providers:
            raise RuntimeError("No generation provider is configured")
        last_error: Exception | None = None
        for index, provider in enumerate(providers):
            try:
                answer = self._generate_with_provider(user_prompt, provider)
                self._active_provider = provider
                return answer
            except Exception as exc:
                last_error = exc
                error_info = classify_gemini_error(exc)
                can_failover = (
                    index < len(providers) - 1
                    and error_info.code not in {"safety_blocked", "invalid_request"}
                )
                if can_failover:
                    logger.warning(
                        "Switching generation provider from %s to %s after code=%s",
                        provider,
                        providers[index + 1],
                        error_info.code,
                    )
                    continue
                raise

        raise RuntimeError("Generation provider routing finished without an answer") from last_error


generator = GeminiGenerator()
