"""Lightweight conversation memory manager.

Stores the last N turns of conversation (user + assistant messages).
Memory is used by the query rewriter for follow-up question contextualisation
and is injected into the LLM prompt for conversational continuity.

IMPORTANT: Memory is never used as a substitute for retrieval.
Every medical answer requires fresh retrieval from the vector store.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from src.config import config

logger = logging.getLogger(__name__)


@dataclass
class Turn:
    """One conversation turn (user or assistant)."""
    role: str           # "user" or "assistant"
    content: str
    category: str = ""  # Category used for this turn's retrieval


class ConversationMemory:
    """Rolling window conversation history.

    Keeps the last `max_turns` complete exchanges (1 exchange = 1 user + 1 assistant turn).
    Older turns are automatically dropped when the window is exceeded.
    """

    def __init__(self, max_turns: Optional[int] = None) -> None:
        self._max_turns = max_turns or config.max_memory_turns
        self._history: list[Turn] = []

    def add_user(self, content: str, category: str = "") -> None:
        """Record a user message."""
        self._history.append(Turn(role="user", content=content.strip(), category=category))
        self._trim()

    def add_assistant(self, content: str) -> None:
        """Record an assistant message."""
        self._history.append(Turn(role="assistant", content=content.strip()))
        self._trim()

    def _trim(self) -> None:
        """Keep only the last max_turns messages."""
        max_messages = self._max_turns * 2  # Each turn = user + assistant
        if len(self._history) > max_messages:
            self._history = self._history[-max_messages:]

    def get_history(self) -> list[dict]:
        """Return history as a list of dicts (for prompt injection and rewriter)."""
        return [{"role": t.role, "content": t.content} for t in self._history]

    def get_gemini_history(self) -> list[dict]:
        """Return history in Gemini chat format.

        Gemini chat history format:
            [{"role": "user", "parts": [...]}, {"role": "model", "parts": [...]}]
        """
        gemini_turns: list[dict] = []
        for turn in self._history[:-1]:  # Exclude the most recent user message
            role = "model" if turn.role == "assistant" else "user"
            gemini_turns.append({"role": role, "parts": [turn.content]})
        return gemini_turns

    def last_user_message(self) -> Optional[str]:
        """Return the content of the last user message, or None."""
        for turn in reversed(self._history):
            if turn.role == "user":
                return turn.content
        return None

    def clear(self) -> None:
        """Clear all conversation history."""
        self._history.clear()
        logger.debug("Conversation memory cleared")

    def __len__(self) -> int:
        return len(self._history)

    def __bool__(self) -> bool:
        return bool(self._history)
