"""Serverless-safe FastAPI entrypoint used by Vercel's Python runtime."""

import logging
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from app import rag_pipeline
from src.config import (
    CATEGORY_ALL,
    CATEGORY_NUTRITION,
    CATEGORY_PREVENTION,
    CATEGORY_TREATMENT,
    config,
)
from src.memory import ConversationMemory
from src.vector_store import vector_store


logger = logging.getLogger(__name__)

api = FastAPI(
    title="Creativa Diabetes RAG",
    description="Serverless API and bilingual web client for the diabetes RAG application.",
    version="0.1.0",
)

STATIC_INDEX = Path(__file__).with_name("static") / "index.html"
VALID_CATEGORIES = {
    CATEGORY_ALL,
    CATEGORY_TREATMENT,
    CATEGORY_PREVENTION,
    CATEGORY_NUTRITION,
}


class ChatMessage(BaseModel):
    """A prior browser-owned conversation message."""

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=12_000)


class ChatRequest(BaseModel):
    """Validated request body for a stateless RAG turn."""

    message: str = Field(min_length=1, max_length=2_000)
    category: str = CATEGORY_ALL
    history: list[ChatMessage] = Field(default_factory=list, max_length=12)


class ChatResponse(BaseModel):
    """A grounded answer and its display metadata."""

    answer: str
    citations: str
    debug: str = ""


@api.get("/api/health", tags=["operations"])
def health() -> dict[str, object]:
    """Return process-level health without contacting external services."""
    return {
        "status": "ok",
        "environment": config.app_env,
        "embedding_provider": config.embedding_provider,
        "embedding_namespace": config.resolved_embedding_namespace,
        "generation_provider": config.generation_provider,
        "database_configured": bool(config.database_url),
        "gemini_configured": bool(config.gemini_api_key),
        "generation_configured": config.generation_configured,
    }


@api.get("/api/ready", tags=["operations"])
def ready() -> dict[str, object]:
    """Verify pgvector and the active knowledge-base namespace."""
    try:
        versions = vector_store.healthcheck()
        category_counts = vector_store.collection_stats()
    except Exception as exc:
        logger.exception("Deployment readiness check failed")
        raise HTTPException(status_code=503, detail="Knowledge base is unavailable") from exc

    indexed_chunks = sum(category_counts.values())
    if indexed_chunks == 0:
        raise HTTPException(status_code=503, detail="Knowledge base is empty")

    return {
        "status": "ready",
        "postgres": versions["postgres"],
        "pgvector": versions["pgvector"],
        "namespace": vector_store.namespace,
        "indexed_chunks": indexed_chunks,
        "categories": category_counts,
    }


@api.post("/api/chat", response_model=ChatResponse, tags=["rag"])
def chat_endpoint(request: ChatRequest) -> ChatResponse:
    """Run one stateless RAG turn using browser-supplied bounded history."""
    category = request.category.strip().lower()
    if category not in VALID_CATEGORIES:
        raise HTTPException(status_code=422, detail="Unsupported knowledge category")
    if not config.generation_configured:
        raise HTTPException(
            status_code=503,
            detail="Answer generation is not configured for this deployment",
        )

    memory = ConversationMemory()
    for message in request.history[-(config.max_memory_turns * 2) :]:
        if message.role == "user":
            memory.add_user(message.content, category=category)
        else:
            memory.add_assistant(message.content)

    try:
        answer, citations, debug = rag_pipeline(request.message, category, memory)
    except Exception as exc:
        logger.exception("RAG request failed")
        raise HTTPException(
            status_code=503,
            detail="The assistant is temporarily unavailable. Please try again.",
        ) from exc

    return ChatResponse(answer=answer, citations=citations, debug=debug)


@api.get("/", response_class=HTMLResponse, include_in_schema=False)
def index() -> HTMLResponse:
    """Serve the dependency-free client that works on stateless functions."""
    return HTMLResponse(STATIC_INDEX.read_text(encoding="utf-8"))


app = api
