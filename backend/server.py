"""Serverless-safe FastAPI entrypoint used by Vercel's Python runtime."""

import logging
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app import generate_from_evidence, rag_pipeline
from src.config import (
    CATEGORY_ALL,
    CATEGORY_NUTRITION,
    CATEGORY_PREVENTION,
    CATEGORY_TREATMENT,
    config,
)
from src.memory import ConversationMemory
from src.generator import generator
from src.evidence_service import rehydrate_evidence, stage_evidence
from src.evidence_service import envelope_chunks
from src.citations import build_citation_records
from src.sample_questions import load_sample_questions
from src.vector_store import vector_store


logger = logging.getLogger(__name__)

api = FastAPI(
    title="Creativa Diabetes RAG",
    description="Serverless API and bilingual web client for the diabetes RAG application.",
    version="0.1.0",
)

STATIC_DIR = Path(__file__).with_name("static")
STATIC_INDEX = STATIC_DIR / "index.html"
api.mount("/assets", StaticFiles(directory=STATIC_DIR), name="assets")
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


class CitationItem(BaseModel):
    evidence_id: str
    source_id: str
    document_name: str
    source_url: str
    publisher: str = ""
    publication_date: str = ""
    page_number: int
    section_title: str = ""


class ChatResponse(BaseModel):
    """A grounded answer and its display metadata."""

    answer: str
    citations: str
    debug: str = ""
    generation_provider: str = ""
    generation_model: str = ""
    sources: list[CitationItem] = Field(default_factory=list)


class EvidenceItem(BaseModel):
    chunk_id: str
    text: str
    score: float
    document_name: str
    page_number: int
    section_title: str
    subsection_title: str = ""
    category: str = ""
    source_id: str
    source_url: str
    publisher: str = ""
    publication_date: str = ""


class RetrieveResponse(BaseModel):
    status: str
    message: str = ""
    error_code: str = ""
    namespace: str
    index_manifest_hash: str = ""
    chunks: list[EvidenceItem] = Field(default_factory=list)


class GenerateRequest(ChatRequest):
    namespace: str
    index_manifest_hash: str
    chunk_ids: list[str] = Field(min_length=1, max_length=5)


class LocalizedText(BaseModel):
    en: str
    ar: str


class SampleQuestion(BaseModel):
    id: str
    language: Literal["en", "ar"]
    category: str
    text: str


class SampleScenario(BaseModel):
    id: str
    title: LocalizedText
    description: LocalizedText
    expected_status: str
    questions: list[SampleQuestion]


class SampleCatalog(BaseModel):
    version: int
    scenarios: list[SampleScenario]


def _build_memory(history: list[ChatMessage], category: str) -> ConversationMemory:
    memory = ConversationMemory()
    for message in history[-(config.max_memory_turns * 2) :]:
        if message.role == "user":
            memory.add_user(message.content, category=category)
        else:
            memory.add_assistant(message.content)
    return memory


@api.get("/api/health", tags=["operations"])
def health() -> dict[str, object]:
    """Return process-level health without contacting external services."""
    return {
        "status": "ok",
        "environment": config.app_env,
        "embedding_provider": config.embedding_provider,
        "embedding_namespace": config.resolved_embedding_namespace,
        "generation_provider": config.generation_provider,
        "configured_generation_provider": config.configured_generation_provider_label,
        "active_generation_provider": generator.active_provider,
        "active_generation_model": generator.active_model,
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


@api.get("/api/sample-questions", response_model=SampleCatalog, tags=["rag"])
def sample_questions() -> dict:
    """Expose the single validated bilingual sample matrix used by the UI and tests."""
    return load_sample_questions()


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

    memory = _build_memory(request.history, category)

    try:
        answer, citations, debug = rag_pipeline(request.message, category, memory)
    except Exception as exc:
        logger.exception("RAG request failed")
        raise HTTPException(
            status_code=503,
            detail="The assistant is temporarily unavailable. Please try again.",
        ) from exc

    return ChatResponse(
        answer=answer,
        citations=citations,
        debug=debug,
        generation_provider=generator.active_provider,
        generation_model=generator.active_model,
    )


@api.post("/api/retrieve", response_model=RetrieveResponse, tags=["rag"])
def retrieve_endpoint(request: ChatRequest) -> RetrieveResponse:
    """Stage and expose evidence before the browser requests answer generation."""
    category = request.category.strip().lower()
    if category not in VALID_CATEGORIES:
        raise HTTPException(status_code=422, detail="Unsupported knowledge category")
    envelope = stage_evidence(request.message, category, _build_memory(request.history, category).get_history())
    return RetrieveResponse(
        status=envelope.status,
        message=envelope.user_message,
        error_code=envelope.error_code,
        namespace=envelope.namespace,
        index_manifest_hash=envelope.index_manifest_hash,
        chunks=[
            EvidenceItem(
                chunk_id=item.chunk_id,
                text=item.text,
                score=item.score,
                document_name=item.document_name,
                page_number=item.page_number,
                section_title=item.section_title,
                subsection_title=item.subsection_title,
                category=item.category,
                source_id=item.source_id,
                source_url=item.source_url,
                publisher=item.publisher,
                publication_date=item.publication_date,
            )
            for item in envelope.chunks
        ],
    )


@api.post("/api/generate", response_model=ChatResponse, tags=["rag"])
def generate_endpoint(request: GenerateRequest) -> ChatResponse:
    """Generate from only the exact evidence IDs returned by /api/retrieve."""
    category = request.category.strip().lower()
    if category not in VALID_CATEGORIES:
        raise HTTPException(status_code=422, detail="Unsupported knowledge category")
    if not config.generation_configured:
        raise HTTPException(status_code=503, detail="Answer generation is not configured for this deployment")
    envelope = rehydrate_evidence(
        request.message,
        category,
        request.namespace,
        request.index_manifest_hash,
        request.chunk_ids,
    )
    if not envelope.is_ready:
        return ChatResponse(answer=envelope.user_message, citations="", debug="")
    memory = _build_memory(request.history, category)
    answer, citations, debug = generate_from_evidence(envelope, memory)
    sources = [CitationItem(**item) for item in build_citation_records(envelope_chunks(envelope))]
    return ChatResponse(
        answer=answer,
        citations=citations,
        debug=debug,
        generation_provider=generator.active_provider,
        generation_model=generator.active_model,
        sources=sources,
    )


@api.get("/", response_class=HTMLResponse, include_in_schema=False)
def index() -> HTMLResponse:
    """Serve the dependency-free client that works on stateless functions."""
    return HTMLResponse(STATIC_INDEX.read_text(encoding="utf-8"))


app = api
