"""FastAPI/ASGI entrypoint used by Vercel's Python runtime."""

import logging

import gradio as gr
from fastapi import FastAPI, HTTPException

from app import build_ui
from src.config import config
from src.vector_store import vector_store


logger = logging.getLogger(__name__)

api = FastAPI(
    title="Creativa Diabetes RAG",
    description="Health and readiness endpoints for the deployed Gradio RAG application.",
    version="0.1.0",
)


@api.get("/api/health", tags=["operations"])
def health() -> dict[str, object]:
    """Return process-level health without contacting external services."""
    return {
        "status": "ok",
        "environment": config.app_env,
        "embedding_provider": config.embedding_provider,
        "embedding_namespace": config.resolved_embedding_namespace,
        "database_configured": bool(config.database_url),
        "gemini_configured": bool(config.gemini_api_key),
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


app = gr.mount_gradio_app(
    api,
    build_ui(),
    path="/",
    show_error=not config.is_deployment,
    enable_monitoring=False,
)
