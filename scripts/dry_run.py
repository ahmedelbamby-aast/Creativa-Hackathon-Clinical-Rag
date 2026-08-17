#!/usr/bin/env python
"""Validate files, modules, embeddings, pgvector, and UI without ingestion."""

import importlib
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ALL_CATEGORIES, config
from src.embeddings import embedder
from src.ingestion.chunker_adapter import chunk_elements
from src.rewriter import rewrite_query
from src.router import route_query
from src.safety import SafetyLevel, classify_safety
from src.vector_store import vector_store


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def check_required_files() -> str:
    required = [
        "app.py",
        "compose.yaml",
        "database/schema.sql",
        "pyproject.toml",
        "uv.lock",
        ".env.development.example",
        ".env.deployment.example",
        "backend/server.py",
        "vercel.json",
        "scripts/ingest.py",
        "scripts/evaluate.py",
        "src/config.py",
        "src/embeddings.py",
        "src/retriever.py",
        "src/vector_store.py",
    ]
    missing = [name for name in required if not (PROJECT_ROOT / name).is_file()]
    if missing:
        raise RuntimeError(f"missing files: {', '.join(missing)}")
    return f"{len(required)} required files found"


def check_environment_template() -> str:
    required = {
        "DATABASE_URL",
        "EMBEDDING_DIMENSION",
        "EMBEDDING_MODEL",
        "EMBEDDING_PROVIDER",
        "GEMINI_API_KEY",
        "GEMINI_MODEL",
        "ONLINE_EMBEDDING_MODEL",
        "TOP_K",
        "SIMILARITY_THRESHOLD",
    }
    lines = []
    for name in (".env.development.example", ".env.deployment.example"):
        lines.extend((PROJECT_ROOT / name).read_text(encoding="utf-8").splitlines())
    available = {
        line.split("=", 1)[0]
        for line in lines
        if "=" in line and not line.startswith("#")
    }
    missing = sorted(required - available)
    if missing:
        raise RuntimeError(f"missing environment keys: {', '.join(missing)}")
    return f"{len(required)} environment keys documented"


def check_runtime_imports() -> str:
    modules = [
        "google.genai",
        "gradio",
        "pgvector",
        "psycopg",
        "pymupdf",
        "pypdf",
    ]
    if config.embedding_provider == "local":
        modules.append("sentence_transformers")
    for module in modules:
        importlib.import_module(module)
    return f"{len(modules)} runtime imports succeeded"


def check_pipeline_logic() -> str:
    assert route_query("What food and meal choices are recommended?") == "nutrition"
    assert "diabetes" in rewrite_query("What about fruit?").lower()
    assert classify_safety("I have chest pain and can't breathe") == SafetyLevel.EMERGENCY
    elements = [
        {
            "document_name": "dry-run.txt",
            "page_number": 1,
            "section_title": "Nutrition",
            "subsection_title": "Meal planning",
            "content": (
                "Diabetes meal planning can include vegetables, whole grains, "
                "fibre, and appropriate portions. "
            ) * 4,
            "content_type": "text",
        }
    ]
    chunks = chunk_elements(elements, document_language="en")
    if not chunks or chunks[0]["category"] != "nutrition":
        raise RuntimeError("parser-to-chunker metadata contract failed")
    return f"pipeline logic produced {len(chunks)} valid chunk(s)"


def check_embedding() -> str:
    vectors = embedder.embed_batch(
        ["diabetes nutrition guidance", "diabetes prevention guidance"]
    )
    dimension = embedder.dimension
    if len(vectors) != 2 or any(len(vector) != dimension for vector in vectors):
        raise RuntimeError("embedding output dimensions are inconsistent")
    if any(not all(math.isfinite(value) for value in vector) for vector in vectors):
        raise RuntimeError("embedding contains non-finite values")
    return f"{embedder.provider}/{embedder.model_name}: {dimension} dimensions"


def check_postgres() -> str:
    vector_store.ensure_schema()
    versions = vector_store.healthcheck()
    probe = embedder.embed_query("diabetes retrieval probe")
    results = vector_store.query(probe, top_k=1)
    stats = vector_store.collection_stats()
    if set(stats) != set(ALL_CATEGORIES):
        raise RuntimeError(f"unexpected category stats: {stats}")
    return (
        f"PostgreSQL {versions['postgres']}, pgvector {versions['pgvector']}, "
        f"namespace={vector_store.namespace}, probe_results={len(results)}"
    )


def check_ui() -> str:
    from app import build_ui

    demo = build_ui()
    if demo.__class__.__name__ != "Blocks":
        raise RuntimeError("Gradio UI did not build as Blocks")
    return "Gradio UI constructed without launching a server"


CHECKS = [
    ("files", check_required_files),
    ("environment", check_environment_template),
    ("imports", check_runtime_imports),
    ("pipeline", check_pipeline_logic),
    ("embedding", check_embedding),
    ("postgres", check_postgres),
    ("ui", check_ui),
]


def main() -> None:
    failures = []
    for name, check in CHECKS:
        try:
            detail = check()
            print(f"[PASS] {name}: {detail}")
        except Exception as exc:
            failures.append(name)
            print(f"[FAIL] {name}: {exc}")
    if failures:
        raise SystemExit(f"Dry run failed: {', '.join(failures)}")
    print("Dry run passed. No documents were ingested and Gemini generation was not called.")


if __name__ == "__main__":
    main()
