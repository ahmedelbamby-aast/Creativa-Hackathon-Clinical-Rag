#!/usr/bin/env python
"""Exercise pgvector insert, search, stats, and delete with temporary records."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.vector_store import VectorStore


def unit_vector(first: float, second: float) -> list[float]:
    vector = [0.0] * 384
    vector[0] = first
    vector[1] = second
    return vector


def record(chunk_id: str, text: str, category: str) -> dict:
    return {
        "chunk_id": chunk_id,
        "document_name": "database-smoke-test.txt",
        "page_number": 1,
        "section_title": "Smoke test",
        "subsection_title": "",
        "category": category,
        "content_type": "text",
        "language": "en",
        "text": text,
        "char_count": len(text),
        "word_count": len(text.split()),
        "quality_score": 1.0,
    }


def main() -> None:
    store = VectorStore(namespace="smoke_test", dimension=384)
    document_name = "database-smoke-test.txt"
    store.ensure_schema()
    store.delete_document(document_name)

    try:
        chunks = [
            record("smoke-relevant", "Relevant diabetes guidance", "treatment"),
            record("smoke-other", "Different reference material", "prevention"),
        ]
        embeddings = [unit_vector(1.0, 0.0), unit_vector(0.0, 1.0)]
        store.add_chunks(chunks, embeddings)

        if not store.has_document(document_name):
            raise RuntimeError("inserted smoke-test document was not found")
        results = store.query(unit_vector(1.0, 0.0), top_k=1)
        if not results or results[0]["id"] != "smoke-relevant":
            raise RuntimeError(f"unexpected nearest-neighbor result: {results}")
        if results[0]["score"] < 0.99:
            raise RuntimeError(f"unexpected similarity score: {results[0]['score']}")
        print("Database smoke test passed: insert, cosine search, and stats succeeded.")
    finally:
        store.delete_document(document_name)

    if store.has_document(document_name):
        raise RuntimeError("smoke-test cleanup failed")
    print("Database smoke-test records were removed.")


if __name__ == "__main__":
    main()
