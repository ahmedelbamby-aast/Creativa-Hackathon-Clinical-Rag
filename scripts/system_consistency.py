#!/usr/bin/env python
"""Deterministic local system checks with optional live Gemini generation.

The default checks are read-only apart from Gradio's session-local clear call.
They never print API keys or database credentials.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gradio_client import Client

from src.config import config
from src.embeddings import embedder
from src.generator import generator
from src.retriever import retrieve
from src.vector_store import vector_store


FIXED_QUERY = "How can diabetes complications be prevented?"
EXPECTED_GRADIO_APIS = {"/on_send", "/on_send_1", "/on_clear", "/create_memory"}


def check(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def check_configuration(require_key: bool) -> None:
    config.validate()
    check(config.gemini_model == "gemini-3.6-flash", "GEMINI_MODEL must be gemini-3.6-flash")
    check(config.embedding_provider == "local", "Deterministic tests require local embeddings")
    check(config.embedding_dimension == 384, "EMBEDDING_DIMENSION must be 384")
    check(config.resolved_embedding_namespace == "local_384", "Namespace must be local_384")
    if require_key:
        check(bool(config.gemini_api_key.strip()), "GEMINI_API_KEY is required for live checks")
    print("[PASS] configuration: gemini-3.6-flash, local_384, 384 dimensions")


def check_database_and_retrieval() -> None:
    versions = vector_store.healthcheck()
    stats = vector_store.collection_stats()
    total = sum(stats.values())
    check(bool(versions.get("postgres")), "PostgreSQL version was not reported")
    check(bool(versions.get("pgvector")), "pgvector version was not reported")
    check(total > 0, "No indexed chunks exist in local_384")

    results = retrieve(FIXED_QUERY, category="prevention", top_k=3, similarity_threshold=0.0)
    check(len(results) == 3, f"Expected 3 retrieval results, received {len(results)}")
    check(all(result.document_name for result in results), "A result has no document name")
    check(all(result.page_number > 0 for result in results), "A result has an invalid page number")
    check(all(0.0 <= result.score <= 1.0 for result in results), "A score is outside [0, 1]")
    check(
        [result.score for result in results] == sorted((result.score for result in results), reverse=True),
        "Retrieval results are not sorted by descending similarity",
    )
    print(
        f"[PASS] database/retrieval: PostgreSQL {versions['postgres']}, "
        f"pgvector {versions['pgvector']}, {total} chunks, best score {results[0].score:.3f}"
    )


def check_gradio(base_url: str) -> None:
    with urlopen(base_url, timeout=10) as response:
        body = response.read().decode("utf-8", errors="replace")
        check(response.status == 200, f"Gradio returned HTTP {response.status}")
        check("gradio" in body.lower(), "Gradio marker missing from the HTML response")

    client = Client(base_url, verbose=False)
    api_names = {endpoint.api_name for endpoint in client.endpoints.values() if endpoint.api_name}
    missing = EXPECTED_GRADIO_APIS - api_names
    check(not missing, f"Missing Gradio APIs: {sorted(missing)}")
    cleared = client.predict(api_name="/on_clear")
    check(len(cleared) == 4, "Clear API returned an unexpected output count")
    check(cleared[0] == [], "Clear API did not empty chat history")
    check("Sources will appear" in cleared[1], "Clear API did not reset citations")
    print(f"[PASS] gradio: HTTP 200 and {len(EXPECTED_GRADIO_APIS)} required APIs available")


def check_live_gemini() -> None:
    response = generator.generate(
        "Reply with exactly READY and no other text. This is a deterministic API readiness check."
    ).strip()
    check(response == "READY", f"Gemini readiness response was {response!r}, expected 'READY'")
    print("[PASS] Gemini API: exact READY response from gemini-3.6-flash")


def check_live_gradio_query(base_url: str) -> None:
    client = Client(base_url, verbose=False)
    result = client.predict(
        FIXED_QUERY,
        [],
        "🛡️ Prevention",
        api_name="/on_send",
    )
    check(len(result) == 4, "Send API returned an unexpected output count")
    messages, citations, _, cleared_input = result
    check([message["role"] for message in messages] == ["user", "assistant"], "Unexpected chat roles")
    assistant_parts = messages[-1].get("content", [])
    assistant_text = " ".join(part.get("text", "") for part in assistant_parts if part.get("type") == "text")
    check(len(assistant_text) >= 100, "Generated answer was unexpectedly short")
    error_markers = ("Configuration Error", "Generation error", "An error occurred during generation")
    check(not any(marker in assistant_text for marker in error_markers), "Generated answer contains an error")
    check("Sources" in citations and "Page" in citations, "Send API returned no grounded citations")
    check(cleared_input == "", "Send API did not clear the query input")
    print("[PASS] Gradio send API: grounded answer, citations, roles, and input reset verified")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gradio-url", default="http://127.0.0.1:7860")
    parser.add_argument("--live-gemini", action="store_true", help="Make one live Gemini request")
    parser.add_argument("--live-gradio", action="store_true", help="Make one end-to-end Gradio query")
    args = parser.parse_args()

    check_configuration(require_key=args.live_gemini)
    check_database_and_retrieval()
    check_gradio(args.gradio_url)
    if args.live_gemini:
        check_live_gemini()
    if args.live_gradio:
        check_live_gradio_query(args.gradio_url)
    print("System consistency checks passed.")


if __name__ == "__main__":
    main()
