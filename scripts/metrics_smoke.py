#!/usr/bin/env python
"""Verify foundational telemetry dual-writes JSON and PostgreSQL, then clean up."""

from __future__ import annotations

import tempfile
import sys
from pathlib import Path

import psycopg

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import config
from src.observability import JsonlStore, MetricsRepository, RequestTrace


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="creativa-metrics-") as directory:
        store = JsonlStore(Path(directory) / "metrics.jsonl")
        repository = MetricsRepository(store)
        trace = RequestTrace("metrics smoke test", "all", conversation_id="metrics-smoke", turn_index=1)
        trace.finish("ok")
        try:
            repository.save(trace.serializable())
            if not store.read() or store.read()[0]["trace_id"] != trace.trace_id:
                raise RuntimeError("JSON metrics write could not be read back")
            with psycopg.connect(config.database_url) as connection:
                row = connection.execute(
                    "SELECT payload FROM rag_metric_events WHERE trace_id = %s::uuid",
                    (trace.trace_id,),
                ).fetchone()
                if not row or row[0]["trace_id"] != trace.trace_id:
                    raise RuntimeError("PostgreSQL metrics write could not be read back")
                connection.execute(
                    "DELETE FROM rag_metric_events WHERE trace_id = %s::uuid",
                    (trace.trace_id,),
                )
        finally:
            # Cleanup is restricted to the exact synthetic UUID created above.
            with psycopg.connect(config.database_url) as connection:
                connection.execute(
                    "DELETE FROM rag_metric_events WHERE trace_id = %s::uuid",
                    (trace.trace_id,),
                )
    print("Metrics smoke test passed: JSON and PostgreSQL writes were read back successfully.")


if __name__ == "__main__":
    main()
