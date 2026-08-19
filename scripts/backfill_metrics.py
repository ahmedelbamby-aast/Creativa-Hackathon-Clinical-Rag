#!/usr/bin/env python
"""Idempotently backfill metrics only where old trace data supports it."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.observability import METRIC_IMPLEMENTATION_VERSION, RequestTrace, metrics_repository


def backfill_records(records: list[dict], *, dry_run: bool, save=metrics_repository.save) -> dict[str, int | bool]:
    """Backfill supplied records without inventing labels absent from the trace."""
    updated = skipped = unavailable = 0
    for record in records:
        if record.get("metric_implementation_version") == METRIC_IMPLEMENTATION_VERSION:
            skipped += 1
            continue
        trace = RequestTrace.from_record(record)
        trace.finish(record.get("status", "ok"), record.get("error", ""))
        patched = trace.serializable()
        patched["backfilled_at"] = datetime.now(timezone.utc).isoformat()
        unavailable += sum(
            not value.get("applicable", False)
            for value in patched.get("quality_metrics", {}).values()
            if isinstance(value, dict)
        )
        if not dry_run:
            save(patched)
        updated += 1
    return {"updated": updated, "skipped": skipped, "unavailable_metrics": unavailable, "dry_run": dry_run}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=2000)
    args = parser.parse_args()
    result = backfill_records(metrics_repository.read(limit=max(1, args.limit)), dry_run=args.dry_run)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
