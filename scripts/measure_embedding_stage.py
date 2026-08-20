#!/usr/bin/env python
"""Measure one fully ingested dimension for sequential rollout acceptance."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import generate_from_evidence
from src.config import config
from src.embedding_profiles import get_embedding_runtime
from src.evidence_service import envelope_chunks, stage_evidence
from src.generator import generator
from src.index_manifests import load_index_manifest, manifest_matches_runtime
from src.memory import ConversationMemory
from src.quality_metrics import gold_dataset, retrieval_metrics, task_success


def _mean(values: list[float]) -> float | None:
    return round(statistics.fmean(values), 6) if values else None


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    return round(ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)], 2)


def _local_checksum(document_name: str) -> str:
    matches = list(Path(config.data_dir).rglob(document_name))
    if len(matches) != 1:
        return ""
    return hashlib.sha256(matches[0].read_bytes()).hexdigest()


def audit_stage(dimension: int) -> dict:
    runtime = get_embedding_runtime(dimension)
    if config.embedding_dimension != dimension:
        raise ValueError("EMBEDDING_DIMENSION must match --dimension")
    manifest = load_index_manifest(runtime.namespace)
    if manifest is None or not manifest_matches_runtime(manifest, runtime.namespace):
        raise RuntimeError(f"missing or stale manifest for {runtime.namespace}")

    audit = runtime.vector_store.namespace_audit()
    expected_documents = set(manifest.document_checksums)
    actual_documents = set(audit["documents"])
    counts_match = all(
        audit["documents"].get(name, {}).get("chunk_count") == count
        for name, count in manifest.document_chunk_counts.items()
    )
    checksums_match = expected_documents == actual_documents and all(
        _local_checksum(name) == checksum
        for name, checksum in manifest.document_checksums.items()
    )

    return {
        "dimension": dimension,
        "namespace": runtime.namespace,
        "expected_documents": len(manifest.document_checksums),
        "verified_documents": audit["document_count"],
        "verified_chunks": sum(item["chunk_count"] for item in audit["documents"].values()),
        "checksums_match": checksums_match and counts_match,
        "invalid_vector_count": audit["invalid_vector_count"],
    }


def measure(dimension: int, preview_logs_clean: bool) -> dict:
    result = audit_stage(dimension)
    quality_values = {name: [] for name in ("hit_rate_at_5", "recall_at_5", "ndcg_at_5", "task_success")}
    retrieval_latencies: list[float] = []
    case_results: list[dict] = []
    provenance_pass = True
    citations_pass = True
    refusal_pass = True

    for case in gold_dataset()["cases"]:
        started = time.perf_counter()
        envelope = stage_evidence(
            case["query"], case.get("category", "all"), [], embedding_dimension=dimension
        )
        retrieval_latencies.append((time.perf_counter() - started) * 1000)
        chunks = envelope_chunks(envelope)
        if case.get("expect_evidence") and any(chunk.retrieval_mode != "vector" for chunk in chunks):
            raise RuntimeError(
                f"{case['case_id']} used non-vector retrieval; discard this benchmark run"
            )
        retrieval, _ = retrieval_metrics(case, chunks, 5)
        for source_name, target_name in (
            ("hit_rate_at_k", "hit_rate_at_5"),
            ("recall_at_k", "recall_at_5"),
            ("ndcg_at_k", "ndcg_at_5"),
        ):
            metric = retrieval[source_name]
            if metric["applicable"]:
                quality_values[target_name].append(float(metric["value"]))

        if envelope.is_ready:
            answer, citations, _ = generate_from_evidence(envelope, ConversationMemory())
            status = "ok"
            generation_provider = generator.active_provider
        else:
            answer, citations = envelope.user_message, ""
            status = envelope.status
            generation_provider = "not_called"

        trace = {
            "status": status,
            "answer": answer,
            "citations": citations,
            "generation_provider": generation_provider,
            "retrieval_count": len(chunks),
            "retrieved_chunks": [
                {
                    "source_id": chunk.source_id,
                    "document_name": chunk.document_name,
                    "page_number": chunk.page_number,
                    "source_url": chunk.source_url,
                }
                for chunk in chunks
            ],
            "stages_ms": {"retrieval": retrieval_latencies[-1]} if chunks else {},
        }
        task_metric, _ = task_success(case, trace)
        if task_metric["applicable"]:
            quality_values["task_success"].append(float(task_metric["value"]))

        provenance_pass = provenance_pass and all(
            chunk.source_id and chunk.document_name and chunk.page_number > 0
            and chunk.source_url.startswith("https://")
            for chunk in chunks
        )
        if case.get("expect_evidence"):
            expected_sources = {
                str(item.get("source_id", "")) for item in case.get("relevant_items", [])
            }
            citations_pass = citations_pass and envelope.is_ready and bool(citations) and bool(
                expected_sources & {chunk.source_id for chunk in chunks}
            )
        else:
            refusal_pass = refusal_pass and envelope.status == case.get("expected_status") and not citations

        case_results.append(
            {
                "case_id": case["case_id"],
                "status": status,
                "retrieval_count": len(chunks),
                "generation_provider": generation_provider,
                "task_success": task_metric["value"] if task_metric["applicable"] else None,
            }
        )

    result.update({
        "quality": {name: _mean(values) for name, values in quality_values.items()},
        "retrieval_p95_ms": _p95(retrieval_latencies),
        "provenance_pass": provenance_pass,
        "citations_pass": citations_pass,
        "refusal_pass": refusal_pass,
        "preview_logs_clean": preview_logs_clean,
        "cases": case_results,
    })
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimension", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preview-logs-clean", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    result = audit_stage(args.dimension) if args.audit_only else measure(args.dimension, args.preview_logs_clean)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "cases"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
