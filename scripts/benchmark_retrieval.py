#!/usr/bin/env python
"""Run, review, and finalize the Phase 2 retrieval benchmark.

Examples:
  uv run python scripts/benchmark_retrieval.py --prepare-indexes --run-dir reports/retrieval/run-001
  # Two reviewers complete review-labels.csv.
  uv run python scripts/benchmark_retrieval.py --finalize reports/retrieval/run-001
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import CHUNK_PROFILES, config
from src.index_manifests import index_manifest_hash, load_index_manifest, manifest_matches_runtime
from src.retrieval_benchmark import (
    CandidateResult,
    build_review_labels,
    calculate_metrics,
    estimate_embedding_cost_usd,
    load_retrieval_cases,
    require_cross_review,
    select_candidate,
)
from src.retrieval_contracts import EvidenceChunk
from src.source_catalog import load_source_catalog, require_catalog_documents


PROVIDERS = ("local", "gemini")
FIELDNAMES = [
    "run_id", "case_id", "rank", "chunk_id", "relevance", "reviewer_a", "reviewer_b",
    "reviewer_a_label", "reviewer_b_label", "rationale", "suggested_relevance", "document_name",
    "page_number", "section_title", "score", "text",
]


def experiment_namespace(profile: str, provider: str) -> str:
    return f"phase2_{profile}_{provider}_384"


def experiment_key(profile: str, provider: str) -> str:
    return f"{profile}-{provider}-384"


def _env_for_experiment(profile: str, provider: str) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "EMBEDDING_PROVIDER": provider,
            "ACTIVE_INDEX_NAMESPACE": experiment_namespace(profile, provider),
            "EMBEDDING_NAMESPACE": "",
            "EMBEDDING_DIMENSION": "384",
            "RETRIEVAL_PROFILE": profile,
            "TOP_K": "5",
        }
    )
    return environment


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def _to_evidence_chunk(chunk) -> EvidenceChunk:
    return EvidenceChunk(
        chunk_id=chunk.chunk_id,
        text=chunk.text,
        score=chunk.score,
        distance=chunk.distance,
        document_name=chunk.document_name,
        page_number=chunk.page_number,
        section_title=chunk.section_title,
        subsection_title=chunk.subsection_title,
        category=chunk.category,
        language=chunk.language,
        source_id=chunk.source_id,
        source_url=chunk.source_url,
    )


def run_single(output: Path, profile: str, provider: str) -> None:
    """Run one immutable profile/provider ranking pass in its configured process."""
    from src.retriever import retrieve
    from src.vector_store import vector_store

    config.validate()
    if config.retrieval_profile != profile or config.embedding_provider != provider:
        raise ValueError("single benchmark process configuration does not match requested experiment")
    cases = load_retrieval_cases()
    catalog = load_source_catalog()
    require_catalog_documents(
        {case.expected_document_name for case in cases if case.expect_evidence}, catalog
    )
    manifest = load_index_manifest(config.resolved_embedding_namespace)
    if manifest is None or not manifest_matches_runtime(manifest, config.resolved_embedding_namespace):
        raise RuntimeError("missing or stale index manifest for experiment namespace")
    vector_store.healthcheck()

    rankings: dict[str, list[dict]] = {}
    latency_ms: list[float] = []
    for case in cases:
        started = time.perf_counter()
        chunks = retrieve(case.query, category=case.category, top_k=5, similarity_threshold=0.0)
        latency_ms.append((time.perf_counter() - started) * 1000)
        rankings[case.case_id] = [asdict(_to_evidence_chunk(chunk)) for chunk in chunks]

    query_tokens = sum(len(case.query.split()) for case in cases)
    result = {
        "run_id": experiment_key(profile, provider),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "profile": profile,
        "provider": provider,
        "model": manifest.embedding_model,
        "namespace": manifest.namespace,
        "index_manifest_hash": index_manifest_hash(manifest),
        "latency_ms": latency_ms,
        "mean_latency_ms": round(sum(latency_ms) / len(latency_ms), 2),
        "p95_latency_ms": round(sorted(latency_ms)[max(0, int(len(latency_ms) * 0.95) - 1)], 2),
        "embedding_calls": len(cases),
        "estimated_cost_usd": estimate_embedding_cost_usd(provider, manifest.token_count + query_tokens),
        "rankings": rankings,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")


def prepare_indexes(run_dir: Path, providers: tuple[str, ...]) -> None:
    """Create six isolated indexes without touching the active application namespace."""
    for profile in CHUNK_PROFILES:
        for provider in providers:
            environment = _env_for_experiment(profile, provider)
            command = [
                sys.executable,
                str(ROOT / "scripts" / "ingest.py"),
                "--force",
                "--write-index-manifest",
            ]
            subprocess.run(command, cwd=ROOT, env=environment, check=True)
    (run_dir / "prepared-at.txt").write_text(
        datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8"
    )


def run_grid(run_dir: Path, providers: tuple[str, ...]) -> None:
    """Run all six retrieval experiments and produce editable review labels."""
    cases = load_retrieval_cases()
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    label_rows: list[dict] = []
    for profile in CHUNK_PROFILES:
        for provider in providers:
            key = experiment_key(profile, provider)
            result_path = raw_dir / f"{key}.json"
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--single-output",
                str(result_path),
                "--profile",
                profile,
                "--provider",
                provider,
            ]
            subprocess.run(command, cwd=ROOT, env=_env_for_experiment(profile, provider), check=True)
            raw = json.loads(result_path.read_text(encoding="utf-8"))
            rankings = {
                case_id: [EvidenceChunk(**chunk) for chunk in chunks]
                for case_id, chunks in raw["rankings"].items()
            }
            label_rows.extend(build_review_labels(key, rankings, cases))
    with (run_dir / "review-labels.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(label_rows)


def _read_labels(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_failure_report(run_dir: Path, reason: str) -> int:
    """Persist a machine- and human-readable failed gate result."""
    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "results": [],
        "selection": {"accepted": False, "error": reason},
    }
    (run_dir / "results.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (run_dir / "summary.md").write_text(
        "# Phase 2 Retrieval Benchmark\n\n"
        f"Run: `{run_dir.name}`\n\n"
        f"**FAILED:** {reason}\n",
        encoding="utf-8",
    )
    return 1


def finalize(run_dir: Path) -> int:
    """Calculate results after cross-review and enforce Phase 2 acceptance gates."""
    cases = load_retrieval_cases()
    labels = _read_labels(run_dir / "review-labels.csv")
    try:
        require_cross_review(labels)
    except ValueError as error:
        return _write_failure_report(run_dir, str(error))
    by_run: dict[str, list[dict]] = {}
    for label in labels:
        by_run.setdefault(label["run_id"], []).append(label)

    candidates: list[CandidateResult] = []
    full_results: list[dict] = []
    for raw_path in sorted((run_dir / "raw").glob("*.json")):
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        rankings = {
            case_id: [EvidenceChunk(**chunk) for chunk in chunks]
            for case_id, chunks in raw["rankings"].items()
        }
        metrics = calculate_metrics(rankings, cases, by_run.get(raw["run_id"], []))
        per_k = {int(key): value for key, value in metrics["macro_by_k"].items()}
        candidate = CandidateResult(
            profile=raw["profile"],
            provider=raw["provider"],
            model=raw["model"],
            precision_at_5=per_k[5]["macro_precision_at_k"],
            hit_at_5=per_k[5]["hit_at_k"],
            mean_latency_ms=raw["mean_latency_ms"],
            estimated_cost_usd=raw["estimated_cost_usd"],
            refusal_pass_rate=per_k[5]["no_evidence_refusal_pass_rate"],
            labels_complete=True,
            per_k=per_k,
        )
        candidates.append(candidate)
        full_results.append({"raw": raw, "metrics": metrics, "candidate": asdict(candidate)})

    output = {"results": full_results, "timestamp": datetime.now(timezone.utc).isoformat()}
    try:
        winner, runtime_k = select_candidate(candidates)
        output["selection"] = {"winner": asdict(winner), "runtime_k": runtime_k, "accepted": True}
        exit_code = 0
    except ValueError as error:
        output["selection"] = {"accepted": False, "error": str(error)}
        exit_code = 1
    (run_dir / "results.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = ["# Phase 2 Retrieval Benchmark", "", f"Run: `{run_dir.name}`", "", "| Profile | Provider | P@5 | Hit@5 | Latency | Cost |", "|---|---|---:|---:|---:|---:|"]
    for result in full_results:
        candidate = result["candidate"]
        lines.append(
            f"| {candidate['profile']} | {candidate['provider']} | {candidate['precision_at_5']:.3f} | "
            f"{candidate['hit_at_5']:.3f} | {candidate['mean_latency_ms']:.1f} ms | "
            f"${candidate['estimated_cost_usd']:.6f} |"
        )
    if output["selection"]["accepted"]:
        selected = output["selection"]
        lines.extend(["", f"Selected: `{selected['winner']['profile']}` / `{selected['winner']['provider']}` / k={selected['runtime_k']}"])
    else:
        lines.extend(["", f"**FAILED:** {output['selection']['error']}"])
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return exit_code


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 2 retrieval benchmark")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--prepare-indexes", action="store_true")
    parser.add_argument(
        "--providers",
        nargs="+",
        choices=PROVIDERS,
        default=PROVIDERS,
        help="Embedding providers to benchmark (default: local gemini)",
    )
    parser.add_argument("--finalize", type=Path)
    parser.add_argument("--single-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--profile", choices=tuple(CHUNK_PROFILES), help=argparse.SUPPRESS)
    parser.add_argument("--provider", choices=PROVIDERS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.single_output:
        if not args.profile or not args.provider:
            parser.error("single experiment requires profile and provider")
        run_single(args.single_output, args.profile, args.provider)
        return 0
    if args.finalize:
        return finalize(args.finalize)
    if not args.run_dir:
        parser.error("--run-dir is required unless using --finalize")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    providers = tuple(args.providers)
    if args.prepare_indexes:
        prepare_indexes(args.run_dir, providers)
    run_grid(args.run_dir, providers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
