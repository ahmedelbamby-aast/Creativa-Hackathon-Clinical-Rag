"""Deterministic, provenance-anchored metric and selection logic for Phase 2."""

from __future__ import annotations

import json
import re
import statistics
import unicodedata
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from src.config import config
from src.retrieval_contracts import EvidenceChunk, RelevanceLabel, RetrievalCase


CASES_PATH = config.project_root / "data" / "retrieval_cases.json"
K_VALUES = (3, 4, 5)
GEMINI_INPUT_COST_PER_MILLION_USD = 0.20
CERTIFIED_ANCHOR_ORACLE = "certified_anchor_oracle"


@dataclass(frozen=True)
class QueryMetrics:
    case_id: str
    k: int
    precision_at_k: float
    hit_at_k: float
    recall_at_k: float
    refusal_pass: bool | None


@dataclass(frozen=True)
class CandidateResult:
    profile: str
    provider: str
    model: str
    precision_at_5: float
    hit_at_5: float
    mean_latency_ms: float
    estimated_cost_usd: float
    refusal_pass_rate: float
    labels_complete: bool
    per_k: dict[int, dict[str, float]]

    @property
    def eligible(self) -> bool:
        return (
            self.precision_at_5 >= 0.40
            and self.hit_at_5 >= 0.90
            and self.refusal_pass_rate == 1.0
            and self.labels_complete
        )


def normalize_text(value: str) -> str:
    """Normalize visible text while retaining Arabic and Latin word content."""
    value = unicodedata.normalize("NFKC", value).lower()
    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)
    return " ".join(value.split())


def load_retrieval_cases(path: Path = CASES_PATH) -> list[RetrievalCase]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    cases = raw.get("cases", raw)
    parsed = [
        RetrievalCase(
            case_id=item["case_id"],
            query=item["query"],
            language=item["language"],
            category=item["category"],
            expect_evidence=bool(item["expect_evidence"]),
            expected_source_id=item.get("expected_source_id", ""),
            expected_document_name=item.get("expected_document_name", ""),
            expected_page=int(item.get("expected_page", 0)),
            expected_section=item.get("expected_section", ""),
            text_anchors=tuple(item.get("text_anchors", [])),
        )
        for item in cases
    ]
    if len({case.case_id for case in parsed}) != len(parsed):
        raise ValueError("retrieval case IDs must be unique")
    positives = [case for case in parsed if case.expect_evidence]
    negatives = [case for case in parsed if not case.expect_evidence]
    if len(positives) < 20 or len(negatives) < 8:
        raise ValueError("Phase 2 requires at least 20 positive and 8 no-evidence cases")
    if any(not case.text_anchors for case in positives):
        raise ValueError("positive retrieval cases require at least one stable text anchor")
    return parsed


def certified_anchor_relevance(case: RetrievalCase, chunk: EvidenceChunk) -> str:
    """Label a result solely from the case's immutable certified provenance anchors."""
    if not case.expect_evidence:
        return "not_relevant"
    provenance_matches = (
        chunk.source_id == case.expected_source_id
        and chunk.document_name == case.expected_document_name
        and chunk.page_number == case.expected_page
    )
    section_matches = (
        not case.expected_section
        or normalize_text(case.expected_section) in normalize_text(chunk.section_title)
    )
    text = normalize_text(chunk.text)
    anchor_matches = any(normalize_text(anchor) in text for anchor in case.text_anchors)
    return "relevant" if provenance_matches and section_matches and anchor_matches else "not_relevant"


def suggested_relevance(case: RetrievalCase, chunk: EvidenceChunk) -> str:
    """Backward-compatible name for the deterministic certified-anchor oracle."""
    return certified_anchor_relevance(case, chunk)


def build_review_labels(
    run_id: str,
    rankings: dict[str, list[EvidenceChunk]],
    cases: Iterable[RetrievalCase],
) -> list[dict]:
    """Create finalized, auditable labels from certified case anchors.

    This is intentionally not a clinical judgment: a result is relevant only
    when its immutable source, document, page, section, and text anchor all
    match the evaluation case.
    """
    by_id = {case.case_id: case for case in cases}
    rows: list[dict] = []
    for case_id, chunks in rankings.items():
        case = by_id[case_id]
        for rank, chunk in enumerate(chunks[:5], start=1):
            relevance = certified_anchor_relevance(case, chunk)
            row = asdict(
                RelevanceLabel(
                    run_id,
                    case_id,
                    rank,
                    chunk.chunk_id,
                    relevance=relevance,
                    rationale="deterministic certified source/page/section/anchor match",
                )
            )
            row.update(
                {
                    "label_method": CERTIFIED_ANCHOR_ORACLE,
                    "suggested_relevance": relevance,
                    "document_name": chunk.document_name,
                    "page_number": chunk.page_number,
                    "section_title": chunk.section_title,
                    "score": chunk.score,
                    "text": chunk.text,
                }
            )
            rows.append(row)
    return rows


def _label_lookup(labels: Iterable[dict]) -> dict[tuple[str, int], str]:
    lookup: dict[tuple[str, int], str] = {}
    for label in labels:
        key = (str(label["case_id"]), int(label["rank"]))
        relevance = str(label.get("relevance", "unjudged"))
        if relevance not in {"relevant", "not_relevant", "unjudged"}:
            raise ValueError(f"invalid relevance label for {key}")
        lookup[key] = relevance
    return lookup


def require_complete_labels(rankings: dict[str, list[EvidenceChunk]], labels: Iterable[dict]) -> None:
    lookup = _label_lookup(labels)
    missing = [
        f"{case_id}#{rank}"
        for case_id, chunks in rankings.items()
        for rank, _ in enumerate(chunks[:5], start=1)
        if lookup.get((case_id, rank), "unjudged") == "unjudged"
    ]
    if missing:
        raise ValueError("unjudged top-five relevance labels: " + _preview_items(missing))


def _preview_items(items: list[str], limit: int = 10) -> str:
    """Keep CLI failure reports readable while preserving details in their CSV artifact."""
    preview = ", ".join(items[:limit])
    remaining = len(items) - limit
    return preview + (f" … and {remaining} more" if remaining > 0 else "")


def require_cross_review(labels: Iterable[dict]) -> None:
    """Accept deterministic certified-anchor labels or legacy human cross-review."""
    incomplete = []
    for label in labels:
        final = str(label.get("relevance", "unjudged"))
        reviewer_a = str(label.get("reviewer_a", "")).strip()
        reviewer_b = str(label.get("reviewer_b", "")).strip()
        decision_a = str(label.get("reviewer_a_label", "unjudged"))
        decision_b = str(label.get("reviewer_b_label", "unjudged"))
        if label.get("label_method") == CERTIFIED_ANCHOR_ORACLE:
            if final in {"relevant", "not_relevant"}:
                continue
            incomplete.append(f"{label.get('case_id')}#{label.get('rank')}")
            continue
        if (
            not reviewer_a
            or not reviewer_b
            or decision_a != final
            or decision_b != final
            or final == "unjudged"
        ):
            incomplete.append(f"{label.get('case_id')}#{label.get('rank')}")
    if incomplete:
        raise ValueError("unresolved cross-review labels: " + _preview_items(incomplete))


def calculate_metrics(
    rankings: dict[str, list[EvidenceChunk]],
    cases: Iterable[RetrievalCase],
    labels: Iterable[dict],
    k_values: tuple[int, ...] = K_VALUES,
) -> dict:
    """Calculate per-query and macro IR metrics from complete human labels."""
    case_list = list(cases)
    require_complete_labels(rankings, labels)
    lookup = _label_lookup(labels)
    per_query: list[QueryMetrics] = []
    macro_by_k: dict[int, dict[str, float]] = {}
    for k in k_values:
        positives: list[QueryMetrics] = []
        negatives: list[QueryMetrics] = []
        for case in case_list:
            relevant = [
                lookup.get((case.case_id, rank), "not_relevant") == "relevant"
                for rank in range(1, k + 1)
            ]
            relevant_count = sum(relevant)
            if case.expect_evidence:
                values = QueryMetrics(
                    case.case_id,
                    k,
                    round(relevant_count / k, 4),
                    float(any(relevant)),
                    float(any(relevant)),
                    None,
                )
                positives.append(values)
            else:
                values = QueryMetrics(
                    case.case_id,
                    k,
                    0.0,
                    0.0,
                    0.0,
                    not any(relevant),
                )
                negatives.append(values)
            per_query.append(values)
        macro_by_k[k] = {
            "macro_precision_at_k": round(statistics.fmean(item.precision_at_k for item in positives), 4),
            "hit_at_k": round(statistics.fmean(item.hit_at_k for item in positives), 4),
            "macro_recall_at_k": round(statistics.fmean(item.recall_at_k for item in positives), 4),
            "no_evidence_refusal_pass_rate": round(
                statistics.fmean(float(bool(item.refusal_pass)) for item in negatives), 4
            ),
        }
    return {
        "per_query": [asdict(item) for item in per_query],
        "macro_by_k": macro_by_k,
    }


def estimate_embedding_cost_usd(provider: str, input_tokens: int) -> float:
    """Estimate text embedding cost; local provider has no per-call API cost."""
    if provider == "local":
        return 0.0
    if provider != "gemini":
        raise ValueError(f"unsupported embedding provider: {provider}")
    return round((input_tokens / 1_000_000) * GEMINI_INPUT_COST_PER_MILLION_USD, 8)


def select_candidate(candidates: Iterable[CandidateResult]) -> tuple[CandidateResult, int]:
    """Apply the Phase 2 acceptance and deterministic model/profile/k rules."""
    eligible = [candidate for candidate in candidates if candidate.eligible]
    if not eligible:
        raise ValueError("no eligible candidate meets Phase 2 acceptance thresholds")

    by_provider: dict[str, list[CandidateResult]] = defaultdict(list)
    for candidate in eligible:
        by_provider[candidate.provider].append(candidate)

    provider_winners = []
    for entries in by_provider.values():
        provider_winners.append(
            sorted(
                entries,
                key=lambda item: (
                    -item.precision_at_5,
                    -item.hit_at_5,
                    item.mean_latency_ms,
                    item.estimated_cost_usd,
                    item.profile,
                ),
            )[0]
        )
    if len(provider_winners) == 1:
        winner = provider_winners[0]
    else:
        local = next((item for item in provider_winners if item.provider == "local"), None)
        gemini = next((item for item in provider_winners if item.provider == "gemini"), None)
        if local is None or gemini is None:
            winner = sorted(provider_winners, key=lambda item: (-item.precision_at_5, item.provider))[0]
        elif abs(local.precision_at_5 - gemini.precision_at_5) >= 0.05:
            winner = local if local.precision_at_5 > gemini.precision_at_5 else gemini
        else:
            winner = sorted(
                (local, gemini),
                key=lambda item: (item.estimated_cost_usd, item.mean_latency_ms, 0 if item.provider == "local" else 1),
            )[0]

    valid_k = [
        k for k, metrics in winner.per_k.items()
        if metrics["hit_at_k"] >= 0.80
    ]
    if not valid_k:
        raise ValueError("winning candidate has no runtime k with Hit@k >= 0.80")
    runtime_k = sorted(
        valid_k,
        key=lambda k: (-winner.per_k[k]["macro_precision_at_k"], k, winner.mean_latency_ms),
    )[0]
    return winner, runtime_k
