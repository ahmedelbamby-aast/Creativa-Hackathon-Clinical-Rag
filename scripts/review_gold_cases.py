#!/usr/bin/env python
"""Validate reviewed-label data and produce a human-review queue.

This command never changes a review status. Medical labels become usable only
after an authorized human updates the JSON data under source control.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.quality_metrics import gold_dataset, normalize_text
from src.source_catalog import load_source_catalog


def _contains_arabic(value: object) -> bool:
    return any("\u0600" <= character <= "\u06ff" for character in str(value))


def validate() -> list[dict[str, str]]:
    dataset = gold_dataset()
    catalog = load_source_catalog()
    seen_ids: set[str] = set()
    seen_queries: set[str] = set()
    issues: list[dict[str, str]] = []
    for case in dataset["cases"]:
        case_id = str(case.get("case_id", ""))
        def add(code: str, detail: str) -> None:
            issues.append({
                "case_id": case_id,
                "language": str(case.get("language", "")),
                "category": str(case.get("category", "")),
                "query": str(case.get("query", "")),
                "expected_status": str(case.get("expected_status", "")),
                "review_status": str(case.get("review", {}).get("status", "")),
                "relevant_source_ids": "; ".join(
                    str(item.get("source_id", "")) for item in case.get("relevant_items", []) if item.get("source_id")
                ),
                "issue": code,
                "detail": detail,
            })
        if not case_id or case_id in seen_ids:
            add("duplicate_or_missing_case_id", case_id or "missing")
        seen_ids.add(case_id)
        for text in [case.get("query", ""), *case.get("query_variants", [])]:
            normalized = normalize_text(str(text))
            if not normalized:
                add("empty_query_or_variant", "")
            elif normalized in seen_queries:
                add("duplicate_query_or_variant", str(text))
            seen_queries.add(normalized)
        if case.get("language") not in {"en", "ar"}:
            add("invalid_language", str(case.get("language")))
        reviewed = case.get("review", {}).get("status") == "reviewed"
        if not reviewed:
            add("pending_human_review", str(case.get("review", {}).get("reviewer_role", "reviewer_role_required")))
        if reviewed and not case.get("review", {}).get("reviewed_at"):
            add("reviewed_case_missing_timestamp", "")
        if reviewed and case.get("expect_evidence"):
            if not case.get("reference_answers"):
                add("missing_language_reference_answer", "")
            elif case.get("language") == "ar" and not any(_contains_arabic(answer) for answer in case.get("reference_answers", [])):
                add("reference_language_content_mismatch", "Arabic case has no Arabic reference answer")
            elif case.get("language") == "en" and any(_contains_arabic(answer) for answer in case.get("reference_answers", [])):
                add("reference_language_content_mismatch", "English case contains an Arabic reference answer")
            if not case.get("required_claims"):
                add("missing_required_claims", "")
            if not case.get("relevant_items"):
                add("missing_relevance_judgments", "")
        for item in case.get("relevant_items", []):
            source_id = str(item.get("source_id", ""))
            source = next((entry for entry in catalog.values() if entry.source_id == source_id), None)
            if not source:
                add("unknown_source_id", source_id)
            elif not source.source_url.startswith("https://"):
                add("non_https_provenance", source_id)
            grade = item.get("relevance_grade")
            if not isinstance(grade, int) or grade <= 0:
                add("invalid_relevance_grade", str(grade))
    return issues


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("reports/gold_case_review.csv"))
    args = parser.parse_args()
    issues = validate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "case_id", "language", "category", "query", "expected_status", "review_status",
            "relevant_source_ids", "issue", "detail",
        ])
        writer.writeheader()
        writer.writerows(issues)
    print(json.dumps({"dataset_version": gold_dataset()["version"], "issues": len(issues), "review_queue": str(args.output)}, ensure_ascii=False))
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
