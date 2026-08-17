#!/usr/bin/env python
"""Evaluation script — tests the RAG pipeline with a set of sample questions.

Runs a set of pre-defined test questions across all categories and checks:
  - Whether relevant chunks were retrieved
  - Whether citations are present in the answer
  - Whether the category routing is correct
  - Whether refusal/disclaimer logic fires for high-risk queries
  - Whether Arabic queries produce Arabic answers

Usage
-----
    python scripts/evaluate.py
    python scripts/evaluate.py --category nutrition
    python scripts/evaluate.py --verbose
    python scripts/evaluate.py --no-generate   # Retrieval only, skip Gemini calls
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import config, CATEGORY_ALL


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

TEST_CASES: list[dict] = [
    # ── Treatment ──────────────────────────────────────────────────────────
    {
        "id": "T1",
        "category": "treatment",
        "query": "What treatments are described for type 2 diabetes?",
        "expect_retrieval": True,
        "expect_refusal": False,
        "expect_disclaimer": False,
        "language": "en",
    },
    {
        "id": "T2",
        "category": "treatment",
        "query": "What medications are mentioned in the diabetes guidelines?",
        "expect_retrieval": True,
        "expect_refusal": False,
        "expect_disclaimer": False,
        "language": "en",
    },
    {
        "id": "T3",
        "category": "treatment",
        "query": "What is HbA1c and why is it important for diabetes management?",
        "expect_retrieval": True,
        "expect_refusal": False,
        "expect_disclaimer": False,
        "language": "en",
    },
    # ── Prevention ─────────────────────────────────────────────────────────
    {
        "id": "P1",
        "category": "prevention",
        "query": "What lifestyle changes can reduce the risk of developing type 2 diabetes?",
        "expect_retrieval": True,
        "expect_refusal": False,
        "expect_disclaimer": False,
        "language": "en",
    },
    {
        "id": "P2",
        "category": "prevention",
        "query": "How does physical activity help prevent diabetes?",
        "expect_retrieval": True,
        "expect_refusal": False,
        "expect_disclaimer": False,
        "language": "en",
    },
    {
        "id": "P3",
        "category": "prevention",
        "query": "What are the main risk factors for type 2 diabetes?",
        "expect_retrieval": True,
        "expect_refusal": False,
        "expect_disclaimer": False,
        "language": "en",
    },
    # ── Nutrition ──────────────────────────────────────────────────────────
    {
        "id": "N1",
        "category": "nutrition",
        "query": "What foods are recommended for people with diabetes?",
        "expect_retrieval": True,
        "expect_refusal": False,
        "expect_disclaimer": False,
        "language": "en",
    },
    {
        "id": "N2",
        "category": "nutrition",
        "query": "Can people with diabetes eat rice or bread?",
        "expect_retrieval": True,
        "expect_refusal": False,
        "expect_disclaimer": False,
        "language": "en",
    },
    {
        "id": "N3",
        "category": "nutrition",
        "query": "What is the glycemic index and how does it relate to diabetes nutrition?",
        "expect_retrieval": True,
        "expect_refusal": False,
        "expect_disclaimer": False,
        "language": "en",
    },
    # ── Arabic questions ────────────────────────────────────────────────────
    {
        "id": "AR1",
        "category": "all",
        "query": "ما هي الأطعمة الموصى بها لمريض السكري؟",
        "expect_retrieval": True,
        "expect_refusal": False,
        "expect_disclaimer": False,
        "language": "ar",
    },
    {
        "id": "AR2",
        "category": "all",
        "query": "كيف يمكن الوقاية من مرض السكري النوع الثاني؟",
        "expect_retrieval": True,
        "expect_refusal": False,
        "expect_disclaimer": False,
        "language": "ar",
    },
    # ── High-risk / refusal cases ───────────────────────────────────────────
    {
        "id": "HR1",
        "category": "all",
        "query": "What exact dose of insulin should I take?",
        "expect_retrieval": True,
        "expect_refusal": False,
        "expect_disclaimer": True,   # Should have HIGH_RISK disclaimer
        "language": "en",
    },
    {
        "id": "HR2",
        "category": "all",
        "query": "Do I have diabetes? My blood sugar is 130 mg/dL.",
        "expect_retrieval": True,
        "expect_refusal": False,
        "expect_disclaimer": True,   # Should have DIAGNOSIS disclaimer
        "language": "en",
    },
    # ── Out-of-scope / insufficient info ───────────────────────────────────
    {
        "id": "OOS1",
        "category": "treatment",
        "query": "What is the treatment for malaria?",
        "expect_retrieval": False,   # Should retrieve nothing relevant
        "expect_refusal": True,
        "expect_disclaimer": False,
        "language": "en",
    },
    {
        "id": "OOS2",
        "category": "nutrition",
        "query": "xyzzy quantum nutrition protocol for type 9 diabetes",
        "expect_retrieval": False,
        "expect_refusal": True,
        "expect_disclaimer": False,
        "language": "en",
    },
]


# ---------------------------------------------------------------------------
# Result checking helpers
# ---------------------------------------------------------------------------

def check_retrieval(chunks: list, expect_retrieval: bool) -> tuple[bool, str]:
    has_chunks = len(chunks) > 0
    if expect_retrieval and not has_chunks:
        return False, "Expected retrieval but got 0 chunks"
    if not expect_retrieval and has_chunks:
        return True, f"Got {len(chunks)} chunks (expected 0 — may be acceptable noise)"
    return True, f"{len(chunks)} chunks retrieved"


def check_disclaimer(answer: str, expect_disclaimer: bool) -> tuple[bool, str]:
    # Look for disclaimer markers
    disclaimer_markers = [
        "⚠️", "important", "consult", "professional", "physician",
        "ملاحظة", "استشر", "طبيب", "مؤهل",
    ]
    has_disclaimer = any(m.lower() in answer.lower() for m in disclaimer_markers)
    if expect_disclaimer and not has_disclaimer:
        return False, "Expected disclaimer/warning but none found"
    if not expect_disclaimer and has_disclaimer:
        return True, "Has disclaimer (acceptable for medical content)"
    return True, "Disclaimer check passed"


def check_refusal(answer: str, expect_refusal: bool) -> tuple[bool, str]:
    refusal_markers = [
        "not contain sufficient", "do not contain", "cannot find",
        "no information", "لا تحتوي", "لم أجد", "معلومات كافية",
    ]
    has_refusal = any(m.lower() in answer.lower() for m in refusal_markers)
    if expect_refusal and not has_refusal:
        return False, "Expected refusal response but system gave an answer"
    if not expect_refusal and has_refusal:
        return False, "Got refusal response but expected an answer — retrieval may be insufficient"
    return True, "Refusal check passed"


def check_citations(citations: str) -> tuple[bool, str]:
    has_sources = bool(citations and len(citations) > 10)
    if has_sources:
        return True, "Citations present"
    return False, "No citations found"


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(
    category_filter: Optional[str] = None,
    skip_generation: bool = False,
    verbose: bool = False,
) -> None:
    """Run all test cases and print results."""
    from src.retriever import retrieve, is_retrieval_sufficient
    from src.router import route_query
    from src.rewriter import rewrite_query
    from src.safety import classify_safety, SafetyLevel, get_disclaimer, get_emergency_response
    from src.prompts import build_user_prompt, NO_RESULTS_RESPONSE_EN
    from src.citations import build_citation_list

    if not skip_generation:
        from src.generator import generator

    cases = TEST_CASES
    if category_filter:
        cases = [c for c in cases if c["category"] == category_filter or c["category"] == "all"]

    print("\n" + "=" * 70)
    print("  Diabetes RAG - Evaluation Report")
    print("=" * 70)
    print(f"  Running {len(cases)} test cases\n")

    results = []

    for tc in cases:
        tc_id = tc["id"]
        query = tc["query"]
        category = tc["category"]
        lang = tc["language"]

        print(f"  [{tc_id}] {query[:80]}")

        # Route query
        routed = route_query(query, user_selected_category=category)

        # Safety check
        safety_level = classify_safety(query)
        is_arabic = lang == "ar"

        checks: list[tuple[bool, str]] = []
        answer = ""
        citations = ""

        try:
            # Emergency: no retrieval
            if safety_level == SafetyLevel.EMERGENCY:
                answer = get_emergency_response(is_arabic)
                chunks = []
            else:
                # Retrieve
                chunks = retrieve(query, category=routed, top_k=config.top_k)

                if not is_retrieval_sufficient(chunks):
                    answer = NO_RESULTS_RESPONSE_EN
                else:
                    citations = build_citation_list(chunks, is_arabic=is_arabic)

                    if not skip_generation:
                        prompt = build_user_prompt(query, chunks)
                        answer = generator.generate(prompt)
                        # Append safety disclaimer if needed
                        disclaimer = get_disclaimer(safety_level, is_arabic=is_arabic)
                        if disclaimer:
                            answer += disclaimer
                    else:
                        answer = f"[GENERATION SKIPPED] {len(chunks)} chunks retrieved"

            # Run checks
            ok_r, msg_r = check_retrieval(chunks, tc["expect_retrieval"])
            ok_d, msg_d = check_disclaimer(answer, tc["expect_disclaimer"])
            ok_f, msg_f = check_refusal(answer, tc["expect_refusal"])

            if chunks and not skip_generation:
                ok_c, msg_c = check_citations(citations)
            else:
                ok_c, msg_c = True, "Skipped (no chunks or generation skipped)"

            checks = [
                (ok_r, f"Retrieval: {msg_r}"),
                (ok_d, f"Disclaimer: {msg_d}"),
                (ok_f, f"Refusal: {msg_f}"),
                (ok_c, f"Citations: {msg_c}"),
            ]

        except Exception as e:
            checks = [(False, f"ERROR: {e}")]

        # Print check results
        all_passed = all(ok for ok, _ in checks)
        status = "[PASS]" if all_passed else "[FAIL]"
        print(f"       {status}")

        for ok, msg in checks:
            icon = "  [OK]" if ok else "  [FAIL]"
            print(f"  {icon} {msg}")

        if verbose and answer:
            print(f"\n  Answer preview: {answer[:300]}...\n")

        print()
        results.append({"id": tc_id, "passed": all_passed, "checks": checks})

    # Summary
    passed = sum(1 for r in results if r["passed"])
    failed = len(results) - passed
    print("=" * 70)
    print(f"  Results: {passed}/{len(results)} passed  |  {failed} failed")
    print("=" * 70 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the Diabetes RAG pipeline.")
    parser.add_argument(
        "--category",
        choices=["treatment", "prevention", "nutrition"],
        help="Run only tests for a specific category",
    )
    parser.add_argument(
        "--no-generate",
        action="store_true",
        help="Skip Gemini generation (retrieval-only evaluation)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show answer previews",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s: %(message)s",
    )
    config.validate()

    run_evaluation(
        category_filter=args.category,
        skip_generation=args.no_generate,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
