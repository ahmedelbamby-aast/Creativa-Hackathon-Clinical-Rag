"""Card-free extractive answer behavior."""

from src.extractive import build_extractive_answer
from src.retriever import RetrievedChunk


def _chunk(text: str, *, page: int = 7) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"chunk-{page}",
        text=text,
        score=0.82,
        distance=0.18,
        document_name="guideline.pdf",
        page_number=page,
        section_title="Prevention",
        subsection_title="Risk factors",
        category="prevention",
        content_type="text",
        language="en",
    )


def test_extractive_answer_prefers_query_matching_sentence() -> None:
    answer = build_extractive_answer(
        "What are the risk factors for type 2 diabetes?",
        [
            _chunk(
                "This introductory sentence discusses the report structure and publication process. "
                "Risk factors for type 2 diabetes include excess weight, physical inactivity, and family history."
            )
        ],
    )

    assert answer.startswith("Based on the most relevant passages")
    assert "excess weight" in answer
    assert "[Source 1, Page 7]" in answer
    assert "report structure" not in answer


def test_extractive_answer_expands_arabic_query_terms() -> None:
    answer = build_extractive_answer(
        "كيف يمكن الوقاية من مرض السكري النوع الثاني؟",
        [
            _chunk(
                "Administrative details are provided in the appendix for reference. "
                "Diabetes prevention includes healthy eating, regular physical activity, and weight management.",
                page=12,
            )
        ],
        is_arabic=True,
    )

    assert answer.startswith("استنادًا")
    assert "Diabetes prevention" in answer
    assert "[المصدر 1, صفحة 12]" in answer


def test_extractive_answer_skips_short_duplicate_passages() -> None:
    first = _chunk("Short evidence", page=1)
    duplicate = _chunk("Short evidence", page=2)

    answer = build_extractive_answer("evidence", [first, duplicate])

    assert "reliable answer" in answer
    assert "Short evidence" not in answer


def test_extractive_answer_skips_fragments_and_uses_later_passage() -> None:
    fragment = _chunk("The high risk of early-onset type 2 diabetes and the", page=2)
    complete = _chunk(
        "Regular physical activity, balanced nutrition, and healthy weight management can reduce the risk of type 2 diabetes.",
        page=3,
    )

    answer = build_extractive_answer(
        "How can type 2 diabetes be prevented?", [fragment, complete]
    )

    assert "early-onset" not in answer
    assert "Regular physical activity" in answer
    assert "[Source 2, Page 3]" in answer


def test_extractive_answer_converts_matching_table_row_to_prose() -> None:
    table = _chunk(
        "Subdomain | Indicator | | --- | --- | | Risk factors for diabetes | 29. Physical inactivity prevalence 30. Overweight and obesity prevalence 31. Tobacco use prevalence 32. Hypertension prevalence |.",
        page=4,
    )

    answer = build_extractive_answer("diabetes risk factors", [table])

    assert "Subdomain" not in answer
    assert "Risk factors for diabetes:" in answer
    assert "Tobacco use prevalence" in answer
    assert "|" not in answer
