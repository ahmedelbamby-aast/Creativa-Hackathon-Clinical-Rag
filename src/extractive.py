"""Deterministic, card-free answer construction from retrieved evidence."""

from __future__ import annotations

import re

from src.retriever import RetrievedChunk


_BOUNDARY = re.compile(r"(?<=[.!?؟])\s+|\n+")
_TOKEN = re.compile(r"[\w\u0600-\u06ff]+", re.UNICODE)
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "can",
    "for",
    "how",
    "in",
    "is",
    "of",
    "the",
    "to",
    "what",
    "with",
    "كيف",
    "ما",
    "من",
    "في",
    "على",
    "هي",
    "هو",
    "يمكن",
}
_ARABIC_EXPANSIONS = {
    "السكري": {"diabetes", "diabetic"},
    "الوقاية": {"prevention", "prevent", "preventing"},
    "العلاج": {"treatment", "therapy", "management"},
    "الأطعمة": {"food", "foods", "diet", "nutrition"},
    "الغذاء": {"food", "diet", "nutrition"},
    "الحمل": {"pregnancy", "pregnant"},
    "المخاطر": {"risk", "risks", "factor", "factors"},
    "مضاعفات": {"complication", "complications"},
    "النوع": {"type"},
    "الثاني": {"2", "two"},
}


def _query_terms(query: str) -> set[str]:
    terms = {token.lower() for token in _TOKEN.findall(query)} - _STOPWORDS
    for token in tuple(terms):
        terms.update(_ARABIC_EXPANSIONS.get(token, set()))
    return terms


def _clean_sentence(sentence: str) -> str:
    sentence = re.sub(r"\s+", " ", sentence).strip(" \t\r\n-*•|#")
    return sentence


def _trim_excerpt(text: str, limit: int = 420) -> str:
    if len(text) <= limit:
        return text
    clipped = text[:limit].rsplit(" ", 1)[0].rstrip(" ,;:")
    return clipped + "…"


def _best_excerpt(text: str, terms: set[str]) -> str:
    sentences = [_clean_sentence(part) for part in _BOUNDARY.split(text)]
    candidates = [sentence for sentence in sentences if len(sentence) >= 45]
    if not candidates:
        return _trim_excerpt(_clean_sentence(text))

    def score(item: tuple[int, str]) -> tuple[int, int, int]:
        index, sentence = item
        words = {token.lower() for token in _TOKEN.findall(sentence)}
        overlap = len(words & terms)
        # Prefer a compact, query-matching sentence and use earlier text as a tie-breaker.
        return overlap, -abs(len(sentence) - 240), -index

    _, selected = max(enumerate(candidates), key=score)
    return _trim_excerpt(selected)


def build_extractive_answer(
    query: str,
    chunks: list[RetrievedChunk],
    *,
    is_arabic: bool = False,
    max_passages: int = 3,
) -> str:
    """Return concise evidence excerpts without calling a generative model.

    This fallback deliberately avoids synthesis: each bullet is copied from a
    retrieved passage and labelled with the same source index used by the prompt
    and citation metadata.
    """
    terms = _query_terms(query)
    intro = (
        "استنادًا إلى أكثر المقاطع صلة في المراجع المفهرسة:"
        if is_arabic
        else "Based on the most relevant passages in the indexed references:"
    )
    bullets: list[str] = []
    seen: set[str] = set()

    for index, chunk in enumerate(chunks[:max_passages]):
        excerpt = _best_excerpt(chunk.text, terms)
        normalized = excerpt.casefold()
        if not excerpt or normalized in seen:
            continue
        seen.add(normalized)
        page_label = "صفحة" if is_arabic else "Page"
        source_label = "المصدر" if is_arabic else "Source"
        page = f", {page_label} {chunk.page_number}" if chunk.page_number else ""
        bullets.append(f"- {excerpt} **[{source_label} {index + 1}{page}]**")

    if not bullets:
        return (
            "لم أتمكن من استخراج إجابة موثوقة من المقاطع المسترجعة."
            if is_arabic
            else "I could not extract a reliable answer from the retrieved passages."
        )
    return intro + "\n\n" + "\n\n".join(bullets)
