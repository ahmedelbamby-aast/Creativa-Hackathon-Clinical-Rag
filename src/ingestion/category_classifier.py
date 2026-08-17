"""Category classifier — assigns treatment / prevention / nutrition / general to chunks.

Two-pass strategy:
1. Document-level label from DOCUMENT_CATEGORY_MAP (filename keyword match).
2. Section-level override using section/subsection title keyword scoring.
3. Content-level fallback using keyword scoring over chunk text.

The section-level and content-level passes allow a "treatment" document to
produce "nutrition" chunks when discussing dietary management, etc.
"""

import re
import logging
from typing import Optional

from src.config import (
    CATEGORY_TREATMENT,
    CATEGORY_PREVENTION,
    CATEGORY_NUTRITION,
    CATEGORY_GENERAL,
    DOCUMENT_CATEGORY_MAP,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Keyword dictionaries for section / content classification
# ---------------------------------------------------------------------------

_TREATMENT_KEYWORDS: list[str] = [
    "treatment", "therapy", "medication", "medicine", "drug", "insulin",
    "metformin", "sulphonylurea", "glp-1", "sglt2", "hba1c", "glycemic control",
    "blood glucose management", "clinical management", "pharmacological",
    "prescription", "dosing", "dose", "glycaemia", "hypoglycaemia",
    "hyperglycaemia", "complication", "مضاعفات", "علاج", "دواء", "أدوية",
    "أنسولين", "نسبة السكر", "التحكم في السكر",
]

_PREVENTION_KEYWORDS: list[str] = [
    "prevention", "prevent", "risk reduction", "risk factor", "lifestyle",
    "physical activity", "exercise", "weight management", "obesity", "overweight",
    "screening", "early detection", "public health", "epidemiology", "incidence",
    "prevalence", "awareness", "education", "programme", "intervention",
    "الوقاية", "منع", "عوامل الخطر", "النشاط البدني", "الوزن", "البدانة",
    "الكشف المبكر", "الصحة العامة",
]

_NUTRITION_KEYWORDS: list[str] = [
    "nutrition", "diet", "dietary", "food", "meal", "carbohydrate", "carb",
    "glycemic index", "glycaemic index", "calorie", "fibre", "fiber",
    "protein", "fat", "saturated fat", "vegetable", "fruit", "whole grain",
    "sugar", "sweetener", "portion", "eating", "drink", "beverage",
    "التغذية", "نظام غذائي", "غذاء", "طعام", "وجبة", "كربوهيدرات",
    "سعرات", "بروتين", "دهون", "خضروات", "فواكه",
]


def _keyword_score(text: str, keywords: list[str]) -> int:
    """Count how many keywords appear in lowercased *text*."""
    text_lower = text.lower()
    return sum(1 for kw in keywords if kw in text_lower)


# ---------------------------------------------------------------------------
# Document-level label
# ---------------------------------------------------------------------------

def get_document_category(document_name: str) -> str:
    """Return the document-level category for a given filename.

    Matches partial substrings (case-insensitive) against DOCUMENT_CATEGORY_MAP.
    Returns CATEGORY_GENERAL if no match is found.
    """
    name_lower = document_name.lower()
    for key, category in DOCUMENT_CATEGORY_MAP.items():
        if key.lower() in name_lower:
            return category
    return CATEGORY_GENERAL


# ---------------------------------------------------------------------------
# Chunk-level classification
# ---------------------------------------------------------------------------

def classify_chunk(
    document_name: str,
    section_title: str,
    subsection_title: str,
    content: str,
    content_type: str,
) -> str:
    """Determine the best category for a single chunk.

    Algorithm:
    1. Get document-level label as a prior.
    2. Score section + subsection title against keyword lists.
    3. If section score is decisive (>= 2 lead), use it.
    4. Otherwise score chunk content.
    5. If content score is decisive, use it.
    6. Fall back to document-level label.

    Returns:
        One of: "treatment", "prevention", "nutrition", "general"
    """
    doc_category = get_document_category(document_name)

    # Combine section titles for scoring
    section_text = f"{section_title} {subsection_title}"

    # Score each category
    section_scores = {
        CATEGORY_TREATMENT:  _keyword_score(section_text, _TREATMENT_KEYWORDS),
        CATEGORY_PREVENTION: _keyword_score(section_text, _PREVENTION_KEYWORDS),
        CATEGORY_NUTRITION:  _keyword_score(section_text, _NUTRITION_KEYWORDS),
    }

    best_section_cat = max(section_scores, key=section_scores.__getitem__)
    best_section_score = section_scores[best_section_cat]
    second_best_section = sorted(section_scores.values(), reverse=True)[1]

    # Decisive section match: clearly better than others
    if best_section_score >= 2 and best_section_score >= second_best_section + 2:
        return best_section_cat

    # Score content text
    content_scores = {
        CATEGORY_TREATMENT:  _keyword_score(content, _TREATMENT_KEYWORDS),
        CATEGORY_PREVENTION: _keyword_score(content, _PREVENTION_KEYWORDS),
        CATEGORY_NUTRITION:  _keyword_score(content, _NUTRITION_KEYWORDS),
    }

    best_content_cat = max(content_scores, key=content_scores.__getitem__)
    best_content_score = content_scores[best_content_cat]
    second_best_content = sorted(content_scores.values(), reverse=True)[1]

    if best_content_score >= 3 and best_content_score >= second_best_content + 2:
        return best_content_cat

    # Fall back to document-level label
    # If general, try to assign by best available content score
    if doc_category == CATEGORY_GENERAL:
        if best_content_score >= 2:
            return best_content_cat
        return CATEGORY_GENERAL

    return doc_category
