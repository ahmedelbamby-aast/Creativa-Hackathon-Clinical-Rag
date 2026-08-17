"""Query router — classifies a user question into a diabetes knowledge category.

Uses a two-stage approach:
1. Fast keyword scoring (no API call) — handles clear-cut cases.
2. Falls back to the user-selected category if keyword scoring is inconclusive.

Manual category selection always overrides automatic routing.

Returns one of: "treatment", "prevention", "nutrition", "general", "all"
"""

import logging
import re

from src.config import (
    CATEGORY_TREATMENT,
    CATEGORY_PREVENTION,
    CATEGORY_NUTRITION,
    CATEGORY_GENERAL,
    CATEGORY_ALL,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Keyword dictionaries  (more specific than the classifier ones)
# ---------------------------------------------------------------------------

_TREATMENT_SIGNALS: list[str] = [
    "treat", "medication", "medicine", "drug", "insulin", "metformin",
    "dose", "dosage", "prescription", "pharmacolog", "therapy",
    "hba1c", "glycemic control", "glycaemic control", "blood sugar management",
    "manage diabetes", "managing diabetes", "clinical", "complication",
    "hypoglycemia", "hyperglycemia", "hypoglycaemia", "hyperglycaemia",
    "sulphonylurea", "glp-1", "sglt2", "dpp-4", "thiazolidinedione",
    "علاج", "دواء", "أدوية", "أنسولين", "علاجات", "التحكم في السكر",
]

_PREVENTION_SIGNALS: list[str] = [
    "prevent", "risk", "reduce", "avoid", "screening", "early detection",
    "lifestyle change", "physical activity", "exercise", "weight loss",
    "obesity", "overweight", "sedentary", "public health", "population",
    "incidence", "prevalence", "epidemiol",
    "الوقاية", "تقليل", "تجنب", "النشاط البدني", "الوزن", "الكشف المبكر",
]

_NUTRITION_SIGNALS: list[str] = [
    "food", "eat", "drink", "diet", "meal", "carbohydrate", "carb",
    "calorie", "sugar", "fruit", "vegetable", "grain", "bread", "rice",
    "protein", "fat", "fibre", "fiber", "glycemic index", "glycaemic index",
    "portion", "snack", "breakfast", "lunch", "dinner", "recipe",
    "غذاء", "طعام", "أكل", "وجبة", "كربوهيدرات", "سعرات", "فواكه", "خضروات",
    "نظام غذائي", "رجيم",
]

# Phrases that clearly indicate a general / informational query
_GENERAL_SIGNALS: list[str] = [
    "what is diabetes", "explain diabetes", "what is hba1c", "hba1c",
    "type 1", "type 2", "gestational", "prediabetes", "diagnos",
    "مرض السكري", "السكري", "ما هو السكري", "السكر التراكمي",
]


def _keyword_score(text: str, signals: list[str]) -> int:
    """Count how many signals appear in the lowercased text."""
    text_lower = text.lower()
    return sum(1 for s in signals if s in text_lower)


def route_query(
    query: str,
    user_selected_category: str = CATEGORY_ALL,
) -> str:
    """Determine the best retrieval category for a query.

    Args:
        query: The user's question (English or Arabic).
        user_selected_category: The category the user chose in the UI.
            If not "all", this takes precedence over automatic routing.

    Returns:
        Category string: "treatment", "prevention", "nutrition", "general", or "all".
    """
    # Manual selection always wins (except "all" which means automatic routing)
    if user_selected_category != CATEGORY_ALL:
        logger.debug("Category manually set to: %s", user_selected_category)
        return user_selected_category

    query_clean = query.strip()
    if not query_clean:
        return CATEGORY_ALL

    # Score all categories
    scores = {
        CATEGORY_TREATMENT:  _keyword_score(query_clean, _TREATMENT_SIGNALS),
        CATEGORY_PREVENTION: _keyword_score(query_clean, _PREVENTION_SIGNALS),
        CATEGORY_NUTRITION:  _keyword_score(query_clean, _NUTRITION_SIGNALS),
        CATEGORY_GENERAL:    _keyword_score(query_clean, _GENERAL_SIGNALS),
    }

    best_cat = max(scores, key=scores.__getitem__)
    best_score = scores[best_cat]

    # Sort to find second-best
    sorted_scores = sorted(scores.values(), reverse=True)
    second_score = sorted_scores[1] if len(sorted_scores) > 1 else 0

    # Decisive if best is >= 2 and clearly ahead
    if best_score >= 2 and best_score >= second_score + 1:
        logger.debug(
            "Auto-routed query to '%s' (score=%d vs second=%d)",
            best_cat, best_score, second_score,
        )
        return best_cat

    # If general signals present but no specific category — use "all"
    if scores[CATEGORY_GENERAL] >= 1:
        return CATEGORY_ALL

    # Inconclusive — search all
    logger.debug("Routing inconclusive (scores=%s) — using 'all'", scores)
    return CATEGORY_ALL
