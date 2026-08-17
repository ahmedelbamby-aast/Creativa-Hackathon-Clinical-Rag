"""Safety classifier — detects and handles high-risk medical queries.

Classifies queries into risk levels before generation:
  - EMERGENCY     : Acute symptoms → skip retrieval, return safety response immediately
  - HIGH_RISK     : Dosing / personal treatment requests → warn + retrieve + disclaim
  - DIAGNOSIS     : Personal diagnosis requests → answer informationally + disclaim
  - INFORMATIONAL : Normal RAG flow

The safety layer operates as a pre-generation gate. It never silences the
system for informational questions — it only adds mandatory disclaimers and
redirects when medically necessary.
"""

import logging
import re
from enum import Enum

logger = logging.getLogger(__name__)


class SafetyLevel(str, Enum):
    EMERGENCY = "emergency"
    HIGH_RISK = "high_risk"
    DIAGNOSIS = "diagnosis"
    INFORMATIONAL = "informational"


# ---------------------------------------------------------------------------
# Signal patterns (English + Arabic)
# ---------------------------------------------------------------------------

_EMERGENCY_SIGNALS: list[str] = [
    "emergency", "unconscious", "faint", "fainting", "seizure", "coma",
    "very high blood sugar", "very low blood sugar", "sugar is 400",
    "sugar is 500", "sugar above 400", "glucose is 400",
    "can't breathe", "chest pain", "call ambulance",
    "إسعاف", "طوارئ", "السكر ٤٠٠", "السكر فوق ٤٠٠", "السكر جداً مرتفع",
    "فقدان الوعي", "إغماء",
]

_HIGH_RISK_SIGNALS: list[str] = [
    "what dose", "how much insulin", "how many units", "dosage for me",
    "my insulin dose", "how much metformin", "what medication should i take",
    "which medication should i", "should i take", "prescribe",
    "كم جرعة", "كم وحدة", "جرعتي", "ماذا آخذ", "هل آخذ", "الجرعة المناسبة لي",
]

_DIAGNOSIS_SIGNALS: list[str] = [
    "do i have diabetes", "am i diabetic", "is my blood sugar normal",
    "do i have prediabetes", "do i have type 2",
    "هل أنا مصاب", "هل عندي سكري", "هل أعاني من", "هل نسبة سكري طبيعية",
]


def _score_signals(text: str, signals: list[str]) -> int:
    text_lower = text.lower()
    return sum(1 for s in signals if s in text_lower)


def classify_safety(query: str) -> SafetyLevel:
    """Classify the safety level of a user query.

    Args:
        query: Raw user question.

    Returns:
        SafetyLevel enum value.
    """
    query = query.strip()
    if not query:
        return SafetyLevel.INFORMATIONAL

    if _score_signals(query, _EMERGENCY_SIGNALS) >= 1:
        return SafetyLevel.EMERGENCY

    if _score_signals(query, _HIGH_RISK_SIGNALS) >= 1:
        return SafetyLevel.HIGH_RISK

    if _score_signals(query, _DIAGNOSIS_SIGNALS) >= 1:
        return SafetyLevel.DIAGNOSIS

    return SafetyLevel.INFORMATIONAL


# ---------------------------------------------------------------------------
# Safety response templates
# ---------------------------------------------------------------------------

EMERGENCY_RESPONSE_EN = (
    "🚨 **This sounds like a medical emergency.**\n\n"
    "Please **call emergency services immediately** (in Egypt: 123 for ambulance) "
    "or go to the nearest hospital emergency department.\n\n"
    "Do not rely on this application in an emergency situation."
)

EMERGENCY_RESPONSE_AR = (
    "🚨 **يبدو أن هذه حالة طوارئ طبية.**\n\n"
    "يرجى **الاتصال بالإسعاف فوراً** (في مصر: 123) أو التوجه إلى أقرب طوارئ مستشفى.\n\n"
    "لا تعتمد على هذا التطبيق في حالات الطوارئ."
)

HIGH_RISK_DISCLAIMER_EN = (
    "\n\n---\n"
    "⚠️ **Important**: The information above is general and sourced directly from the "
    "referenced documents. It is **not a personalised medical prescription**. "
    "Medication dosages must always be determined by a qualified physician or "
    "pharmacist based on your individual health condition."
)

HIGH_RISK_DISCLAIMER_AR = (
    "\n\n---\n"
    "⚠️ **ملاحظة مهمة**: المعلومات أعلاه عامة ومستخرجة من المصادر المُستشهد بها. "
    "إنها **ليست وصفة طبية شخصية**. "
    "يجب دائماً تحديد جرعات الأدوية من قِبَل طبيب أو صيدلاني مؤهل "
    "بناءً على حالتك الصحية الفردية."
)

DIAGNOSIS_DISCLAIMER_EN = (
    "\n\n---\n"
    "ℹ️ **Note**: This application provides general educational information about diabetes. "
    "It cannot diagnose any medical condition. Please consult a qualified healthcare "
    "professional for diagnosis and personalised medical advice."
)

DIAGNOSIS_DISCLAIMER_AR = (
    "\n\n---\n"
    "ℹ️ **ملاحظة**: يوفر هذا التطبيق معلومات تعليمية عامة حول مرض السكري. "
    "لا يمكنه تشخيص أي حالة طبية. يرجى استشارة متخصص رعاية صحية مؤهل "
    "للحصول على تشخيص ونصيحة طبية شخصية."
)

GENERAL_DISCLAIMER_EN = (
    "\n\n---\n"
    "*This information is sourced from the referenced documents and is for "
    "educational purposes only. Always consult a healthcare professional for "
    "medical decisions.*"
)

GENERAL_DISCLAIMER_AR = (
    "\n\n---\n"
    "*هذه المعلومات مستخرجة من المصادر المُستشهد بها وهي لأغراض تعليمية فقط. "
    "استشر دائماً أخصائي رعاية صحية لاتخاذ أي قرار طبي.*"
)


def get_disclaimer(safety_level: SafetyLevel, is_arabic: bool = False) -> str:
    """Return the appropriate disclaimer for the safety level and language."""
    if safety_level == SafetyLevel.EMERGENCY:
        return ""  # Emergency uses full response, not a disclaimer

    if safety_level == SafetyLevel.HIGH_RISK:
        return HIGH_RISK_DISCLAIMER_AR if is_arabic else HIGH_RISK_DISCLAIMER_EN

    if safety_level == SafetyLevel.DIAGNOSIS:
        return DIAGNOSIS_DISCLAIMER_AR if is_arabic else DIAGNOSIS_DISCLAIMER_EN

    return GENERAL_DISCLAIMER_AR if is_arabic else GENERAL_DISCLAIMER_EN


def get_emergency_response(is_arabic: bool = False) -> str:
    """Return the emergency safety response (skips retrieval entirely)."""
    return EMERGENCY_RESPONSE_AR if is_arabic else EMERGENCY_RESPONSE_EN
