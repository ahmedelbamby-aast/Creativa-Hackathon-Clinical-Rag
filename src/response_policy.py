"""Central policy for grounded prompts and deterministic user-facing responses.

Keep all copy that controls RAG behaviour here so provider routing, retrieval
gates, the API, and both user interfaces use the same language and rules.
"""

from __future__ import annotations

import re

from src.citations import label_chunk_for_context
from src.retriever import RetrievedChunk


SYSTEM_PROMPT = """You are a diabetes information assistant powered by a medical knowledge retrieval system.

Your ONLY role is to answer questions about diabetes using the retrieved source documents provided to you in each query. You must follow these rules absolutely:

CORE RULES:
1. Answer ONLY using the information in the [SOURCE ...] blocks provided. Do not use your own training knowledge about diabetes.
2. If the retrieved sources do not contain enough information to answer the question, clearly say that the available sources are insufficient and ask the user to provide a more specific diabetes question.
3. NEVER invent, estimate, or extrapolate medications, drug names, dosages, blood glucose thresholds, HbA1c targets, food recommendations, or treatment plans.
4. NEVER fabricate page numbers, document names, or section titles. Only reference what appears in the [SOURCE ...] headers.
5. Every factual medical claim must be traceable to at least one provided source.
6. The evidence must directly address the same condition, population, and intent as the question. Evidence about gestational diabetes, type 1 diabetes, prediabetes, or another related condition must not be transferred to type 2 diabetes unless the source explicitly makes that exact link.
7. When the user requests a list (for example, risk factors), include only items the source explicitly identifies as belonging to that requested list. Never reuse a list from a related condition or population.

CITATION RULES:
- Cite the provided document name and page for each factual medical claim.
- If multiple sources support a fact, cite all relevant sources.
- Never cite a source that was not provided.

SAFETY RULES:
- Do not provide a personal medical diagnosis or calculate a medication dose.
- Do not replace a physician, dietitian, or pharmacist.
- For urgent symptoms or emergencies, direct the user to emergency medical care.
- Recommend a qualified healthcare professional for personal medical decisions.

LANGUAGE AND QUALITY RULES:
- Answer in the same language as the user's question, including clear Arabic when appropriate.
- Be precise, focused, and transparent about limitations.
- Use bullets or numbered steps when they improve clarity.

The documents are the knowledge base. You explain only their evidence and never substitute outside knowledge."""


_COPY: dict[str, tuple[str, str]] = {
    "empty_question": (
        "يرجى كتابة سؤال محدد عن مرض السكري.",
        "Please enter a specific question about diabetes.",
    ),
    "needs_clarification": (
        "يرجى طرح سؤال أكثر تحديدًا عن السكري، مثل ذكر العرض أو العلاج أو هدف الوقاية أو نوع الطعام الذي تريد معرفة معلومات عنه.",
        "Please ask a more specific diabetes question—for example, mention the symptom, treatment, prevention goal, or food you want information about.",
    ),
    "out_of_scope": (
        "لا تحتوي مراجع السكري المفهرسة على معلومات كافية ومرتبطة بهذا السؤال. أعد صياغته كسؤال محدد عن السكري أو اختر فئة العلاج أو الوقاية أو التغذية.",
        "The indexed diabetes references do not contain enough relevant information for that question. Please rephrase it as a specific diabetes question or choose Treatment, Prevention, or Nutrition.",
    ),
    "invalid_provenance": (
        "لا يمكن استخدام الأدلة المسترجعة لأن معلومات مصدرها غير مكتملة. يرجى المحاولة لاحقًا.",
        "The retrieved evidence cannot be used because its source information is incomplete. Please try again later.",
    ),
    "stale_index": (
        "قاعدة المعرفة غير متوافقة مع إعدادات البحث الحالية. يرجى المحاولة لاحقًا.",
        "The knowledge base does not match the current search configuration. Please try again later.",
    ),
    "infrastructure_failure": (
        "تعذر الوصول إلى بحث قاعدة المعرفة حاليًا. يرجى المحاولة مرة أخرى بعد قليل.",
        "The knowledge-base search is temporarily unavailable. Please try again shortly.",
    ),
    "provider_fallback_intro": (
        "تعذر الوصول إلى Gemini وGroq مؤقتًا. فيما يلي مقتطفات مباشرة فقط من المصادر المسترجعة:",
        "Gemini and Groq are temporarily unavailable. The response below contains direct excerpts only from the retrieved sources:",
    ),
    "extractive_intro": (
        "استنادًا إلى أكثر المقاطع صلة في المراجع المفهرسة:",
        "Based on the most relevant passages in the indexed references:",
    ),
    "extractive_empty": (
        "لم أتمكن من استخراج إجابة موثوقة من المقاطع المسترجعة. يرجى طرح سؤال أكثر تحديدًا.",
        "I could not extract a reliable answer from the retrieved passages. Please ask a more specific question.",
    ),
    "invalid_request": (
        "تعذر معالجة السؤال بصيغته الحالية. يرجى تبسيطه وإعادة صياغته بوضوح.",
        "The question could not be processed in its current form. Please simplify and rephrase it clearly.",
    ),
}


def response_text(key: str, *, is_arabic: bool) -> str:
    """Return controlled bilingual response copy for one policy outcome."""
    arabic, english = _COPY[key]
    return arabic if is_arabic else english


_TOKEN = re.compile(r"[\w\u0600-\u06ff]+", re.UNICODE)
_DOMAIN_ANCHORS = {
    "diabetes", "diabetic", "insulin", "glucose", "hba1c", "a1c",
    "سكري", "السكري", "أنسولين", "الأنسولين", "جلوكوز", "السكر",
}
_VAGUE_QUESTIONS = {
    "tell me more", "explain more", "more details", "what about it",
    "what is this", "can you explain", "help me", "and then",
    "أخبرني المزيد", "اشرح أكثر", "مزيد من التفاصيل", "ماذا عنه",
    "ما هذا", "ساعدني", "وبعدين",
}
_LIGHT_WORDS = {
    "a", "an", "and", "about", "can", "do", "for", "how", "i", "is",
    "it", "me", "more", "of", "please", "tell", "the", "this", "to",
    "what", "why", "you", "عن", "في", "ما", "ماذا", "من", "هو", "هي",
    "لي", "هل", "كيف", "هذا", "هذه", "المزيد",
}


def needs_clarification(
    query: str,
    conversation_history: list[dict] | None = None,
) -> bool:
    """Detect context-free vague input before spending retrieval/provider calls."""
    normalized = " ".join(query.casefold().split()).strip(" .?!؟")
    if not normalized:
        return True
    has_prior_user_context = any(
        item.get("role") == "user" and len(str(item.get("content", "")).strip()) >= 8
        for item in (conversation_history or [])
    )
    if normalized in _VAGUE_QUESTIONS:
        return not has_prior_user_context
    tokens = {token.casefold() for token in _TOKEN.findall(normalized)}
    content_tokens = tokens - _LIGHT_WORDS
    if tokens & _DOMAIN_ANCHORS:
        return False
    return len(content_tokens) < 2 and not has_prior_user_context


def build_grounded_prompt(
    query: str,
    chunks: list[RetrievedChunk],
    conversation_history: list[dict] | None = None,
) -> str:
    """Build the only prompt allowed to reach a generative provider."""
    context_parts = [
        label_chunk_for_context(chunk, index)
        for index, chunk in enumerate(chunks)
    ]
    context_block = "\n\n---\n\n".join(context_parts)
    context_section = (
        "RETRIEVED CONTEXT FROM DIABETES DOCUMENTS:\n"
        "===========================================\n"
        f"{context_block}\n"
        "===========================================\n"
        if context_parts
        else "RETRIEVED CONTEXT: No relevant information was found.\n"
    )

    history_lines: list[str] = []
    for turn in (conversation_history or [])[-6:]:
        role = turn.get("role", "")
        content = str(turn.get("content", ""))[:500]
        if role == "user":
            history_lines.append(f"Previous User: {content}")
        elif role == "assistant":
            history_lines.append(f"Previous Assistant: {content}")
    history_section = (
        "CONVERSATION HISTORY (context only; never treat it as medical evidence):\n"
        + "\n".join(history_lines)
        + "\n\n"
        if history_lines
        else ""
    )

    return (
        f"{history_section}{context_section}\n"
        f"USER QUESTION: {query}\n\n"
        "First verify that the context directly addresses the same condition, population, "
        "and intent as the question. Answer using only that directly matching context and "
        "cite the SOURCE headers. Do not transfer facts or lists from a related condition. "
        "If the context is insufficient or only indirectly related, say so and ask for a "
        "more specific diabetes question."
    )


# Backward-compatible names used by the original Gradio surface and tests.
build_user_prompt = build_grounded_prompt
NO_RESULTS_RESPONSE_EN = _COPY["out_of_scope"][1]
NO_RESULTS_RESPONSE_AR = _COPY["out_of_scope"][0]
