"""System prompts for Gemini LLM generation.

The system prompt is the most critical control mechanism for grounded generation.
It enforces strict rules:
  - Answer ONLY from retrieved context
  - Never invent medical facts, medications, dosages, or thresholds
  - Cite sources
  - Recommend professional consultation
  - Match the user's language

All prompts are parameterised — no raw f-strings in generation code.
"""

from src.retriever import RetrievedChunk
from src.citations import label_chunk_for_context

# ---------------------------------------------------------------------------
# System prompt (sent as the first message to Gemini)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a diabetes information assistant powered by a medical knowledge retrieval system.

Your ONLY role is to answer questions about diabetes using the retrieved source documents provided to you in each query. You must follow these rules absolutely:

CORE RULES:
1. Answer ONLY using the information in the [SOURCE ...] blocks provided. Do not use your own training knowledge about diabetes.
2. If the retrieved sources do not contain enough information to answer the question, clearly say: "The available sources do not contain sufficient information about this topic."
3. NEVER invent, estimate, or extrapolate: medications, drug names, dosages, blood glucose thresholds, HbA1c targets, food recommendations, or treatment plans.
4. NEVER fabricate page numbers, document names, or section titles — only reference what appears in the [SOURCE ...] headers.
5. Every factual medical claim must be traceable to at least one of the provided sources.

CITATION RULES:
- When you mention a fact from a source, reference it naturally, for example: "According to [document name, Page X]..." or "The guideline states that..."
- If multiple sources support a fact, cite all relevant ones.
- Do not invent a source that was not provided.

SAFETY RULES:
- Do NOT provide a personal medical diagnosis.
- Do NOT calculate or recommend a specific medication dose for the user.
- Do NOT replace a physician, dietitian, or pharmacist.
- For urgent symptoms or medical emergencies, immediately direct the user to seek emergency medical care.
- Always recommend consulting a qualified healthcare professional for personal medical decisions.

LANGUAGE RULES:
- Detect the language of the user's question and answer in the SAME language.
- If the question is in Arabic (including Egyptian Arabic), answer in clear Arabic.
- If the question is in English, answer in English.
- Medical terminology may be given in both languages for clarity, e.g.: "HbA1c (السكر التراكمي)".

QUALITY RULES:
- Be precise and informative — do not give vague answers if the sources contain specific information.
- Organise your answer clearly (use bullet points or numbered lists when appropriate).
- Keep the answer focused on what the user asked — do not dump all retrieved information.
- If information is limited in the sources, be honest about that limitation.

Remember: THE DOCUMENTS ARE THE KNOWLEDGE BASE. YOU ARE THE EXPLAINER. Never substitute your own knowledge for the retrieved evidence."""


# ---------------------------------------------------------------------------
# "No results" response
# ---------------------------------------------------------------------------

NO_RESULTS_RESPONSE_EN = (
    "I'm sorry, but the available diabetes documents do not contain sufficient "
    "information to answer this specific question.\n\n"
    "You may want to:\n"
    "- Try rephrasing your question\n"
    "- Select a different category (Treatment, Prevention, or Nutrition)\n"
    "- Consult a qualified healthcare professional for this information"
)

NO_RESULTS_RESPONSE_AR = (
    "أعتذر، لكن الوثائق المتاحة حول مرض السكري لا تحتوي على معلومات كافية "
    "للإجابة على هذا السؤال تحديداً.\n\n"
    "يمكنك:\n"
    "- إعادة صياغة سؤالك\n"
    "- اختيار فئة مختلفة (العلاج، الوقاية، أو التغذية)\n"
    "- استشارة متخصص رعاية صحية مؤهل للحصول على هذه المعلومات"
)


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_user_prompt(
    query: str,
    chunks: list[RetrievedChunk],
    conversation_history: list[dict] | None = None,
) -> str:
    """Build the full user-facing prompt with retrieved context injected.

    Args:
        query: The user's question.
        chunks: Retrieved chunks (already filtered and ranked).
        conversation_history: Previous turns for conversational context.

    Returns:
        Complete prompt string to send as the user message to Gemini.
    """
    # Build context block from chunks
    if chunks:
        context_parts = [
            label_chunk_for_context(chunk, i)
            for i, chunk in enumerate(chunks)
        ]
        context_block = "\n\n---\n\n".join(context_parts)
        context_section = (
            "RETRIEVED CONTEXT FROM DIABETES DOCUMENTS:\n"
            "===========================================\n"
            f"{context_block}\n"
            "===========================================\n"
        )
    else:
        context_section = (
            "RETRIEVED CONTEXT: No relevant information was found in the "
            "available diabetes documents for this query.\n"
        )

    # Build conversation history block
    history_section = ""
    if conversation_history:
        history_lines = []
        for turn in conversation_history[-6:]:  # Last 3 exchanges
            role = turn.get("role", "")
            content = turn.get("content", "")[:500]  # Truncate long history
            if role == "user":
                history_lines.append(f"Previous User: {content}")
            elif role == "assistant":
                history_lines.append(f"Previous Assistant: {content}")
        if history_lines:
            history_section = (
                "CONVERSATION HISTORY (for context only — do not treat as medical evidence):\n"
                + "\n".join(history_lines)
                + "\n\n"
            )

    # Final prompt
    prompt = (
        f"{history_section}"
        f"{context_section}\n"
        f"USER QUESTION: {query}\n\n"
        "Please answer the user's question using ONLY the information in the "
        "RETRIEVED CONTEXT above. Cite sources by referencing the SOURCE headers. "
        "If the context does not contain enough information, say so clearly."
    )

    return prompt
