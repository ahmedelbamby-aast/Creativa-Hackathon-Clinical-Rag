"""Central grounded-response policy behavior."""

from src.response_policy import (
    SYSTEM_PROMPT,
    build_grounded_prompt,
    is_out_of_domain,
    needs_clarification,
    response_text,
)
from src.retriever import RetrievedChunk


def _chunk() -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="chunk-1",
        text="Diabetes prevention guidance from the indexed source.",
        score=0.9,
        distance=0.1,
        document_name="guide.pdf",
        page_number=7,
        section_title="Prevention",
        subsection_title="",
        category="prevention",
        content_type="text",
        language="en",
        source_id="guide",
        source_url="https://example.test/guide",
    )


def test_vague_input_requires_context_but_valid_follow_up_uses_history() -> None:
    assert needs_clarification("Tell me more") is True
    assert needs_clarification(
        "Tell me more",
        [{"role": "user", "content": "What are diabetes risk factors?"}],
    ) is False
    assert needs_clarification("What are diabetes risk factors?") is False


def test_obviously_non_diabetes_question_is_rejected_but_contextual_follow_up_is_allowed() -> None:
    assert is_out_of_domain("Who won the football World Cup in 2022?") is True
    assert is_out_of_domain("What is the treatment for malaria?") is True
    assert is_out_of_domain("How many adults are living with diabetes?") is False
    assert is_out_of_domain("كم عدد البالغين المصابين بالسكري؟") is False
    assert is_out_of_domain("ما هدف التحكم في سكر الدم بشكل فردي؟") is False
    assert is_out_of_domain(
        "What about the 2050 figure?",
        [{"role": "user", "content": "How many adults are living with diabetes?"}],
    ) is False


def test_grounded_prompt_contains_only_supplied_evidence_and_history_is_not_evidence() -> None:
    prompt = build_grounded_prompt(
        "How can diabetes be prevented?",
        [_chunk()],
        [{"role": "assistant", "content": "Prior conversational context"}],
    )

    assert "Diabetes prevention guidance" in prompt
    assert "Prior conversational context" in prompt
    assert "never treat it as medical evidence" in prompt
    assert "Answer using only that directly matching context" in prompt
    assert "Do not transfer facts or lists from a related condition" in prompt
    assert "Do not use your own training knowledge" in SYSTEM_PROMPT
    assert "Never reuse a list from a related condition" in SYSTEM_PROMPT
    assert "transparent arithmetic" in SYSTEM_PROMPT
    assert "Do not add related statistics" in SYSTEM_PROMPT
    assert "prefer one direct sentence" in SYSTEM_PROMPT
    assert "display equations between \\[ and \\]" in SYSTEM_PROMPT
    assert "cite evidence inline as [E1]" in prompt
    assert "format it as valid LaTeX" in prompt


def test_controlled_copy_is_bilingual_and_actionable() -> None:
    english = response_text("out_of_scope", is_arabic=False)
    arabic = response_text("out_of_scope", is_arabic=True)

    assert "specific diabetes question" in english
    assert "السكري" in arabic
