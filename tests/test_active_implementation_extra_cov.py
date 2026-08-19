"""Additional coverage tests for active implementation code.

Covers:
1. src/citations.py
2. src/context_builder.py
"""

from __future__ import annotations

import pytest
from src.retriever import RetrievedChunk
from src.citations import (
    build_citation_list,
    build_debug_info,
    label_chunk_for_context,
    normalize_inline_citations,
)
from src.context_builder import build_context


@pytest.fixture
def sample_chunks() -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk_id="chunk_1",
            text="First chunk content for diabetes.",
            score=0.9,
            distance=0.1,
            document_name="IDF_Diabetes_Atlas.pdf",
            page_number=10,
            section_title="Prevention",
            subsection_title="Diet",
            category="prevention",
            content_type="text",
            language="en",
        ),
        RetrievedChunk(
            chunk_id="chunk_2",
            text="Second chunk content about women and pregnancy.",
            score=0.8,
            distance=0.2,
            document_name="WHO_recommendations.pdf",
            page_number=22,
            section_title="Pregnancy Care",
            subsection_title="",
            category="treatment",
            content_type="text",
            language="en",
        ),
    ]


def test_citations_module(sample_chunks):
    """Test all functions in src/citations.py to achieve full coverage."""
    # 1. label_chunk_for_context with full metadata
    label1 = label_chunk_for_context(sample_chunks[0], 0)
    assert "[E1 | IDF_Diabetes_Atlas.pdf | Section: Prevention | Subsection: Diet | Page 10]" in label1
    
    # 2. label_chunk_for_context with minimal metadata (tests missing branch coverage)
    minimal_chunk = RetrievedChunk(
        chunk_id="chunk_min",
        text="Minimal chunk.",
        score=0.9,
        distance=0.1,
        document_name="Minimal.pdf",
        page_number=0,  # missing page
        section_title="",  # missing section
        subsection_title="",  # missing subsection
        category="general",
        content_type="text",
        language="en",
    )
    label2 = label_chunk_for_context(minimal_chunk, 1)
    assert "[E2 | Minimal.pdf]" in label2
    
    # 3. build_citation_list with duplicates (tests key in seen branch)
    citations_dup = build_citation_list([sample_chunks[0], sample_chunks[0]], is_arabic=False)
    # The count should only show one citation bullet
    assert citations_dup.count("•") == 1

    # 4. build_citation_list with no section/page
    citations_min = build_citation_list([minimal_chunk], is_arabic=False)
    assert "Section" not in citations_min
    assert "Page" not in citations_min

    # 5. build_citation_list (English vs Arabic)
    assert "Sources:" in build_citation_list(sample_chunks, is_arabic=False)
    assert "المصادر:" in build_citation_list(sample_chunks, is_arabic=True)
    assert normalize_inline_citations("Fact [ E1 | صفحة 12 ] and 【E2】.", 2) == "Fact [E1] and [E2]."
    assert normalize_inline_citations("Unknown [E9]", 2) == "Unknown [E9]"
    assert build_citation_list([]) == ""
    
    # 6. build_debug_info
    debug_str = build_debug_info("query", "rewritten", "treatment", sample_chunks)
    assert "── DEBUG INFO ──" in debug_str


def test_context_builder_module(sample_chunks):
    """Test build_context function in src/context_builder.py to achieve full coverage."""
    # 1. Empty list
    assert build_context([]) == ""
    
    # 2. Normal build
    ctx = build_context(sample_chunks)
    assert "[SOURCE 1] | Document: IDF_Diabetes_Atlas.pdf" in ctx
    assert "First chunk content for diabetes." in ctx
    
    # 3. Context builder with minimal metadata (no page/section)
    minimal_chunk = RetrievedChunk(
        chunk_id="chunk_min",
        text="Minimal chunk.",
        score=0.9,
        distance=0.1,
        document_name="Minimal.pdf",
        page_number=0,
        section_title="",
        subsection_title="",
        category="general",
        content_type="text",
        language="en",
    )
    ctx_min = build_context([minimal_chunk])
    assert "Section:" not in ctx_min
    assert "Page:" not in ctx_min

    # 4. Truncation test (requires at least one chunk to already be in context_parts)
    long_chunk = RetrievedChunk(
        chunk_id="chunk_long",
        text="A" * 15_000,
        score=0.85,
        distance=0.05,
        document_name="Long.pdf",
        page_number=1,
        section_title="Long Section",
        subsection_title="",
        category="general",
        content_type="text",
        language="en",
    )
    ctx_truncated = build_context([sample_chunks[0], long_chunk])
    assert "[truncated]" in ctx_truncated

    # 5. Truncation test with remaining <= 200 (tests remaining <= 200 check)
    exact_chunk = RetrievedChunk(
        chunk_id="chunk_exact",
        text="A" * 11_800,  # leaves less than 200 characters under the 12,000 limit
        score=0.95,
        distance=0.05,
        document_name="Exact.pdf",
        page_number=1,
        section_title="Exact Section",
        subsection_title="",
        category="general",
        content_type="text",
        language="en",
    )
    ctx_exact = build_context([exact_chunk, long_chunk])
    assert "Exact.pdf" in ctx_exact
    assert "Long.pdf" not in ctx_exact


def test_prompts_module(sample_chunks):
    """Test all functions in src/prompts.py to achieve full coverage."""
    from src.prompts import build_user_prompt, SYSTEM_PROMPT, NO_RESULTS_RESPONSE_EN, NO_RESULTS_RESPONSE_AR
    
    # Check variables exist
    assert len(SYSTEM_PROMPT) > 0
    assert len(NO_RESULTS_RESPONSE_EN) > 0
    assert len(NO_RESULTS_RESPONSE_AR) > 0
    
    # 1. build_user_prompt with chunks and history
    history = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"}
    ]
    prompt_with_history = build_user_prompt("What is diabetes?", sample_chunks, history)
    assert "Previous User: Hello" in prompt_with_history
    assert "Previous Assistant: Hi" in prompt_with_history
    assert "RETRIEVED CONTEXT FROM DIABETES DOCUMENTS:" in prompt_with_history
    assert "USER QUESTION: What is diabetes?" in prompt_with_history

    # 2. build_user_prompt without chunks and history
    prompt_no_context = build_user_prompt("What is diabetes?", [])
    assert "No relevant information was found" in prompt_no_context


