"""Diabetes RAG Assistant — Gradio Web UI.

Provides a clean, bilingual (English + Arabic) chat interface with:
  - Category selector (All / Treatment / Prevention / Nutrition)
  - Chat history with source citations
  - Debug panel (enabled via DEBUG=true in the selected environment)
  - Clear conversation button

Run with:
    python app.py
"""

import html
import logging
import re
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# Add project root to Python path when running directly
sys.path.insert(0, str(Path(__file__).resolve().parent))

import gradio as gr

from src.config import config, CATEGORY_ALL, CATEGORY_TREATMENT, CATEGORY_PREVENTION, CATEGORY_NUTRITION
from src.memory import ConversationMemory
from src.safety import classify_safety, get_disclaimer
from src.prompts import build_user_prompt
from src.citations import build_citation_list, build_debug_info, ensure_inline_citations
from src.generator import generator
from src.extractive import build_extractive_answer
from src.vector_store import vector_store
from src.observability import (
    RequestTrace,
    diagnostics_markdown,
    record_trace,
    run_retrieval_benchmark,
)
from src.evidence_service import envelope_chunks, render_evidence, stage_evidence
from src.gemini_errors import classify_gemini_error, gemini_user_message
from src.retrieval_contracts import RetrievalEnvelope
from src.response_policy import response_text
from src.sample_questions import flat_sample_questions

logging.basicConfig(
    level=logging.DEBUG if config.debug else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: detect Arabic
# ---------------------------------------------------------------------------
def _is_arabic(text: str) -> bool:
    return bool(re.search(r"[\u0600-\u06FF]", text))


# ---------------------------------------------------------------------------
# Core RAG pipeline
# ---------------------------------------------------------------------------

def rag_pipeline(
    query: str,
    category: str,
    memory: ConversationMemory,
) -> tuple[str, str, str]:
    """Run the full RAG pipeline for a single user query.

    Returns:
        (answer_text, citations_text, debug_text)
    """
    query = query.strip()
    if not query:
        return "Please enter a question.", "", ""

    trace = RequestTrace(query=query[:200], requested_category=category)
    history = memory.get_history()
    with trace.stage("retrieval"):
        envelope = stage_evidence(query, category, history)
    trace.capture_retrieval(envelope)
    if not envelope.is_ready:
        response = envelope.user_message
        memory.add_user(query, category=category)
        memory.add_assistant(response)
        trace.finish(envelope.status)
        record_trace(trace)
        return response, "", render_evidence(envelope)
    answer, citations, debug = generate_from_evidence(envelope, memory, trace)
    memory.add_user(query, category=category)
    memory.add_assistant(answer)
    return answer, citations, debug


def generate_from_evidence(
    envelope: RetrievalEnvelope,
    memory: ConversationMemory,
    trace: RequestTrace | None = None,
) -> tuple[str, str, str]:
    """Generate strictly from the previously displayed, immutable evidence envelope."""
    if not envelope.is_ready:
        return envelope.user_message, "", render_evidence(envelope)
    query = envelope.original_query
    is_ar = _is_arabic(query)
    chunks = envelope_chunks(envelope)
    citations = build_citation_list(chunks, is_arabic=is_ar)
    safety_level = classify_safety(query)

    # Step 7: Generate answer
    try:
        stage = trace.stage("generation") if trace else None
        if stage:
            stage.__enter__()
        try:
            if config.generation_provider == "extractive":
                answer = build_extractive_answer(query, chunks, is_arabic=is_ar)
                generator.mark_extractive_fallback()
            else:
                prompt = build_user_prompt(query, chunks, conversation_history=memory.get_history())
                answer = generator.generate(prompt)
        finally:
            if stage:
                stage.__exit__(None, None, None)
    except Exception as e:
        error_info = classify_gemini_error(e)
        logger.warning(
            "Generation provider route unavailable; using controlled fallback: code=%s",
            error_info.code,
        )
        if error_info.code in {"safety_blocked", "invalid_request"}:
            answer = (
                response_text("invalid_request", is_arabic=is_ar)
                if error_info.code == "invalid_request"
                else gemini_user_message(e, is_arabic=is_ar, scope="generation")
            )
        else:
            answer = build_extractive_answer(
                query,
                chunks,
                is_arabic=is_ar,
                provider_fallback=True,
            )
            generator.mark_extractive_fallback()
        if trace:
            trace.error = f"generation:{error_info.code}"

    answer = ensure_inline_citations(answer, len(chunks), is_arabic=is_ar)

    # Step 8: Append safety disclaimer
    disclaimer = get_disclaimer(safety_level, is_arabic=is_ar)
    if disclaimer:
        answer += disclaimer

    debug = render_evidence(envelope)
    if config.debug:
        debug += "\n\n" + build_debug_info(query, envelope.rewritten_query, envelope.routed_category, chunks)
    if trace:
        trace.capture_generation(
            answer, generator.active_provider, generator.active_model,
            generator.last_usage, generator.last_attempts, citations,
        )
        trace.finish("generation_error" if trace.error else "ok", trace.error)
        record_trace(trace)
    return answer, citations, debug


# ---------------------------------------------------------------------------
# Gradio chat function (called by the UI)
# ---------------------------------------------------------------------------

# Session memory — one per Gradio session (via gr.State)
def create_memory() -> ConversationMemory:
    return ConversationMemory()


def chat(
    message: str,
    chat_history: list[list[str]],
    category: str,
    memory: ConversationMemory,
) -> tuple[list[list[str]], str, str, str, ConversationMemory]:
    """Gradio chat callback.

    Args:
        message: User's new message.
        chat_history: Current chat history [[user, assistant], ...].
        category: Selected knowledge category.
        memory: Session conversation memory.

    Returns:
        (updated_chat_history, citations_md, debug_md, empty_input, memory)
    """
    if not message.strip():
        return chat_history, "", "", "", memory

    answer, citations, debug = rag_pipeline(message, category, memory)

    chat_history = chat_history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": answer},
    ]
    return chat_history, citations, debug, "", memory


def retrieve_for_ui(
    message: str,
    chat_history: list[dict],
    category: str,
    memory: ConversationMemory,
) -> tuple[list[dict], str, RetrievalEnvelope | None, str]:
    """First UI event: render evidence before the chained generation event starts."""
    if not message.strip():
        return chat_history, "", None, ""
    envelope = stage_evidence(message, category, memory.get_history())
    return (
        chat_history + [{"role": "user", "content": message}],
        render_evidence(envelope),
        envelope,
        "",
    )


def generate_for_ui(
    chat_history: list[dict],
    envelope: RetrievalEnvelope | None,
    memory: ConversationMemory,
) -> tuple[list[dict], str, str, ConversationMemory, str]:
    """Second UI event: answer using only the envelope emitted by retrieve_for_ui."""
    if envelope is None:
        return chat_history, "", "", memory, generation_provider_status()
    answer, citations, debug = generate_from_evidence(envelope, memory)
    memory.add_user(envelope.original_query, category=envelope.requested_category)
    memory.add_assistant(answer)
    return (
        chat_history + [{"role": "assistant", "content": answer}],
        citations,
        debug,
        memory,
        generation_provider_status(),
    )


def clear_chat(memory: ConversationMemory) -> tuple:
    memory.clear()
    return [], "*Sources will appear here after your first question.*", "", "", memory


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

CATEGORY_CHOICES = {
    "All categories": CATEGORY_ALL,
    "Treatment": CATEGORY_TREATMENT,
    "Prevention": CATEGORY_PREVENTION,
    "Nutrition": CATEGORY_NUTRITION,
}

CATEGORY_DISPLAY_TO_VALUE = CATEGORY_CHOICES
CATEGORY_VALUE_TO_DISPLAY = {v: k for k, v in CATEGORY_CHOICES.items()}


def example_loader(question: str):
    """Return a zero-argument callback for a predefined question button."""
    def load_question() -> str:
        return question

    return load_question


def knowledge_status_html() -> str:
    """Render a concise, patient-safe knowledge-base status."""
    try:
        indexed_chunks = sum(vector_store.collection_stats().values())
        state = "ready" if indexed_chunks else "empty"
        label = f"Knowledge base ready · {indexed_chunks} indexed passages"
    except Exception:
        state = "offline"
        label = "Knowledge base unavailable"
    provider_label = html.escape(config.configured_generation_provider_label)
    model_label = html.escape(generator.active_model)
    return (
        f'<div id="knowledge-status" class="status-{state}" role="status">'
        f'<span class="status-dot" aria-hidden="true"></span>{label}'
        f'<span class="status-meta">Local multilingual retrieval · Answer provider: {provider_label} · {model_label}</span>'
        "</div>"
    )


def generation_provider_status() -> str:
    """Return the provider and model that produced (or will produce) the answer."""
    if config.generation_provider == "extractive":
        return "**Answer provider:** Evidence excerpts (no LLM call)"
    return (
        f"**Answer provider:** {config.configured_generation_provider_label} "
        f"· `{generator.active_model}`"
    )

CUSTOM_CSS = """
/* ── Global ─────────────────────────────────────────────────────────── */
:root {
    --surface-page: #0f172a;
    --surface-card: #17233a;
    --surface-raised: #1e3a5f;
    --text-primary: #f8fafc;
    --text-secondary: #bfdbfe;
    --accent: #42a5f5;
    --focus: #7dd3fc;
    --border: rgba(147, 197, 253, 0.32);
    --radius-sm: 10px;
    --radius-lg: 16px;
}

* { font-family: 'Inter', sans-serif !important; }

body, .gradio-container {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f2744 100%) !important;
    min-height: 100dvh;
}

.gradio-container {
    max-width: 1120px !important;
    margin-inline: auto !important;
    padding: 24px !important;
    overflow-x: hidden;
}

button, textarea, input, [role="combobox"] {
    min-height: 44px !important;
}

button:focus-visible, textarea:focus-visible, input:focus-visible,
[role="combobox"]:focus-visible, summary:focus-visible {
    outline: 3px solid var(--focus) !important;
    outline-offset: 3px !important;
}

/* ── Header ─────────────────────────────────────────────────────────── */
#header-box {
    background: linear-gradient(135deg, #1e3a5f 0%, #0d47a1 50%, #1565c0 100%);
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 20px;
    border: 1px solid rgba(100, 181, 246, 0.3);
    box-shadow: 0 8px 32px rgba(0,0,0,0.4), 0 0 80px rgba(21,101,192,0.2);
}

#header-title {
    font-size: 2.2rem !important;
    font-weight: 700 !important;
    color: #e3f2fd !important;
    letter-spacing: -0.5px;
    margin: 0 !important;
}

#header-title-row {
    display: flex;
    align-items: center;
    gap: 14px;
}

#header-mark {
    width: 38px;
    height: 38px;
    flex: 0 0 auto;
    color: #7dd3fc;
}

#header-subtitle {
    font-size: 1rem !important;
    color: var(--text-secondary) !important;
    margin-top: 8px !important;
}

#knowledge-status {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 12px 16px;
    margin: -4px 0 20px;
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    background: rgba(23, 35, 58, 0.86);
    color: var(--text-primary);
    font-size: 0.9rem;
}

.status-dot {
    width: 10px;
    height: 10px;
    flex: 0 0 auto;
    border-radius: 50%;
    background: #4ade80;
    box-shadow: 0 0 0 4px rgba(74, 222, 128, 0.14);
}
.status-empty .status-dot { background: #fbbf24; box-shadow: 0 0 0 4px rgba(251,191,36,.14); }
.status-offline .status-dot { background: #f87171; box-shadow: 0 0 0 4px rgba(248,113,113,.14); }
.status-meta { margin-left: auto; color: var(--text-secondary); }

/* ── Category Selector ──────────────────────────────────────────────── */
#category-label label {
    color: #90caf9 !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}

#category-selector .wrap {
    background: rgba(30, 58, 95, 0.6) !important;
    border: 1px solid rgba(100, 181, 246, 0.3) !important;
    border-radius: 10px !important;
    color: #e3f2fd !important;
}

#category-help, #category-help p {
    color: var(--text-secondary) !important;
    font-size: 0.92rem !important;
    line-height: 1.55 !important;
    margin-top: -4px !important;
}

/* ── Chat Interface ──────────────────────────────────────────────────── */
#chatbot {
    background: rgba(15, 23, 42, 0.8) !important;
    border: 1px solid rgba(100, 181, 246, 0.2) !important;
    border-radius: 16px !important;
    box-shadow: 0 4px 24px rgba(0,0,0,0.3) !important;
    min-height: 340px !important;
}

#chatbot p {
    color: var(--text-primary) !important;
}

#chatbot .message.user {
    background: linear-gradient(135deg, #1565c0, #0d47a1) !important;
    border-radius: 16px 16px 4px 16px !important;
    color: #e3f2fd !important;
    border: none !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3) !important;
}

#chatbot .message.bot {
    background: rgba(30, 58, 95, 0.7) !important;
    border-radius: 16px 16px 16px 4px !important;
    color: #e3f2fd !important;
    border: 1px solid rgba(100, 181, 246, 0.15) !important;
}

/* ── Input Box ──────────────────────────────────────────────────────── */
#query-input textarea {
    background: rgba(15, 23, 42, 0.9) !important;
    border: 1.5px solid rgba(100, 181, 246, 0.4) !important;
    border-radius: 12px !important;
    color: #e3f2fd !important;
    font-size: 1rem !important;
    line-height: 1.55 !important;
    transition: border-color 0.2s ease;
}

#query-input textarea:focus {
    border-color: #42a5f5 !important;
    box-shadow: 0 0 0 3px rgba(66, 165, 245, 0.15) !important;
}

#query-input textarea::placeholder {
    color: rgba(191, 219, 254, 0.72) !important;
}

#query-input label, #category-selector label {
    color: var(--text-secondary) !important;
    font-weight: 600 !important;
}

/* ── Buttons ─────────────────────────────────────────────────────────── */
#send-btn {
    background: linear-gradient(135deg, #1565c0, #1976d2) !important;
    border: none !important;
    border-radius: 10px !important;
    color: white !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 12px rgba(21, 101, 192, 0.4) !important;
}

#send-btn:hover {
    background: linear-gradient(135deg, #1976d2, #1e88e5) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 16px rgba(21, 101, 192, 0.5) !important;
}

#clear-btn {
    background: rgba(30, 58, 95, 0.5) !important;
    border: 1px solid rgba(100, 181, 246, 0.3) !important;
    border-radius: 10px !important;
    color: #90caf9 !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
}

#clear-btn:hover {
    background: rgba(30, 58, 95, 0.8) !important;
    border-color: #42a5f5 !important;
}

/* ── Citations Panel ─────────────────────────────────────────────────── */
#citations-box {
    background: rgba(15, 23, 42, 0.7) !important;
    border: 1px solid rgba(100, 181, 246, 0.2) !important;
    border-radius: 12px !important;
    padding: 16px !important;
    color: #90caf9 !important;
    font-size: 0.88rem !important;
    line-height: 1.6;
    min-height: 60px;
}

/* ── Debug Panel ─────────────────────────────────────────────────────── */
#debug-box {
    background: rgba(10, 15, 28, 0.9) !important;
    border: 1px solid rgba(66, 165, 245, 0.2) !important;
    border-radius: 12px !important;
    color: #64b5f6 !important;
    font-family: 'Courier New', monospace !important;
    font-size: 0.78rem !important;
}

/* ── Section Labels ──────────────────────────────────────────────────── */
.section-label {
    color: #64b5f6 !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
    margin-bottom: 6px !important;
}

#citations-box, #citations-box p, #citations-box em {
    color: var(--text-secondary) !important;
}

@media (max-width: 768px) {
    .gradio-container { padding: 12px !important; }
    #header-box { padding: 20px !important; border-radius: 12px; }
    #header-title { font-size: 1.65rem !important; line-height: 1.2; }
    #header-subtitle { font-size: 0.95rem !important; line-height: 1.5; }
    #header-mark { width: 32px; height: 32px; }
    #knowledge-status { align-items: flex-start; flex-wrap: wrap; }
    .status-meta { width: 100%; margin-left: 20px; }
    #chatbot { min-height: 300px !important; }
    #send-btn, #clear-btn { width: 100% !important; }
    .section-label { font-size: 0.82rem !important; }
}

@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        scroll-behavior: auto !important;
        transition-duration: 0.01ms !important;
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
    }
    #send-btn:hover { transform: none !important; }
}

/* ── Scrollbars ──────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: rgba(15, 23, 42, 0.5); border-radius: 3px; }
::-webkit-scrollbar-thumb { background: rgba(100, 181, 246, 0.4); border-radius: 3px; }
"""


GRADIO_THEME = gr.themes.Base(
    primary_hue="blue",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "sans-serif"],
)


def build_ui() -> gr.Blocks:
    """Construct and return the Gradio UI."""

    with gr.Blocks(title="Diabetes RAG Assistant") as demo:

        # Session state
        memory_state = gr.State(value=create_memory)
        evidence_state = gr.State(value=None)

        # ── Header ─────────────────────────────────────────────────────
        with gr.Column(elem_id="header-box"):
            gr.HTML(
                """
                <div>
                  <div id="header-title-row">
                    <svg id="header-mark" viewBox="0 0 24 24" fill="none" aria-hidden="true">
                      <path d="M6 3v5a6 6 0 0 0 12 0V3M6 5H4m14 0h2M12 14v2a4 4 0 0 0 4 4h1"
                            stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>
                      <circle cx="19" cy="20" r="2" stroke="currentColor" stroke-width="1.8"/>
                    </svg>
                    <div id="header-title">Diabetes RAG Assistant</div>
                  </div>
                  <div id="header-subtitle">
                    Grounded answers from verified diabetes guidelines · English &amp; العربية
                  </div>
                </div>
                """
            )

        knowledge_status_output = gr.HTML(
            '<div id="knowledge-status" class="status-offline" role="status">'
            '<span class="status-dot" aria-hidden="true"></span>Checking knowledge base…'
            '</div>'
        )
        provider_output = gr.Markdown(value=generation_provider_status(), elem_id="provider-status")
        demo.load(
            fn=knowledge_status_html,
            outputs=knowledge_status_output,
            api_name="knowledge_status",
            queue=False,
        )

        # ── Category Selector ───────────────────────────────────────────
        with gr.Column():
            category_input = gr.Dropdown(
                choices=list(CATEGORY_CHOICES.keys()),
                value="All categories",
                label="Knowledge Category",
                elem_id="category-selector",
                interactive=True,
            )

            gr.Markdown(
                "Select a category to focus retrieval, or use **All Categories** for cross-domain questions.",
                elem_id="category-help",
            )

        gr.HTML('<div style="height:4px"></div>')

        # ── Chat Interface ──────────────────────────────────────────────
        chatbot = gr.Chatbot(
            label="",
            elem_id="chatbot",
            height=380,
            show_label=False,
            render_markdown=True,
            placeholder="Ask a question to receive a grounded answer with document and page citations.",
        )

        with gr.Column():
            query_input = gr.Textbox(
                placeholder="Ask about diabetes treatment, prevention, nutrition... (English or Arabic)",
                label="Your question",
                elem_id="query-input",
                lines=2,
                max_lines=5,
                show_label=True,
                autofocus=True,
            )
            with gr.Row():
                send_btn = gr.Button("Send question", elem_id="send-btn", variant="primary")
                clear_btn = gr.Button("Clear chat", elem_id="clear-btn", variant="secondary")

        # ── Evidence and citations panels ───────────────────────────────
        gr.HTML('<div class="section-label" style="margin-top:16px">Retrieved evidence</div>')
        evidence_output = gr.Markdown(
            value="*Evidence will appear here before an answer is generated.*",
            elem_id="evidence-box",
        )
        gr.HTML('<div class="section-label" style="margin-top:16px">Sources</div>')
        citations_output = gr.Markdown(
            value="*Sources will appear here after your first question.*",
            elem_id="citations-box",
        )

        # ── Debug Panel (only shown when DEBUG=true) ────────────────────
        if config.debug:
            gr.HTML('<div class="section-label" style="margin-top:12px">Debug information</div>')
            debug_output = gr.Markdown(elem_id="debug-box")
        else:
            debug_output = gr.Markdown(visible=False)

        # ── Example Questions ───────────────────────────────────────────
        gr.HTML('<div class="section-label" style="margin-top:16px">Example questions</div>')
        example_questions = [item["text"] for item in flat_sample_questions()]
        with gr.Row():
            for example_question in example_questions:
                example_button = gr.Button(
                    example_question,
                    variant="secondary",
                    size="sm",
                )
                example_button.click(
                    fn=example_loader(example_question),
                    outputs=query_input,
                    queue=False,
                )

        # ── Developer diagnostics page ────────────────────────────────
        with gr.Accordion("Developer diagnostics", open=False):
            gr.Markdown(
                "Request-stage timings and repeatable retrieval benchmarks. "
                "Benchmark history is stored locally under `.runtime/` and never includes API keys."
            )
            diagnostics_output = gr.Markdown(value=diagnostics_markdown(), elem_id="diagnostics-output")
            with gr.Row():
                refresh_diagnostics_btn = gr.Button("Refresh Traces", variant="secondary")
                run_benchmark_btn = gr.Button("Run Retrieval Benchmark", variant="primary")

            refresh_diagnostics_btn.click(
                fn=diagnostics_markdown,
                outputs=diagnostics_output,
                api_name="refresh_diagnostics",
                queue=False,
            )

            def run_benchmark_and_render():
                run_retrieval_benchmark()
                return diagnostics_markdown()

            run_benchmark_btn.click(
                fn=run_benchmark_and_render,
                outputs=diagnostics_output,
                api_name="run_retrieval_benchmark",
                queue=False,
            )

        # ── Footer ─────────────────────────────────────────────────────
        gr.HTML(
            """
            <div style="text-align:center; color:rgba(144,202,249,0.5);
                        font-size:0.75rem; margin-top:20px; padding:12px;
                        border-top: 1px solid rgba(100,181,246,0.1);">
              For educational purposes only · Not a substitute for professional medical advice
              · Always consult a qualified healthcare professional
            </div>
            """
        )

        # ── Event handlers ──────────────────────────────────────────────

        def _get_category_value(display_label: str) -> str:
            return CATEGORY_CHOICES.get(display_label, CATEGORY_ALL)

        def on_retrieve(message, chat_hist, cat_display, mem):
            cat = _get_category_value(cat_display)
            return retrieve_for_ui(message, chat_hist, cat, mem)

        def on_clear(mem):
            mem.clear()
            return (
                [],
                "*Evidence will appear here before an answer is generated.*",
                "*Sources will appear here after your first question.*",
                "",
                "",
                generation_provider_status(),
                mem,
            )

        send_event = send_btn.click(
            fn=on_retrieve,
            inputs=[query_input, chatbot, category_input, memory_state],
            outputs=[chatbot, evidence_output, evidence_state, query_input],
            queue=True,
        )
        send_event.success(
            fn=generate_for_ui,
            inputs=[chatbot, evidence_state, memory_state],
            outputs=[chatbot, citations_output, debug_output, memory_state, provider_output],
            queue=True,
        )

        submit_event = query_input.submit(
            fn=on_retrieve,
            inputs=[query_input, chatbot, category_input, memory_state],
            outputs=[chatbot, evidence_output, evidence_state, query_input],
            queue=True,
        )
        submit_event.success(
            fn=generate_for_ui,
            inputs=[chatbot, evidence_state, memory_state],
            outputs=[chatbot, citations_output, debug_output, memory_state, provider_output],
            queue=True,
        )

        clear_btn.click(
            fn=on_clear,
            inputs=[memory_state],
            outputs=[chatbot, evidence_output, citations_output, debug_output, query_input, provider_output, memory_state],
            queue=False,
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    config.validate()

    print("\n" + "═" * 60)
    print("  Diabetes RAG Assistant")
    print("═" * 60)
    print(f"  Embedding model : {config.embedding_model}")
    print(f"  Gemini model    : {config.gemini_model}")
    print(f"  Embedding source: {config.embedding_provider}")
    print(f"  DB namespace    : {config.resolved_embedding_namespace}")
    print(f"  Debug mode      : {'ON' if config.debug else 'off'}")
    print("═" * 60)

    # Check that PostgreSQL has indexed data
    from src.vector_store import vector_store
    stats = vector_store.collection_stats()
    total = sum(stats.values())
    if total == 0:
        print(
            "\n  ⚠️  WARNING: PostgreSQL contains no chunks for this namespace.\n"
            "  Run 'uv run python scripts/ingest.py' first.\n"
        )
    else:
        print(f"\n  Knowledge base: {total} chunks indexed")
        for cat, count in stats.items():
            print(f"    {cat:<15} {count} chunks")
        print()

    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        quiet=False,
        theme=GRADIO_THEME,
        css=CUSTOM_CSS,
    )
