"""Diabetes RAG Assistant — Gradio Web UI.

Provides a clean, bilingual (English + Arabic) chat interface with:
  - Category selector (All / Treatment / Prevention / Nutrition)
  - Chat history with source citations
  - Debug panel (enabled via DEBUG=true in .env)
  - Clear conversation button

Run with:
    python app.py
"""

import logging
import re
import sys
from pathlib import Path

# Add project root to Python path when running directly
sys.path.insert(0, str(Path(__file__).resolve().parent))

import gradio as gr

from src.config import config, CATEGORY_ALL, CATEGORY_TREATMENT, CATEGORY_PREVENTION, CATEGORY_NUTRITION
from src.memory import ConversationMemory
from src.retriever import retrieve, is_retrieval_sufficient
from src.router import route_query
from src.rewriter import rewrite_query
from src.safety import classify_safety, SafetyLevel, get_disclaimer, get_emergency_response
from src.prompts import build_user_prompt, NO_RESULTS_RESPONSE_EN, NO_RESULTS_RESPONSE_AR
from src.citations import build_citation_list, build_debug_info
from src.generator import generator

logging.basicConfig(
    level=logging.DEBUG if config.debug else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.WARNING)
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

    is_ar = _is_arabic(query)

    # Step 1: Safety check
    safety_level = classify_safety(query)

    if safety_level == SafetyLevel.EMERGENCY:
        logger.warning("Emergency query detected: %r", query[:80])
        response = get_emergency_response(is_arabic=is_ar)
        memory.add_user(query, category=category)
        memory.add_assistant(response)
        return response, "", ""

    # Step 2: Rewrite query for better retrieval
    history = memory.get_history()
    rewritten = rewrite_query(query, conversation_history=history)

    # Step 3: Route to category
    routed_category = route_query(rewritten, user_selected_category=category)

    # Step 4: Retrieve
    chunks = retrieve(rewritten, category=routed_category, top_k=config.top_k)

    # Step 5: Check sufficiency
    if not is_retrieval_sufficient(chunks):
        logger.info("Insufficient retrieval for query: %r", query[:80])
        no_info = NO_RESULTS_RESPONSE_AR if is_ar else NO_RESULTS_RESPONSE_EN
        memory.add_user(query, category=category)
        memory.add_assistant(no_info)
        return no_info, "", ""

    # Step 6: Build citations
    citations = build_citation_list(chunks, is_arabic=is_ar)

    # Step 7: Generate answer
    try:
        prompt = build_user_prompt(query, chunks, conversation_history=history)
        answer = generator.generate(prompt)
    except RuntimeError as e:
        if "GEMINI_API_KEY" in str(e):
            answer = (
                "⚠️ **Configuration Error**: Gemini API key is not set.\n\n"
                "Please copy `.env.example` to `.env` and add your `GEMINI_API_KEY`."
            )
        else:
            answer = f"⚠️ **Generation error**: {e}"
        citations = ""
    except Exception as e:
        logger.error("Generation failed: %s", e)
        answer = f"⚠️ **An error occurred during generation**: {e}"
        citations = ""

    # Step 8: Append safety disclaimer
    disclaimer = get_disclaimer(safety_level, is_arabic=is_ar)
    if disclaimer:
        answer += disclaimer

    # Step 9: Update memory
    memory.add_user(query, category=category)
    memory.add_assistant(answer)

    # Step 10: Debug info
    debug = ""
    if config.debug:
        debug = build_debug_info(query, rewritten, routed_category, chunks)

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


def clear_chat(memory: ConversationMemory) -> tuple:
    memory.clear()
    return [], "*Sources will appear here after your first question.*", "", "", memory


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

CATEGORY_CHOICES = {
    "🔍 All Categories": CATEGORY_ALL,
    "💊 Treatment": CATEGORY_TREATMENT,
    "🛡️ Prevention": CATEGORY_PREVENTION,
    "🥗 Nutrition": CATEGORY_NUTRITION,
}

CATEGORY_DISPLAY_TO_VALUE = CATEGORY_CHOICES
CATEGORY_VALUE_TO_DISPLAY = {v: k for k, v in CATEGORY_CHOICES.items()}

CUSTOM_CSS = """
/* ── Global ─────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif !important; }

body, .gradio-container {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f2744 100%) !important;
    min-height: 100vh;
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

#header-subtitle {
    font-size: 1rem !important;
    color: #90caf9 !important;
    margin-top: 8px !important;
}

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

/* ── Chat Interface ──────────────────────────────────────────────────── */
#chatbot {
    background: rgba(15, 23, 42, 0.8) !important;
    border: 1px solid rgba(100, 181, 246, 0.2) !important;
    border-radius: 16px !important;
    box-shadow: 0 4px 24px rgba(0,0,0,0.3) !important;
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
    font-size: 0.95rem !important;
    transition: border-color 0.2s ease;
}

#query-input textarea:focus {
    border-color: #42a5f5 !important;
    box-shadow: 0 0 0 3px rgba(66, 165, 245, 0.15) !important;
}

#query-input textarea::placeholder {
    color: rgba(144, 202, 249, 0.5) !important;
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

        # ── Header ─────────────────────────────────────────────────────
        with gr.Column(elem_id="header-box"):
            gr.HTML(
                """
                <div>
                  <div id="header-title">🩺 Diabetes RAG Assistant</div>
                  <div id="header-subtitle">
                    Grounded answers from verified diabetes guidelines · English &amp; العربية
                  </div>
                </div>
                """
            )

        # ── Category Selector ───────────────────────────────────────────
        with gr.Row():
            category_input = gr.Dropdown(
                choices=list(CATEGORY_CHOICES.keys()),
                value="🔍 All Categories",
                label="Knowledge Category",
                elem_id="category-selector",
                scale=2,
                interactive=True,
            )
            gr.HTML(
                """
                <div style="color:#90caf9; font-size:0.82rem; padding-top:28px; line-height:1.5;">
                  Select a category to focus retrieval, or use <strong>All Categories</strong>
                  for cross-domain questions.
                </div>
                """,
                elem_id="category-help",
            )

        gr.HTML('<div style="height:4px"></div>')

        # ── Chat Interface ──────────────────────────────────────────────
        chatbot = gr.Chatbot(
            label="",
            elem_id="chatbot",
            height=460,
            show_label=False,
            render_markdown=True,
        )

        with gr.Row():
            query_input = gr.Textbox(
                placeholder="Ask about diabetes treatment, prevention, nutrition... (English or Arabic)",
                label="",
                elem_id="query-input",
                lines=2,
                max_lines=5,
                scale=5,
                show_label=False,
                autofocus=True,
            )
            with gr.Column(scale=1, min_width=120):
                send_btn = gr.Button("Send ↵", elem_id="send-btn", variant="primary")
                clear_btn = gr.Button("🗑️ Clear", elem_id="clear-btn", variant="secondary")

        # ── Citations Panel ─────────────────────────────────────────────
        gr.HTML('<div class="section-label" style="margin-top:16px">📚 Sources</div>')
        citations_output = gr.Markdown(
            value="*Sources will appear here after your first question.*",
            elem_id="citations-box",
        )

        # ── Debug Panel (only shown when DEBUG=true) ────────────────────
        if config.debug:
            gr.HTML('<div class="section-label" style="margin-top:12px">🔍 Debug Info</div>')
            debug_output = gr.Markdown(elem_id="debug-box")
        else:
            debug_output = gr.Markdown(visible=False)

        # ── Example Questions ───────────────────────────────────────────
        gr.HTML('<div class="section-label" style="margin-top:16px">💡 Example Questions</div>')
        gr.Examples(
            examples=[
                ["What foods are recommended for people with diabetes?"],
                ["How does physical activity help prevent type 2 diabetes?"],
                ["What medications are mentioned in the diabetes guidelines?"],
                ["ما هي الأطعمة الموصى بها لمريض السكري؟"],
                ["كيف يمكن الوقاية من مرض السكري النوع الثاني؟"],
                ["What is HbA1c and why is it important?"],
                ["Can a person with diabetes eat rice or bread?"],
                ["What are the main risk factors for type 2 diabetes?"],
            ],
            inputs=query_input,
            label="",
        )

        # ── Footer ─────────────────────────────────────────────────────
        gr.HTML(
            """
            <div style="text-align:center; color:rgba(144,202,249,0.5);
                        font-size:0.75rem; margin-top:20px; padding:12px;
                        border-top: 1px solid rgba(100,181,246,0.1);">
              ⚕️ For educational purposes only · Not a substitute for professional medical advice
              · Always consult a qualified healthcare professional
            </div>
            """
        )

        # ── Event handlers ──────────────────────────────────────────────

        def _get_category_value(display_label: str) -> str:
            return CATEGORY_CHOICES.get(display_label, CATEGORY_ALL)

        def on_send(message, chat_hist, cat_display, mem):
            cat = _get_category_value(cat_display)
            updated_hist, cits, dbg, empty_msg, updated_mem = chat(
                message, chat_hist, cat, mem
            )
            return updated_hist, cits, dbg, empty_msg, updated_mem

        def on_clear(mem):
            mem.clear()
            return [], "*Sources will appear here after your first question.*", "", "", mem

        send_btn.click(
            fn=on_send,
            inputs=[query_input, chatbot, category_input, memory_state],
            outputs=[chatbot, citations_output, debug_output, query_input, memory_state],
        )

        query_input.submit(
            fn=on_send,
            inputs=[query_input, chatbot, category_input, memory_state],
            outputs=[chatbot, citations_output, debug_output, query_input, memory_state],
        )

        clear_btn.click(
            fn=on_clear,
            inputs=[memory_state],
            outputs=[chatbot, citations_output, debug_output, query_input, memory_state],
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
    print(f"  ChromaDB path   : {config.chroma_db_dir}")
    print(f"  Debug mode      : {'ON' if config.debug else 'off'}")
    print("═" * 60)

    # Check that ChromaDB has data
    from src.vector_store import vector_store
    stats = vector_store.collection_stats()
    total = sum(stats.values())
    if total == 0:
        print(
            "\n  ⚠️  WARNING: ChromaDB collections are empty.\n"
            "  Run 'python scripts/ingest.py' first to ingest your documents.\n"
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
