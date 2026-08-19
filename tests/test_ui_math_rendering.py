"""Static contract for safe equation rendering in the serverless chat UI."""

import hashlib
from pathlib import Path


INDEX = (
    Path(__file__).resolve().parents[1] / "backend" / "static" / "index.html"
).read_text(encoding="utf-8")


STATIC = Path(__file__).resolve().parents[1] / "backend" / "static"


def test_katex_is_pinned_and_self_hosted() -> None:
    assert 'href="/assets/katex/katex.min.css"' in INDEX
    assert 'src="/assets/katex/katex.min.js"' in INDEX
    assert "cdn.jsdelivr.net" not in INDEX
    assert (STATIC / "katex" / "LICENSE").is_file()
    assert (STATIC / "katex" / "fonts" / "KaTeX_Main-Regular.woff2").is_file()
    assert hashlib.sha256((STATIC / "katex" / "katex.min.js").read_bytes()).hexdigest() == (
        "68b9115510b8cedb9909a10de7799c94c0707481296f755c0a8888cb8fcde216"
    )


def test_math_renderer_supports_display_and_inline_delimiters() -> None:
    assert "function appendMath(container, token)" in INDEX
    assert "function renderMathNode(node)" in INDEX
    assert "math-display" in INDEX and "math-inline" in INDEX
    assert "\\\\\\[[\\s\\S]*?\\\\\\]" in INDEX
    assert "\\\\\\([\\s\\S]*?\\\\\\)" in INDEX
    assert "\\$\\$[\\s\\S]*?\\$\\$" in INDEX


def test_math_rendering_remains_untrusted_and_has_plain_text_fallback() -> None:
    assert "trust: false" in INDEX
    assert "strict: 'error'" in INDEX
    assert "maxExpand: 1000" in INDEX
    assert "node.textContent = latex.trim()" in INDEX
    assert "container.innerHTML" not in INDEX


def test_incomplete_arabic_display_math_is_repaired_before_katex() -> None:
    assert "function closeUnbalancedLatexBraces(latex)" in INDEX
    assert "function repairIncompleteDisplayMath(content)" in INDEX
    assert "closeUnbalancedLatexBraces(trimmed)" in INDEX
    assert "const repairedContent = repairIncompleteDisplayMath(content)" in INDEX
    assert "function splitArabicMathLabel(latex)" in INDEX
    assert "node.dataset.mathLabel = parsed.label" in INDEX
    assert "label.dir = 'rtl'" in INDEX
    assert "target.className = 'math-katex'" in INDEX
    assert "throwOnError: true" in INDEX
    assert "node.dir = 'ltr'" in INDEX
