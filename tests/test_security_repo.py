"""Security and repository hygiene tests.

Verifies that secrets are not committed, Docker build context is clean,
and no API keys are hard-coded in active source files.

IMPORTANT: These tests do NOT print, log, or assert the actual value of any
secret.  They only check structural properties (file existence, pattern
absence).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# .env handling
# ---------------------------------------------------------------------------

def test_dotenv_is_listed_in_gitignore():
    """.gitignore must contain a rule that covers .env."""
    gitignore = ROOT / ".gitignore"
    assert gitignore.exists(), ".gitignore file must exist"
    content = gitignore.read_text(encoding="utf-8")
    # Accept ".env" as a standalone entry (with or without leading slash)
    lines = [ln.strip() for ln in content.splitlines()]
    assert any(ln in (".env", "/.env") for ln in lines), (
        ".gitignore must contain a '.env' entry"
    )


def test_dockerignore_excludes_dotenv():
    """.dockerignore must list .env so it is excluded from any Docker build context."""
    dockerignore = ROOT / ".dockerignore"
    assert dockerignore.exists(), ".dockerignore must exist at the project root"
    content = dockerignore.read_text(encoding="utf-8")
    lines = [ln.strip() for ln in content.splitlines()]
    assert any(ln in (".env", "/.env") for ln in lines), (
        ".dockerignore must contain a '.env' entry"
    )


# ---------------------------------------------------------------------------
# No hard-coded secrets
# ---------------------------------------------------------------------------

# Pattern for Google AI / Gemini API key prefix (never matches the env-var name itself)
_GEMINI_KEY_RE = re.compile(r"AIzaSy[0-9A-Za-z\-_]{33}")
# Pattern for OpenAI-style keys
_OPENAI_KEY_RE = re.compile(r"sk-[A-Za-z0-9]{20,}")

_SOURCE_EXTENSIONS = {".py", ".yaml", ".yml", ".toml", ".json"}

# Directories to exclude from scanning (third-party code, compiled output)
_EXCLUDED_DIRS = {".venv", ".git", "__pycache__", "htmlcov", ".pytest_cache"}


def _collect_source_files() -> list[Path]:
    files = []
    for path in ROOT.rglob("*"):
        if path.suffix not in _SOURCE_EXTENSIONS:
            continue
        if any(part in _EXCLUDED_DIRS for part in path.parts):
            continue
        if path.is_file():
            files.append(path)
    return files


def test_no_gemini_api_key_hard_coded_in_source():
    """No source file may contain a literal Gemini API key value."""
    for path in _collect_source_files():
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        assert not _GEMINI_KEY_RE.search(content), (
            f"Possible Gemini API key found in {path.relative_to(ROOT)} — "
            "remove it and rotate the key immediately."
        )


def test_no_openai_style_key_hard_coded_in_source():
    """No source file may contain an OpenAI-style sk- secret key."""
    for path in _collect_source_files():
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        assert not _OPENAI_KEY_RE.search(content), (
            f"Possible secret key found in {path.relative_to(ROOT)} — "
            "remove it and rotate the key immediately."
        )
