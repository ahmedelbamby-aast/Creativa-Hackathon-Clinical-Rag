"""Consistency checks for deterministic project defaults."""

from pathlib import Path

from src.config import AppConfig


def test_gemini_model_default_and_template_are_consistent(monkeypatch):
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    assert AppConfig().gemini_model == "gemini-3.6-flash"
    template = Path(".env.example").read_text(encoding="utf-8")
    assert "GEMINI_MODEL=gemini-3.6-flash" in template
