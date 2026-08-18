"""Tests for named chunk profiles.

Verifies that the three named profiles exist and have the correct
parameters (chunk size and overlap).
"""

from __future__ import annotations

from src.config import CHUNK_PROFILES


def test_chunk_profiles_definitions():
    """Verify that Small, Balanced, and Large profiles have correct size and overlap."""
    assert "small" in CHUNK_PROFILES
    assert CHUNK_PROFILES["small"]["chunk_size"] == 1200
    assert CHUNK_PROFILES["small"]["chunk_overlap"] == 0

    assert "balanced" in CHUNK_PROFILES
    assert CHUNK_PROFILES["balanced"]["chunk_size"] == 2000
    assert CHUNK_PROFILES["balanced"]["chunk_overlap"] == 200

    assert "large" in CHUNK_PROFILES
    assert CHUNK_PROFILES["large"]["chunk_size"] == 3000
    assert CHUNK_PROFILES["large"]["chunk_overlap"] == 300
