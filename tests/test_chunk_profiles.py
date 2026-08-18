"""Tests for named chunk profiles.

Verifies that the three named profiles exist and have the correct
parameters (chunk size and overlap).
"""

from __future__ import annotations

from src.config import CHUNK_PROFILES


def test_chunk_profiles_definitions():
    """Verify that Small, Balanced, and Large profiles have correct size and overlap."""
    assert "small" in CHUNK_PROFILES
    assert CHUNK_PROFILES["small"] == (1200, 0)

    assert "balanced" in CHUNK_PROFILES
    assert CHUNK_PROFILES["balanced"] == (2000, 200)

    assert "large" in CHUNK_PROFILES
    assert CHUNK_PROFILES["large"] == (3000, 300)
