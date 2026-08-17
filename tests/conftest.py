"""Shared fixtures for the Creativa Diabetes test suite."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def pipeline_main():
    """Load the CLI module without relying on ``chunking`` being a package."""
    path = ROOT / "chunking" / "main.py"
    spec = importlib.util.spec_from_file_location("chunking_main", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

