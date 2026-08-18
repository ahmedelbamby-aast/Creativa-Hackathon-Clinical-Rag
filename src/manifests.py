"""Manifest models for source tracking and index reproducibility.

SourceManifestEntry  — describes a single ingested knowledge source.
IndexManifest        — describes a specific vector index build.

Both models use plain dataclasses (matching project style in config.py and
retriever.py) with __post_init__ validation instead of a third-party library.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime


# ---------------------------------------------------------------------------
# SourceManifestEntry
# ---------------------------------------------------------------------------

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass
class SourceManifestEntry:
    """Typed record describing a single knowledge-base source document.

    Required fields must be non-empty.  Optional fields (version,
    licensing_note) default to empty string and are always accepted.
    """

    source_id: str
    title: str
    publisher: str
    source_url: str
    publication_date: str
    checksum_sha256: str
    enabled: bool
    version: str = ""
    licensing_note: str = ""

    def __post_init__(self) -> None:
        # Validate required non-empty string fields
        for attr in ("source_id", "title", "publisher", "source_url", "publication_date"):
            if not getattr(self, attr):
                raise ValueError(f"SourceManifestEntry.{attr} must not be empty")

        # Validate URL scheme
        if not self.source_url.startswith(("http://", "https://")):
            raise ValueError(
                f"SourceManifestEntry.source_url must start with http:// or https://, "
                f"got: {self.source_url!r}"
            )

        # Validate SHA-256 hex digest (exactly 64 lowercase hex characters)
        if not _SHA256_RE.match(self.checksum_sha256):
            raise ValueError(
                "SourceManifestEntry.checksum_sha256 must be a valid SHA-256 "
                "hexadecimal string (64 lowercase hex characters)"
            )


# ---------------------------------------------------------------------------
# IndexManifest
# ---------------------------------------------------------------------------

@dataclass
class IndexManifest:
    """Typed record describing a reproducible vector-index build.

    Captures enough information to recreate or verify an index:
    which namespace it lives in, what corpus it was built from,
    which chunking profile and embedding model were used, and when.
    """

    namespace: str
    corpus_hash: str
    chunk_profile: str
    embedding_model: str
    embedding_dimension: int
    created_at: datetime

    def __post_init__(self) -> None:
        for attr in ("namespace", "corpus_hash", "chunk_profile", "embedding_model"):
            if not getattr(self, attr):
                raise ValueError(f"IndexManifest.{attr} must not be empty")

        if self.embedding_dimension <= 0:
            raise ValueError(
                f"IndexManifest.embedding_dimension must be a positive integer, "
                f"got: {self.embedding_dimension}"
            )
