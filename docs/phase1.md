# Phase 1 Documentation

This document describes the Day 1 Foundation specifications implemented in Phase 1 of the Clinical RAG system.

---

## 1. Certified Two-Source Corpus

The knowledge base is built from exactly two certified, verified source documents. Their details and checksums are captured in `data/sources.json`:

1. **IDF Diabetes Atlas (11th Edition, 2025)**
   - **ID**: `idf-atlas-11-2025`
   - **Publisher**: International Diabetes Federation
   - **SHA-256 Checksum**: `81d01e73d486adbe7d4f14752644e109ed43151738e3e852303ebd3967a81d0d`
   
2. **WHO Recommendations on Care for Women with Diabetes During Pregnancy (2023)**
   - **ID**: `who-women-diabetes-pregnancy`
   - **Publisher**: World Health Organization
   - **SHA-256 Checksum**: `fb0dcf418df0757764bd6fd2d142c8d7e2690d6659ef67d049db39ac4537c606`

---

## 2. Models & Manifests

### Source Manifest Model (`SourceManifestEntry`)
A typed Python dataclass located in `src/manifests.py` representing one verified knowledge source. It validates identity fields (non-empty strings), SHA-256 checksum format (64 lowercase hex characters), and URL format (`http://` or `https://`).

### Index Manifest Model (`IndexManifest`)
A typed Python dataclass located in `src/manifests.py` describing a reproducible index build. It ensures that the namespace, corpus hash, chunk profile, and embedding model parameters are recorded on creation, validating that `embedding_dimension > 0`.

---

## 3. Chunk Profiles

Three named chunk profiles are configured in `src/config.py` under the `CHUNK_PROFILES` mapping for future Phase 2 evaluation:

- **Small**: `chunk_size = 1200` characters, `chunk_overlap = 0`
- **Balanced**: `chunk_size = 2000` characters, `chunk_overlap = 200` *(Default production profile)*
- **Large**: `chunk_size = 3000` characters, `chunk_overlap = 300`

---

## 4. Configuration Variables

Three new environment-backed variables are supported in `src/config.py`:
- `ACTIVE_INDEX_NAMESPACE`: Configures the active namespace partition in pgvector to retrieve chunks from.
- `RETRIEVAL_PROFILE`: Configures the retrieval pipeline parameters.
- `TOP_K`: Restricts retrieval limit (validated in `validate()` as positive integer).

---

## 5. Indexing Safety & Rollback

Atomic indexing is implemented in `src/ingestion/indexer.py` via `ingest_certified_corpus()`:
- Built-in checksum verification ensures that no corrupted files are processed.
- All documents are indexed and embedded into the target namespace completely before saving the `IndexManifest` file.
- Any parsing, chunking, or database failure aborts the process immediately, leaving the previous active namespace intact.
- Rollback is supported naturally since historical partitions are not deleted.
