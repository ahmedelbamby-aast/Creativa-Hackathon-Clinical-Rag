"""PostgreSQL/pgvector storage for RAG chunks and similarity search.

Parent table selection
----------------------
The baseline ``rag_chunks`` table stores 384-d embeddings and remains
untouched for rollback safety.  Every higher dimension introduced by the
Gemini Embedding 2 sequential rollout lives in its own parent table:

    Dimension │ Parent table        │ Schema file
    ──────────┼─────────────────────┼────────────────────────
    384       │ rag_chunks          │ database/schema.sql
    768       │ rag_chunks_d768     │ database/schema_d768.sql
    1024      │ rag_chunks_d1024    │ database/schema_d1024.sql
    2048      │ rag_chunks_d2048    │ database/schema_d2048.sql
    3072      │ rag_chunks_d3072    │ database/schema_d3072.sql

Each parent table uses ``PARTITION BY LIST (namespace)`` so multiple
Gemini-embedding namespaces (e.g. ``gemini_768``, ``gemini_768_v2``) can
coexist without schema changes.  Partitions and per-partition HNSW indexes
are created dynamically by :meth:`VectorStore.ensure_schema`.

Vectors from different parent tables are *never* mixed: the cosine-distance
operator ``<=>`` would silently return wrong results for mismatched
dimensions.  :class:`VectorStore` enforces the dimension contract on both
reads and writes.
"""

import re
from pathlib import Path
from typing import Optional

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector
from psycopg import sql
from psycopg.rows import dict_row

from src.config import ALL_CATEGORIES, CATEGORY_ALL, CATEGORY_GENERAL, config
from src.scoring import cosine_distance_to_score
from src.source_catalog import load_source_catalog


_DATABASE_DIR = Path(__file__).resolve().parent.parent / "database"

# Mapping: output dimension → (parent table name, schema SQL file).
# 384 is the backward-compatible baseline; add entries here for every new
# Gemini Embedding 2 stage before running its ingestion migration.
_DIMENSION_TABLE_MAP: dict[int, tuple[str, Path]] = {
    384:  ("rag_chunks",       _DATABASE_DIR / "schema.sql"),
    768:  ("rag_chunks_d768",  _DATABASE_DIR / "schema_d768.sql"),
    1024: ("rag_chunks_d1024", _DATABASE_DIR / "schema_d1024.sql"),
    2048: ("rag_chunks_d2048", _DATABASE_DIR / "schema_d2048.sql"),
    3072: ("rag_chunks_d3072", _DATABASE_DIR / "schema_d3072.sql"),
}

# Convenience alias kept for callers that reference SCHEMA_PATH directly.
SCHEMA_PATH = _DATABASE_DIR / "schema.sql"

_LEXICAL_STOPWORDS = {
    "about", "adults", "answer", "approximately", "based", "condition", "diabetes",
    "does", "figures", "from", "have", "living", "many", "million", "number", "percentage",
    "source", "that", "their", "them", "topic", "using", "what", "when", "with", "years",
}


def _lexical_terms(query: str) -> list[str]:
    """Return bounded, discriminative English/numeric terms for SQL fallback."""
    terms = []
    for token in re.findall(r"[a-z0-9]+", query.casefold()):
        if token in _LEXICAL_STOPWORDS or (not token.isdigit() and len(token) < 4):
            continue
        if token not in terms:
            terms.append(token)
    return terms[:12]


def normalize_namespace(namespace: str) -> str:
    """Return a safe PostgreSQL identifier component."""
    normalized = re.sub(r"[^a-z0-9_]+", "_", namespace.lower()).strip("_")
    if not normalized:
        raise ValueError("Embedding namespace cannot be empty")
    return normalized[:40]


def _resolve_parent_table(dimension: int) -> tuple[str, Path]:
    """Return the (parent_table_name, schema_path) for a given dimension.

    Raises
    ------
    ValueError
        If no parent table has been registered for *dimension*.  Always add
        a row to ``_DIMENSION_TABLE_MAP`` and supply the matching SQL
        migration file before using a new dimension.
    """
    entry = _DIMENSION_TABLE_MAP.get(dimension)
    if entry is None:
        supported = ", ".join(str(d) for d in sorted(_DIMENSION_TABLE_MAP))
        raise ValueError(
            f"No parent table registered for {dimension}-d embeddings. "
            f"Supported dimensions: {supported}. "
            f"Add a migration to database/ and register it in _DIMENSION_TABLE_MAP."
        )
    return entry


class VectorStore:
    """Store and retrieve one embedding namespace in PostgreSQL.

    The active parent table is selected automatically from ``dimension``.
    Vectors from different parent tables are never mixed in a single query.
    """

    def __init__(
        self,
        database_url: Optional[str] = None,
        namespace: Optional[str] = None,
        dimension: Optional[int] = None,
    ) -> None:
        self.database_url = database_url or config.database_url
        self.namespace = normalize_namespace(
            namespace or config.resolved_embedding_namespace
        )
        self.dimension = dimension or config.embedding_dimension

        # Resolve the parent table and schema file for this dimension.
        # Raises ValueError immediately if the dimension is not registered.
        self.parent_table, self._schema_path = _resolve_parent_table(self.dimension)

        # The namespace partition lives inside the dimension-specific parent.
        self.partition_name = f"{self.parent_table}_{self.namespace}"
        self._schema_ready = False

    def ensure_schema(self) -> None:
        """Enable pgvector and create the current namespace partition/index.

        The parent table must exist before this method creates the
        namespace partition.  Run the dimension-specific SQL migration
        (e.g. ``database/schema_d768.sql``) against the direct/unpooled
        connection on an isolated Neon branch first.

        Dimension safety check
        ----------------------
        Reads ``atttypmod`` from ``pg_attribute`` for *this* parent table to
        confirm that the deployed schema matches the configured dimension.
        Raises ``RuntimeError`` if they diverge, preventing silent
        mixed-dimension corruption.
        """
        schema_sql = self._schema_path.read_text(encoding="utf-8")
        with psycopg.connect(config.schema_database_url, autocommit=True) as connection:
            connection.execute(schema_sql)
            register_vector(connection)

            # Check the vector column width on the correct parent table, not
            # the hardcoded 384-d baseline.  atttypmod stores the declared
            # number of dimensions for pgvector columns.
            actual_dimension = connection.execute(
                sql.SQL(
                    """
                    SELECT atttypmod
                    FROM pg_attribute
                    WHERE attrelid = {}::regclass
                      AND attname = 'embedding'
                      AND NOT attisdropped
                    """
                ).format(sql.Literal(self.parent_table)),
            ).fetchone()

            if actual_dimension is None:
                raise RuntimeError(
                    f"Parent table '{self.parent_table}' does not exist or has no "
                    f"'embedding' column.  Run the {self._schema_path.name} migration "
                    f"against this database before calling ensure_schema()."
                )

            if actual_dimension[0] != self.dimension:
                raise RuntimeError(
                    f"Parent table '{self.parent_table}' stores {actual_dimension[0]}-d "
                    f"vectors but configuration requires {self.dimension}-d.  "
                    f"Check EMBEDDING_DIMENSION and the applied migration."
                )

            connection.execute(
                sql.SQL(
                    "CREATE TABLE IF NOT EXISTS {} PARTITION OF {} "
                    "FOR VALUES IN ({})"
                ).format(
                    sql.Identifier(self.partition_name),
                    sql.Identifier(self.parent_table),
                    sql.Literal(self.namespace),
                )
            )
            connection.execute(
                sql.SQL(
                    "CREATE INDEX IF NOT EXISTS {} ON {} "
                    "USING hnsw (embedding vector_cosine_ops) "
                    "WITH (m = 16, ef_construction = 64)"
                ).format(
                    sql.Identifier(f"{self.partition_name}_embedding_hnsw_idx"),
                    sql.Identifier(self.partition_name),
                )
            )
        self._schema_ready = True

    def _connect(self):
        if not self._schema_ready and config.auto_create_schema:
            self.ensure_schema()
        connection = psycopg.connect(self.database_url, row_factory=dict_row)
        register_vector(connection)
        return connection

    def add_chunks(
        self,
        chunk_records: list[dict],
        embeddings: list[list[float]],
        batch_size: int = 100,
    ) -> dict[str, int]:
        """Upsert chunk records and embeddings in batches."""
        if len(chunk_records) != len(embeddings):
            raise ValueError("chunk records and embeddings must have equal lengths")
        if any(len(embedding) != self.dimension for embedding in embeddings):
            raise ValueError(f"all embeddings must have {self.dimension} dimensions")

        statement = sql.SQL(
            """
            INSERT INTO {} (
                namespace, chunk_id, document_name, page_number,
                section_title, subsection_title, category, content_type,
                language, source_id, source_url, content, char_count, word_count, quality_score,
                embedding
            ) VALUES (
                %(namespace)s, %(chunk_id)s, %(document_name)s, %(page_number)s,
                %(section_title)s, %(subsection_title)s, %(category)s,
                %(content_type)s, %(language)s, %(source_id)s, %(source_url)s, %(content)s,
                %(char_count)s, %(word_count)s, %(quality_score)s, %(embedding)s
            )
            ON CONFLICT (namespace, chunk_id) DO UPDATE SET
                document_name = EXCLUDED.document_name,
                page_number = EXCLUDED.page_number,
                section_title = EXCLUDED.section_title,
                subsection_title = EXCLUDED.subsection_title,
                category = EXCLUDED.category,
                content_type = EXCLUDED.content_type,
                language = EXCLUDED.language,
                source_id = EXCLUDED.source_id,
                source_url = EXCLUDED.source_url,
                content = EXCLUDED.content,
                char_count = EXCLUDED.char_count,
                word_count = EXCLUDED.word_count,
                quality_score = EXCLUDED.quality_score,
                embedding = EXCLUDED.embedding,
                updated_at = now()
            """
        ).format(sql.Identifier(self.parent_table))

        rows = []
        category_counts = {category: 0 for category in ALL_CATEGORIES}
        for record, embedding in zip(chunk_records, embeddings):
            category = record.get("category", CATEGORY_GENERAL)
            if category == CATEGORY_GENERAL:
                for name in ALL_CATEGORIES:
                    category_counts[name] += 1
            elif category in category_counts:
                category_counts[category] += 1

            rows.append(
                {
                    "namespace": self.namespace,
                    "chunk_id": record["chunk_id"],
                    "document_name": record["document_name"],
                    "page_number": record.get("page_number"),
                    "section_title": record.get("section_title", ""),
                    "subsection_title": record.get("subsection_title", ""),
                    "category": category,
                    "content_type": record.get("content_type", "text"),
                    "language": record.get("language", "en"),
                    "source_id": record.get("source_id", ""),
                    "source_url": record.get("source_url", ""),
                    "content": record["text"],
                    "char_count": record.get("char_count", len(record["text"])),
                    "word_count": record.get("word_count", len(record["text"].split())),
                    "quality_score": record.get("quality_score"),
                    "embedding": Vector(embedding),
                }
            )

        with self._connect() as connection:
            for start in range(0, len(rows), batch_size):
                with connection.cursor() as cursor:
                    cursor.executemany(statement, rows[start : start + batch_size])
        return category_counts

    def has_document(self, document_name: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                sql.SQL(
                    """
                    SELECT EXISTS (
                        SELECT 1 FROM {}
                        WHERE namespace = %s AND document_name = %s
                    ) AS found
                    """
                ).format(sql.Identifier(self.parent_table)),
                (self.namespace, document_name),
            ).fetchone()
        return bool(row["found"])

    def delete_document(self, document_name: str) -> None:
        with self._connect() as connection:
            connection.execute(
                sql.SQL(
                    "DELETE FROM {} WHERE namespace = %s AND document_name = %s"
                ).format(sql.Identifier(self.parent_table)),
                (self.namespace, document_name),
            )

    def query(
        self,
        query_embedding: list[float],
        category: str = CATEGORY_ALL,
        top_k: int = 5,
        where: Optional[dict] = None,
    ) -> list[dict]:
        """Return nearest chunks using cosine distance."""
        if where:
            raise ValueError("Use the category argument for retrieval filtering")
        if len(query_embedding) != self.dimension:
            raise ValueError(f"query embedding must have {self.dimension} dimensions")

        category_clause_sql = sql.SQL("")
        parameters: list = []
        if category != CATEGORY_ALL and category in ALL_CATEGORIES:
            category_clause_sql = sql.SQL("AND (category = %s OR category = %s)")
            parameters.extend([category, CATEGORY_GENERAL])
        vector = Vector(query_embedding)
        limit = max(0, min(top_k, 1000))
        statement = sql.SQL(
            """
            SELECT
                chunk_id AS id,
                content AS document,
                document_name,
                page_number,
                section_title,
                subsection_title,
                category,
                content_type,
                language,
                quality_score,
                embedding <=> %s AS distance
            FROM {}
            WHERE namespace = %s
            {}
            ORDER BY embedding <=> %s
            LIMIT %s
            """
        ).format(
            sql.Identifier(self.parent_table),
            category_clause_sql,
        )
        query_parameters = [vector, self.namespace, *parameters, vector, limit]
        with self._connect() as connection:
            rows = connection.execute(statement, query_parameters).fetchall()

        results = []
        source_catalog = load_source_catalog()
        for row in rows:
            distance = float(row.pop("distance"))
            chunk_id = row.pop("id")
            document = row.pop("document")
            source = source_catalog.get(row.get("document_name", ""))
            row["source_id"] = row.get("source_id") or (
                source.source_id if source and source.enabled else ""
            )
            row["source_url"] = row.get("source_url") or (
                source.source_url if source and source.enabled else ""
            )
            row["publisher"] = getattr(source, "publisher", "") if source and source.enabled else ""
            row["publication_date"] = getattr(source, "publication_date", "") if source and source.enabled else ""
            row["source_checksum"] = getattr(source, "checksum", "") if source and source.enabled else ""
            results.append(
                {
                    "id": chunk_id,
                    "document": document,
                    "metadata": {"chunk_id": chunk_id, **row},
                    "distance": distance,
                    "score": cosine_distance_to_score(distance),
                }
            )
        return results

    def get_chunks(self, chunk_ids: list[str]) -> list[dict]:
        """Load exact evidence by ID without embedding or nearest-neighbor search."""
        if not chunk_ids:
            return []
        unique_ids = list(dict.fromkeys(chunk_ids))
        statement = sql.SQL(
            """
            SELECT
                chunk_id AS id,
                content AS document,
                document_name,
                page_number,
                section_title,
                subsection_title,
                category,
                content_type,
                language,
                quality_score
            FROM {}
            WHERE namespace = %s AND chunk_id = ANY(%s)
            """
        ).format(sql.Identifier(self.parent_table))
        with self._connect() as connection:
            rows = connection.execute(statement, (self.namespace, unique_ids)).fetchall()
        source_catalog = load_source_catalog()
        by_id = {}
        for row in rows:
            source = source_catalog.get(row.get("document_name", ""))
            row["source_id"] = row.get("source_id") or (
                source.source_id if source and source.enabled else ""
            )
            row["source_url"] = row.get("source_url") or (
                source.source_url if source and source.enabled else ""
            )
            row["publisher"] = getattr(source, "publisher", "") if source and source.enabled else ""
            row["publication_date"] = getattr(source, "publication_date", "") if source and source.enabled else ""
            row["source_checksum"] = getattr(source, "checksum", "") if source and source.enabled else ""
            by_id[row["id"]] = {
                "id": row["id"],
                "document": row["document"],
                "metadata": {
                    key: value
                    for key, value in row.items()
                    if key not in {"id", "document"}
                },
                "distance": 0.0,
                "score": 0.0,
            }
        return [by_id[chunk_id] for chunk_id in chunk_ids if chunk_id in by_id]

    def keyword_query(
        self,
        query: str,
        category: str = CATEGORY_ALL,
        top_k: int = 5,
    ) -> list[dict]:
        """Search corpus text without an embedding provider, ranked by term coverage."""
        terms = _lexical_terms(query)
        if not terms:
            return []
        category_clause = ""
        parameters: list[object] = []
        if category != CATEGORY_ALL and category in ALL_CATEGORIES:
            category_clause = "AND (category = %s OR category = %s)"
            parameters.extend([category, CATEGORY_GENERAL])

        match_parts = ["CASE WHEN content ILIKE %s THEN 1 ELSE 0 END" for _ in terms]
        score_expression = " + ".join(match_parts)
        match_parameters = [f"%{term}%" for term in terms]
        limit = max(0, min(top_k, 1000))
        statement = sql.SQL(
            f"""
            SELECT
                chunk_id AS id,
                content AS document,
                document_name,
                page_number,
                section_title,
                subsection_title,
                category,
                content_type,
                language,
                quality_score,
                ({score_expression})::float / %s AS lexical_score
            FROM {{}}
            WHERE namespace = %s
              {category_clause}
              AND ({score_expression}) > 0
            ORDER BY lexical_score DESC, quality_score DESC, chunk_id
            LIMIT %s
            """
        ).format(sql.Identifier(self.parent_table))
        query_parameters = [
            *match_parameters,
            len(terms),
            self.namespace,
            *parameters,
            *match_parameters,
            limit,
        ]
        with self._connect() as connection:
            rows = connection.execute(statement, query_parameters).fetchall()

        source_catalog = load_source_catalog()
        results: list[dict] = []
        for row in rows:
            coverage = float(row.pop("lexical_score"))
            chunk_id = row.pop("id")
            document = row.pop("document")
            source = source_catalog.get(row.get("document_name", ""))
            row["source_id"] = source.source_id if source and source.enabled else ""
            row["source_url"] = source.source_url if source and source.enabled else ""
            row["publisher"] = getattr(source, "publisher", "") if source and source.enabled else ""
            row["publication_date"] = getattr(source, "publication_date", "") if source and source.enabled else ""
            row["source_checksum"] = getattr(source, "checksum", "") if source and source.enabled else ""
            score = round(0.5 + (0.5 * max(0.0, min(1.0, coverage))), 4)
            results.append(
                {
                    "id": chunk_id,
                    "document": document,
                    "metadata": {"chunk_id": chunk_id, **row},
                    "distance": round(1.0 - score, 4),
                    "score": score,
                }
            )
        return results

    def collection_stats(self) -> dict[str, int]:
        """Return category counts, including general chunks in each category."""
        with self._connect() as connection:
            rows = connection.execute(
                sql.SQL(
                    """
                    SELECT category, count(*) AS count
                    FROM {}
                    WHERE namespace = %s
                    GROUP BY category
                    """
                ).format(sql.Identifier(self.parent_table)),
                (self.namespace,),
            ).fetchall()
        counts = {row["category"]: row["count"] for row in rows}
        general = counts.get(CATEGORY_GENERAL, 0)
        return {
            category: counts.get(category, 0) + general
            for category in ALL_CATEGORIES
        }

    def reset_all(self) -> None:
        """Remove every chunk in the active embedding namespace."""
        with self._connect() as connection:
            connection.execute(
                sql.SQL(
                    "DELETE FROM {} WHERE namespace = %s"
                ).format(sql.Identifier(self.parent_table)),
                (self.namespace,),
            )

    def healthcheck(self) -> dict[str, str]:
        """Return PostgreSQL and pgvector versions."""
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT
                    current_setting('server_version') AS postgres,
                    extversion AS pgvector
                FROM pg_extension
                WHERE extname = 'vector'
                """
            ).fetchone()
        return {"postgres": row["postgres"], "pgvector": row["pgvector"]}

    def namespace_audit(self) -> dict:
        """Return document counts and vector-width violations for the active index."""
        with self._connect() as connection:
            rows = connection.execute(
                sql.SQL(
                    """
                    SELECT document_name, count(*) AS chunk_count,
                           count(*) FILTER (WHERE vector_dims(embedding) <> %s) AS invalid_vectors
                    FROM {}
                    WHERE namespace = %s
                    GROUP BY document_name
                    ORDER BY document_name
                    """
                ).format(sql.Identifier(self.parent_table)),
                (self.dimension, self.namespace),
            ).fetchall()
        documents = {
            row["document_name"]: {
                "chunk_count": int(row["chunk_count"]),
                "invalid_vectors": int(row["invalid_vectors"]),
            }
            for row in rows
        }
        return {
            "namespace": self.namespace,
            "table_family": self.parent_table,
            "dimension": self.dimension,
            "document_count": len(documents),
            "invalid_vector_count": sum(item["invalid_vectors"] for item in documents.values()),
            "documents": documents,
        }


vector_store = VectorStore()
