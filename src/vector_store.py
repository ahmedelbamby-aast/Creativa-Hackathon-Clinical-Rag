"""PostgreSQL/pgvector storage for RAG chunks and similarity search."""

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


SCHEMA_PATH = Path(__file__).resolve().parent.parent / "database" / "schema.sql"


def normalize_namespace(namespace: str) -> str:
    """Return a safe PostgreSQL identifier component."""
    normalized = re.sub(r"[^a-z0-9_]+", "_", namespace.lower()).strip("_")
    if not normalized:
        raise ValueError("Embedding namespace cannot be empty")
    return normalized[:40]


class VectorStore:
    """Store and retrieve one embedding namespace in PostgreSQL."""

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
        self.partition_name = f"rag_chunks_{self.namespace}"
        self._schema_ready = False

    def ensure_schema(self) -> None:
        """Enable pgvector and create the current namespace partition/index."""
        schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")
        with psycopg.connect(self.database_url, autocommit=True) as connection:
            connection.execute(schema_sql)
            register_vector(connection)

            actual_dimension = connection.execute(
                """
                SELECT atttypmod
                FROM pg_attribute
                WHERE attrelid = 'rag_chunks'::regclass
                  AND attname = 'embedding'
                  AND NOT attisdropped
                """
            ).fetchone()[0]
            if actual_dimension != self.dimension:
                raise RuntimeError(
                    f"Database vector dimension is {actual_dimension}; "
                    f"configuration requires {self.dimension}."
                )

            connection.execute(
                sql.SQL(
                    "CREATE TABLE IF NOT EXISTS {} PARTITION OF rag_chunks "
                    "FOR VALUES IN ({})"
                ).format(
                    sql.Identifier(self.partition_name),
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
        if not self._schema_ready:
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

        statement = """
            INSERT INTO rag_chunks (
                namespace, chunk_id, document_name, page_number,
                section_title, subsection_title, category, content_type,
                language, content, char_count, word_count, quality_score,
                embedding
            ) VALUES (
                %(namespace)s, %(chunk_id)s, %(document_name)s, %(page_number)s,
                %(section_title)s, %(subsection_title)s, %(category)s,
                %(content_type)s, %(language)s, %(content)s, %(char_count)s,
                %(word_count)s, %(quality_score)s, %(embedding)s
            )
            ON CONFLICT (namespace, chunk_id) DO UPDATE SET
                document_name = EXCLUDED.document_name,
                page_number = EXCLUDED.page_number,
                section_title = EXCLUDED.section_title,
                subsection_title = EXCLUDED.subsection_title,
                category = EXCLUDED.category,
                content_type = EXCLUDED.content_type,
                language = EXCLUDED.language,
                content = EXCLUDED.content,
                char_count = EXCLUDED.char_count,
                word_count = EXCLUDED.word_count,
                quality_score = EXCLUDED.quality_score,
                embedding = EXCLUDED.embedding,
                updated_at = now()
        """
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
                """
                SELECT EXISTS (
                    SELECT 1 FROM rag_chunks
                    WHERE namespace = %s AND document_name = %s
                ) AS found
                """,
                (self.namespace, document_name),
            ).fetchone()
        return bool(row["found"])

    def delete_document(self, document_name: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM rag_chunks WHERE namespace = %s AND document_name = %s",
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

        category_clause = ""
        parameters: list = []
        if category != CATEGORY_ALL and category in ALL_CATEGORIES:
            category_clause = "AND (category = %s OR category = %s)"
            parameters.extend([category, CATEGORY_GENERAL])
        vector = Vector(query_embedding)
        limit = max(0, min(top_k, 1000))
        statement = f"""
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
            FROM rag_chunks
            WHERE namespace = %s
            {category_clause}
            ORDER BY embedding <=> %s
            LIMIT %s
        """
        query_parameters = [vector, self.namespace, *parameters, vector, limit]
        with self._connect() as connection:
            rows = connection.execute(statement, query_parameters).fetchall()

        results = []
        for row in rows:
            distance = float(row.pop("distance"))
            chunk_id = row.pop("id")
            document = row.pop("document")
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

    def collection_stats(self) -> dict[str, int]:
        """Return category counts, including general chunks in each category."""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT category, count(*) AS count
                FROM rag_chunks
                WHERE namespace = %s
                GROUP BY category
                """,
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
                "DELETE FROM rag_chunks WHERE namespace = %s",
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


vector_store = VectorStore()
