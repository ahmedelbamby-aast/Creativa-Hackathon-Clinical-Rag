"""Full ingestion pipeline service."""

import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models import Chunk, Cluster, Document, ProcessingStatus
from src.services.embedding.embedder import embedder
from src.services.ingestion.chunker import smart_chunker
from src.services.ingestion.cleaner import extract_text
from src.services.ingestion.domain_classifier import domain_classifier
from src.services.ingestion.language_detector import language_detector
from src.services.ingestion.quality_filter import quality_filter

logger = logging.getLogger(__name__)


async def process_document(
    document_id: uuid.UUID,
    db: AsyncSession,
) -> tuple[int, int]:
    """Run the full ingestion pipeline on a document.

    Steps:
    1. Extract text from file
    2. Detect language
    3. Classify domain
    4. Smart chunking
    5. Quality filtering
    6. Embedding
    7. Save chunks to database
    8. Clustering (UMAP + HDBSCAN)
    9. Save clusters to database

    Args:
        document_id: ID of the document to process.
        db: Database session.

    Returns:
        Tuple of (chunk_count, cluster_count).

    Raises:
        ValueError: If document not found or processing fails.
    """
    # Get document
    result = await db.execute(select(Document).where(Document.id == document_id))
    document = result.scalar_one_or_none()

    if not document:
        raise ValueError(f"Document {document_id} not found")

    logger.info(f"[{document_id}] Step 0: Starting ingestion pipeline")

    # Update status to processing
    document.processing_status = ProcessingStatus.PROCESSING.value
    await db.commit()

    try:
        # Step 1: Extract text
        logger.info(f"[{document_id}] Step 1: Extracting text from {document.file_path}")
        text = extract_text(document.file_path, document.content_type)
        if not text:
            raise ValueError("Failed to extract text from document")
        logger.info(f"[{document_id}] Step 1: Extracted {len(text)} characters")

        # Step 2: Detect language
        logger.info(f"[{document_id}] Step 2: Detecting language")
        language = language_detector.detect_document(text)
        document.language = language
        logger.info(f"[{document_id}] Step 2: Detected language = {language}")

        # Step 3: Classify domain
        logger.info(f"[{document_id}] Step 3: Classifying domain")
        domain = domain_classifier.classify(text)
        document.domain = domain
        logger.info(f"[{document_id}] Step 3: Classified domain = {domain}")

        # Step 4: Smart chunking
        logger.info(f"[{document_id}] Step 4: Chunking text")
        chunks = smart_chunker.chunk(text)
        logger.info(f"[{document_id}] Step 4: Created {len(chunks)} chunks")

        # Step 5: Quality filtering
        logger.info(f"[{document_id}] Step 5: Filtering chunks by quality")
        scored_chunks = quality_filter.filter(chunks)
        logger.info(f"[{document_id}] Step 5: {len(scored_chunks)} chunks passed quality filter")

        # Step 6: Embedding
        chunk_texts = [c for c, _ in scored_chunks]
        if not chunk_texts:
            raise ValueError("No valid chunks after quality filtering")

        logger.info(f"[{document_id}] Step 6: Generating embeddings for {len(chunk_texts)} chunks")
        embeddings = embedder.embed_batch(chunk_texts)
        logger.info(f"[{document_id}] Step 6: Embeddings generated (dim={len(embeddings[0])})")

        # Step 7: Save chunks
        logger.info(f"[{document_id}] Step 7: Saving chunks to database")
        chunk_records = []
        for i, ((chunk_text, score), embedding) in enumerate(zip(scored_chunks, embeddings)):
            chunk = Chunk(
                document_id=document_id,
                chunk_order=i,
                text=chunk_text,
                embedding=embedding,
                language=language,
                domain=domain,
                quality_score=score,
            )
            chunk_records.append(chunk)
            db.add(chunk)

        await db.flush()
        logger.info(f"[{document_id}] Step 7: {len(chunk_records)} chunks saved")

        # Step 8: Clustering
        if len(embeddings) >= 5:
            logger.info(f"[{document_id}] Step 8: Running UMAP + HDBSCAN clustering")
            from src.services.clustering.clusterer import clusterer
            from src.services.clustering.dimensionality_reduction import dimensionality_reducer

            reduced = dimensionality_reducer.fit_transform(embeddings)
            labels, probabilities = clusterer.cluster(reduced.tolist())

            # Assign noise points
            labels = clusterer.assign_noise_points(reduced, labels)

            logger.info(f"[{document_id}] Step 8: Found {len(set(labels))} clusters (including noise)")

            # Step 9: Save clusters
            cluster_map = {}
            unique_labels = set(labels.tolist())

            for label in unique_labels:
                mask = labels == label
                chunk_ids = [chunk_records[i].id for i in range(len(labels)) if mask[i]]

                if len(chunk_ids) == 0:
                    continue

                # Calculate centroid
                cluster_embeddings = [embeddings[i] for i in range(len(labels)) if mask[i]]
                centroid = [sum(e[j] for e in cluster_embeddings) / len(cluster_embeddings)
                           for j in range(len(cluster_embeddings[0]))]

                # Determine dominant domain and language
                cluster_chunks = [chunk_records[i] for i in range(len(labels)) if mask[i]]
                domains = [c.domain for c in cluster_chunks]
                languages = [c.language for c in cluster_chunks]
                dominant_domain = max(set(domains), key=domains.count)
                dominant_language = max(set(languages), key=languages.count)

                cluster = Cluster(
                    document_id=document_id,
                    chunk_count=len(chunk_ids),
                    dominant_domain=dominant_domain,
                    dominant_language=dominant_language,
                    centroid=centroid,
                    is_noise=(label == -1),
                )
                db.add(cluster)
                await db.flush()

                # Update chunk cluster_ids
                cluster_map[label] = cluster.id
                for chunk in cluster_chunks:
                    chunk.cluster_id = cluster.id

            await db.flush()
            logger.info(f"[{document_id}] Step 9: Saved {len(cluster_map)} clusters")
        else:
            logger.info(f"[{document_id}] Step 8/9: Too few chunks for clustering, creating single cluster")
            # Not enough chunks for clustering, create single cluster
            all_embeddings = embeddings
            centroid = [sum(e[j] for e in all_embeddings) / len(all_embeddings)
                       for j in range(len(all_embeddings[0]))]

            cluster = Cluster(
                document_id=document_id,
                chunk_count=len(chunk_records),
                dominant_domain=domain,
                dominant_language=language,
                centroid=centroid,
                is_noise=False,
            )
            db.add(cluster)
            await db.flush()

            for chunk in chunk_records:
                chunk.cluster_id = cluster.id

            await db.flush()
            logger.info(f"[{document_id}] Step 9: Saved single cluster")

        # Update document status
        logger.info(f"[{document_id}] Final: Setting status to COMPLETED")
        document.processing_status = ProcessingStatus.COMPLETED.value
        await db.commit()

        # Count results
        chunk_count = len(chunk_records)
        cluster_count = len(cluster_map) if cluster_map else 1

        logger.info(f"[{document_id}] Pipeline completed: {chunk_count} chunks, {cluster_count} clusters")

        return chunk_count, cluster_count

    except Exception as e:
        logger.error(f"[{document_id}] Pipeline FAILED: {type(e).__name__}: {e}", exc_info=True)
        try:
            await db.rollback()
            from sqlalchemy import update
            await db.execute(
                update(Document)
                .where(Document.id == document_id)
                .values(
                    processing_status=ProcessingStatus.FAILED.value,
                    error_message=str(e),
                )
            )
            await db.commit()
            logger.info(f"[{document_id}] Status updated to FAILED")
        except Exception as inner_e:
            logger.error(f"[{document_id}] Failed to update status: {inner_e}")
        raise
