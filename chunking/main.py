"""Chunking pipeline: extract -> clean -> detect language -> chunk -> filter -> save JSON."""

import argparse
import json
import logging
import os
from datetime import datetime, timezone

from src.services.ingestion.chunker import SmartChunker
from src.services.ingestion.cleaner import (
    clean_pages,
    detect_content_type,
    extract_pages,
    get_extractor_label,
)
from src.services.ingestion.language_detector import language_detector
from src.services.ingestion.quality_filter import quality_filter

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "rew_data", "books")
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def build_chunk_records(
    scored_chunks: list[tuple[str, float]],
    language: str,
    page_number: int,
    start_order: int = 0,
) -> list[dict]:
    """Build JSON-safe chunk records with metadata.

    Args:
        scored_chunks: List of (chunk_text, quality_score) tuples.
        language: Detected document language.
        page_number: Source page number of these chunks.
        start_order: Starting index for chunk ordering.

    Returns:
        List of chunk dicts.
    """
    records = []
    for i, (chunk_text, score) in enumerate(scored_chunks):
        records.append({
            "index": start_order + i,
            "page_number": page_number,
            "text": chunk_text,
            "char_count": len(chunk_text),
            "word_count": len(chunk_text.split()),
            "quality_score": score,
            "language": language,
        })
    return records


def process_file(
    file_path: str,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    min_quality_score: float = 0.1,
    max_chunk_size: int = 500,
    overlap_size: int = 50,
) -> dict:
    """Run the full chunking pipeline on a document and save JSON output.

    Steps:
    1. Extract per-page text from file
    2. Clean pages (unicode, hyphenation, page numbers, boilerplate)
    3. Detect language
    4. Smart chunking (page by page, preserving page metadata)
    5. Quality filtering
    6. Save chunk records to JSON

    Args:
        file_path: Path to the document file.
        output_dir: Directory where the JSON output is saved.
        min_quality_score: Minimum quality score to keep a chunk.
        max_chunk_size: Maximum chunk size in characters.
        overlap_size: Overlap between consecutive chunks in characters.

    Returns:
        Output metadata dict describing the result.

    Raises:
        ValueError: If text extraction fails or no valid chunks remain.
    """
    file_name = os.path.basename(file_path)
    chunker = SmartChunker(max_chunk_size=max_chunk_size, overlap_size=overlap_size)

    # Step 1: Extract per-page text
    logger.info(f"[{file_name}] Step 1: Extracting per-page text")
    pages = extract_pages(file_path)
    if not pages:
        raise ValueError(f"Failed to extract text from {file_path}")
    logger.info(f"[{file_name}] Step 1: Extracted {len(pages)} pages")

    # Step 2: Clean pages
    logger.info(f"[{file_name}] Step 2: Cleaning pages")
    cleaned_pages, cleaning_stats = clean_pages(pages)
    logger.info(f"[{file_name}] Step 2: {len(cleaned_pages)} pages after cleaning "
                f"({cleaning_stats['chars_before']} -> {cleaning_stats['chars_after']} chars)")

    if not cleaned_pages:
        if all(not page["text"].strip() for page in pages):
            raise ValueError(
                f"Extracted pages contain no text (file may be scanned/image-based; OCR required): {file_path}"
            )
        raise ValueError("No text left after cleaning")

    # Step 3: Detect language
    full_text = "\n\n".join(page["text"] for page in cleaned_pages)
    logger.info(f"[{file_name}] Step 3: Detecting language")
    language = language_detector.detect_document(full_text)
    logger.info(f"[{file_name}] Step 3: Detected language = {language}")

    # Step 4+5: Chunk and filter page by page
    logger.info(f"[{file_name}] Step 4: Chunking pages "
                f"(max_chunk_size={max_chunk_size}, overlap_size={overlap_size})")
    chunk_records = []
    global_index = 0
    for page in cleaned_pages:
        page_chunks = chunker.chunk(page["text"])
        if not page_chunks:
            continue
        scored_chunks = quality_filter.filter(page_chunks, min_score=min_quality_score)
        records = build_chunk_records(
            scored_chunks,
            language=language,
            page_number=page["page_number"],
            start_order=global_index,
        )
        chunk_records.extend(records)
        global_index += len(records)

    logger.info(f"[{file_name}] Step 4+5: {len(chunk_records)} chunks passed quality filter")

    if not chunk_records:
        raise ValueError("No valid chunks after quality filtering")

    # Step 6: Build output and save JSON
    os.makedirs(output_dir, exist_ok=True)
    stem = os.path.splitext(file_name)[0]
    output_path = os.path.join(output_dir, f"{stem}.chunks.json")

    output = {
        "source_file": os.path.abspath(file_path),
        "file_name": file_name,
        "file_size_bytes": os.path.getsize(file_path),
        "content_type": detect_content_type(file_path),
        "extractor": get_extractor_label(file_path),
        "language": language,
        "page_count": len(cleaned_pages),
        "total_chars": cleaning_stats["chars_after"],
        "chunk_count": len(chunk_records),
        "chunk_params": {
            "max_chunk_size": max_chunk_size,
            "overlap_size": overlap_size,
            "min_quality_score": min_quality_score,
        },
        "cleaning": cleaning_stats,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "chunks": chunk_records,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    logger.info(f"[{file_name}] Step 6: Saved {len(chunk_records)} chunks to {output_path}")
    logger.info(f"[{file_name}] Pipeline completed")

    return output


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Chunk a document and save results to JSON.")
    parser.add_argument(
        "file_path",
        nargs="?",
        help="Path to the document to chunk (default: first PDF in data/rew_data/books).",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for JSON output (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--max-chunk-size",
        type=int,
        default=500,
        help="Maximum chunk size in characters (default: 500).",
    )
    parser.add_argument(
        "--overlap-size",
        type=int,
        default=50,
        help="Overlap between consecutive chunks in characters (default: 50).",
    )
    parser.add_argument(
        "--min-quality-score",
        type=float,
        default=0.1,
        help="Minimum quality score to keep a chunk (default: 0.1).",
    )
    args = parser.parse_args()

    # Default to the first PDF in the data folder if no file given
    file_path = args.file_path
    if not file_path:
        data_dir = os.path.normpath(DEFAULT_DATA_DIR)
        if not os.path.isdir(data_dir):
            raise SystemExit(f"Data directory not found: {data_dir}")
        pdfs = sorted(
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.lower().endswith(".pdf")
        )
        if not pdfs:
            raise SystemExit(f"No PDF files found in {data_dir}")
        file_path = pdfs[0]
        logger.info(f"No file specified, using first PDF: {file_path}")

    file_path = os.path.abspath(file_path)
    if not os.path.isfile(file_path):
        raise SystemExit(f"File not found: {file_path}")

    process_file(
        file_path,
        output_dir=args.output_dir,
        min_quality_score=args.min_quality_score,
        max_chunk_size=args.max_chunk_size,
        overlap_size=args.overlap_size,
    )


if __name__ == "__main__":
    main()
