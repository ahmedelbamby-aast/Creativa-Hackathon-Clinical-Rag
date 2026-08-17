# Creativa Diabetes

Creativa Diabetes is an early-stage document ingestion project for preparing diabetes-related reference material for downstream search and retrieval workflows. The current runnable pipeline extracts text from source documents, cleans common extraction artifacts, detects whether the content is Arabic, English, or mixed, splits it into overlapping chunks, assigns a simple quality score, and writes page-aware JSON records.

The repository includes a small corpus of public diabetes publications and two generated chunk files that demonstrate the expected output format.

## What the pipeline does

For each document, the pipeline performs the following steps:

1. Extracts text from PDF, DOCX, or plain-text files.
2. Preserves PDF page numbers in the extracted records.
3. Normalizes Unicode and whitespace, repairs line-break hyphenation, and removes repeated headers, footers, and standalone page numbers.
4. Detects Arabic, English, or mixed-language content using character distribution.
5. Splits content hierarchically at paragraph, line, sentence, clause, word, and character boundaries.
6. Adds configurable overlap while keeping chunks within the requested maximum size.
7. Scores and filters chunks using length, text density, and technical-content signals.
8. Saves the document metadata, cleaning statistics, parameters, and chunks as JSON.

The chunker also attempts to keep fenced code, display math, HTML tables, and Markdown tables intact.

## Repository structure

```text
.
├── chunking/
│   ├── main.py                         # Runnable command-line pipeline
│   ├── output/                         # Generated JSON examples
│   └── src/services/ingestion/
│       ├── cleaner.py                  # Extraction and text cleanup
│       ├── chunker.py                  # Hierarchical, overlapping chunker
│       ├── language_detector.py        # Arabic/English language detection
│       └── quality_filter.py           # Chunk scoring and filtering
├── data/rew_data/books/                # Source diabetes publications
└── example/services/                   # Reference code for a larger architecture
```

`example/services/` contains exploratory components for embedding, clustering, security, and database-backed ingestion. It is not a complete runnable application in the current repository because the surrounding models, settings, and some services are not included.

## Requirements

- Python 3.10 or newer (required by the development test plugins)
- [`pypdf`](https://pypi.org/project/pypdf/) for PDF input
- [`python-docx`](https://pypi.org/project/python-docx/) for DOCX input

Only the dependency for the selected input type is required. Plain-text processing uses the Python standard library.

## Setup

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/A7MED-SA/Creativa_Diabetes.git
cd Creativa_Diabetes
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Or on macOS/Linux:

```bash
source .venv/bin/activate
```

Install the complete project environment with [UV](https://docs.astral.sh/uv/):

```bash
uv sync --all-groups
```

On Windows, the equivalent bootstrap helper is:

```powershell
.\scripts\install.ps1
```

On macOS/Linux:

```bash
./scripts/install.sh
```

UV installs the runtime document readers and the development test dependencies
from `pyproject.toml` and `uv.lock`.

## Testing

Run the complete test suite with:

```bash
uv run pytest
```

Or use `.\scripts\test.ps1` on Windows and `./scripts/test.sh` on macOS/Linux.
The default configuration runs tests in parallel with `pytest-xdist`, randomizes
test order with `pytest-randomly`, reports failures immediately with
`pytest-instafail`, keeps output concise with `pytest-tldr`, enforces per-test
timeouts with `pytest-timeout`, and reports coverage with `pytest-cov`.

`pytest-installfail` is not a published PyPI package; the project uses
`pytest-instafail`, the established plugin for immediate failure output.

## Usage

Run the pipeline from the repository root and provide a source document:

```bash
python chunking/main.py "data/rew_data/books/IDF_Diabetes_Atlas_11th_Edition_2025_WEB.pdf"
```

The result is written to `chunking/output/<document-name>.chunks.json` by default.

If no document is supplied, the command processes the first PDF (alphabetically) in `data/rew_data/books/`:

```bash
python chunking/main.py
```

### Options

```text
--output-dir PATH          Directory for generated JSON files
--max-chunk-size INTEGER   Maximum characters per chunk (default: 500)
--overlap-size INTEGER     Overlap between adjacent chunks (default: 50)
--min-quality-score FLOAT  Minimum accepted quality score (default: 0.1)
```

Example with custom parameters:

```bash
python chunking/main.py "path/to/document.pdf" \
  --output-dir "output" \
  --max-chunk-size 800 \
  --overlap-size 80 \
  --min-quality-score 0.2
```

`overlap-size` must be smaller than `max-chunk-size`.

## Output format

Each generated file contains document-level metadata and a `chunks` array:

```json
{
  "file_name": "document.pdf",
  "content_type": "application/pdf",
  "extractor": "pypdf",
  "language": "en",
  "page_count": 42,
  "chunk_count": 315,
  "chunk_params": {
    "max_chunk_size": 500,
    "overlap_size": 50,
    "min_quality_score": 0.1
  },
  "chunks": [
    {
      "index": 0,
      "page_number": 1,
      "text": "Extracted and cleaned text...",
      "char_count": 486,
      "word_count": 73,
      "quality_score": 0.684,
      "language": "en"
    }
  ]
}
```

The file also records source size, total cleaned characters, cleaning statistics, and a UTC creation timestamp. `source_file` is stored as an absolute path from the machine that generated the output.

## Current limitations

- Image-only or scanned PDFs require OCR before this pipeline can process them.
- DOCX and TXT inputs are represented as a single page because those formats do not expose PDF-style pagination here.
- Language detection currently distinguishes only Arabic, English, and mixed Arabic/English text.
- Quality scoring is heuristic and should be tuned for a specific retrieval or evaluation workload.
- Protected semantic blocks that are themselves larger than the configured chunk limit may exceed the intended content budget after restoration.
- The files under `example/services/` require additional application modules and dependencies before they can be executed.

## Data and medical-use notice

The documents under `data/rew_data/books/` remain subject to their original publishers' terms and attribution requirements. Review those terms before redistributing the source files.

This repository prepares documents for technical experimentation. It does not provide medical advice, diagnosis, or treatment recommendations, and generated chunks should be validated against their original sources before use in a health-related system.

## License

No project license is currently included. Unless a license is added, the repository's code should not be assumed to grant reuse, modification, or redistribution rights.
