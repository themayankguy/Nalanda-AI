"""Integration service for running nalanda_extractor.py and ingesting extracted data.

CRITICAL CONSTRAINTS:
1. nalanda_extractor.py is invoked via subprocess without ANY modifications.
2. Extracted content (text, markdown, JSON, tables) is stored securely in the database
   and backend storage. It is NEVER exposed to users in the frontend or public APIs.
"""
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional

from sqlalchemy.orm import Session
from backend.database import SessionLocal
from backend.models import Document, DocumentExtraction

logger = logging.getLogger("nalanda.extraction_service")

# Path to nalanda_extractor.py at project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EXTRACTOR_SCRIPT = PROJECT_ROOT / "nalanda_extractor.py"

# Default timeout for subprocess execution (in seconds)
SUBPROCESS_TIMEOUT_SECONDS = 300


def run_extraction_subprocess(input_path: Path, output_dir: Path, timeout: int = SUBPROCESS_TIMEOUT_SECONDS) -> subprocess.CompletedProcess:
    """Run nalanda_extractor.py using subprocess with sys.executable.
    
    Uses list of arguments and shell=False for safe execution.
    """
    if not EXTRACTOR_SCRIPT.exists():
        raise FileNotFoundError(f"Extractor script not found at {EXTRACTOR_SCRIPT}")

    cmd = [
        sys.executable,
        str(EXTRACTOR_SCRIPT),
        str(input_path),
        "--output",
        str(output_dir),
    ]

    # Inherit current environment so all installed packages (paddle, fitz, etc.) are available
    env = os.environ.copy()

    logger.info("Executing extractor subprocess: %s", " ".join(cmd))
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
        check=False,
    )
    return result


def parse_and_store_extraction_output(
    db: Session,
    document: Document,
    original_path: Path,
    output_dir: Path,
) -> DocumentExtraction:
    """Parse generated outputs from nalanda_extractor.py and save to DocumentExtraction.
    
    Reads:
    - <output_dir>/json/<stem>.json
    - <output_dir>/markdown/<stem>.md
    - <output_dir>/txt/<stem>.txt
    - <output_dir>/processing_summary.json
    - <output_dir>/tables/*.html
    """
    stem = original_path.stem
    json_path = output_dir / "json" / f"{stem}.json"
    markdown_path = output_dir / "markdown" / f"{stem}.md"
    txt_path = output_dir / "txt" / f"{stem}.txt"
    summary_path = output_dir / "processing_summary.json"
    tables_dir = output_dir / "tables"

    # Verify primary JSON output exists
    if not json_path.exists():
        raise FileNotFoundError(f"Expected extractor JSON output not found: {json_path}")

    # Read JSON content safely
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
    except Exception as exc:
        raise ValueError(f"Failed to parse extractor JSON file: {exc}") from exc

    # Read Markdown content if available
    markdown_content = ""
    if markdown_path.exists():
        try:
            markdown_content = markdown_path.read_text(encoding="utf-8")
        except Exception as exc:
            logger.warning("Could not read markdown output: %s", exc)

    # Read TXT content if available
    full_text = ""
    if txt_path.exists():
        try:
            full_text = txt_path.read_text(encoding="utf-8")
        except Exception as exc:
            logger.warning("Could not read txt output: %s", exc)

    # If full_text is empty, assemble from ordered pages in JSON data
    if not full_text and "pages" in json_data:
        ordered_pages = sorted(json_data.get("pages", []), key=lambda p: p.get("page", 0))
        full_text = "\n\n".join(p.get("text", "") for p in ordered_pages if p.get("text"))

    # Collect table files
    table_files = []
    if tables_dir.exists():
        table_files = [str(p) for p in sorted(tables_dir.glob(f"{stem}_page_*_table_*.html"))]

    total_pages = json_data.get("total_pages", len(json_data.get("pages", [])))
    total_characters = len(full_text)
    total_tables = sum(len(p.get("tables", [])) for p in json_data.get("pages", []))

    # Clean metadata without storing sensitive local paths to frontend
    metadata = {
        "filename": json_data.get("filename", original_path.name),
        "file_type": json_data.get("file_type", document.file_type),
        "file_sha256": json_data.get("file_sha256"),
        "total_pages": total_pages,
        "total_characters": total_characters,
        "total_tables": total_tables,
        "total_direct_time": json_data.get("total_direct_time", 0.0),
        "total_ocr_time": json_data.get("total_ocr_time", 0.0),
        "total_processing_time": json_data.get("total_processing_time", 0.0),
    }

    # If extraction already exists for this document (e.g. on retry), update it
    extraction = (
        db.query(DocumentExtraction)
        .filter(DocumentExtraction.document_id == document.id)
        .first()
    )

    if extraction is None:
        extraction = DocumentExtraction(
            document_id=document.id,
            full_text=full_text,
            markdown_content=markdown_content,
            json_storage_path=str(json_path),
            output_directory=str(output_dir),
            table_files_json=json.dumps(table_files),
            metadata_json=json.dumps(metadata),
            total_pages=total_pages,
            total_characters=total_characters,
            total_tables=total_tables,
        )
        db.add(extraction)
    else:
        extraction.full_text = full_text
        extraction.markdown_content = markdown_content
        extraction.json_storage_path = str(json_path)
        extraction.output_directory = str(output_dir)
        extraction.table_files_json = json.dumps(table_files)
        extraction.metadata_json = json.dumps(metadata)
        extraction.total_pages = total_pages
        extraction.total_characters = total_characters
        extraction.total_tables = total_tables

    return extraction


def process_document_extraction(document_id: str) -> None:
    """Entry point for background extraction task.
    
    Manages its own database session to ensure safe execution in asynchronous background tasks.
    """
    db = SessionLocal()
    try:
        document = db.query(Document).filter(Document.id == document_id).first()
        if not document:
            logger.error("Document %s not found for extraction", document_id)
            return

        # Update status to processing
        document.status = "processing"
        document.error_message = None
        db.commit()

        original_file_path = Path(document.storage_path)
        if not original_file_path.exists():
            document.status = "failed"
            document.error_message = "Original document file was not found on disk."
            db.commit()
            return

        # Generate isolated extraction directory
        doc_extracted_dir = original_file_path.parent.parent / "extracted"
        doc_extracted_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Run extractor subprocess
            result = run_extraction_subprocess(original_file_path, doc_extracted_dir)

            if result.returncode != 0:
                logger.error(
                    "Extractor exited with code %s for document %s. stderr: %s",
                    result.returncode,
                    document_id,
                    result.stderr[:500],
                )
                document.status = "failed"
                # Keep error message generic and safe (no internal server filepaths exposed)
                document.error_message = "Document extraction failed during OCR/text parsing."
                db.commit()
                return

            # Check processing_summary.json for reported failures
            summary_path = doc_extracted_dir / "processing_summary.json"
            if summary_path.exists():
                try:
                    summary_data = json.loads(summary_path.read_text(encoding="utf-8"))
                    failures = summary_data.get("failures", [])
                    if failures:
                        logger.error("Processing summary recorded failure: %s", failures)
                        document.status = "failed"
                        document.error_message = "Document extraction failed during page processing."
                        db.commit()
                        return
                except Exception as exc:
                    logger.warning("Could not read processing_summary.json: %s", exc)

            # Ingest output files into database
            parse_and_store_extraction_output(db, document, original_file_path, doc_extracted_dir)

            # Success!
            document.status = "completed"
            document.error_message = None
            db.commit()
            logger.info("Extraction successfully completed for document %s", document_id)

        except subprocess.TimeoutExpired:
            logger.error("Extraction subprocess timed out for document %s", document_id)
            document.status = "failed"
            document.error_message = "Extraction timed out. The document may be too large or complex."
            db.commit()
        except Exception as exc:
            logger.exception("Unexpected error during document extraction for %s: %s", document_id, exc)
            document.status = "failed"
            document.error_message = "An unexpected error occurred during extraction processing."
            db.commit()

    finally:
        db.close()
