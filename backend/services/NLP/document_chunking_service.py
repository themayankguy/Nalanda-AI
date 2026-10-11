"""Document chunking service for Nalanda AI.

Splits extracted document text into semantically cohesive chunks adhering to:
- Paragraph, heading, and page boundaries.
- Configurable chunk size (approx 400-600 tokens / 1200-1800 chars) and overlap.
- Retention of rich source metadata (page numbers, section titles, document/folder/user IDs).
"""
import hashlib
import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger("nalanda.chunking")

# Constants for text splitting
DEFAULT_TARGET_CHUNK_CHARS = 1400  # ~450 tokens
DEFAULT_OVERLAP_CHARS = 250        # ~75 tokens
MIN_CHUNK_CHARS = 15


def compute_sha256(text: str) -> str:
    """Compute SHA-256 hash of a string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def detect_section_heading(text_block: str) -> Optional[str]:
    """Detect section heading or title in a text block."""
    for line in text_block.splitlines():
        line = line.strip()
        if not line:
            continue
        # Markdown headings (e.g., "# Title", "## Section")
        md_match = re.match(r"^#{1,4}\s+(.+)$", line)
        if md_match:
            return md_match.group(1).strip()
        # Numbered headings (e.g., "1. Introduction", "3.2 Histogram")
        num_match = re.match(r"^\d+(\.\d+)*\s+([A-Z][\w\s\-]{3,60})$", line)
        if num_match:
            return line[:60].strip()
        # All caps or short title line
        if len(line) < 60 and line.isupper() and len(line) > 3:
            return line.strip()
    return None


def split_text_into_paragraphs(text: str) -> List[str]:
    """Split text into distinct non-empty paragraphs."""
    if not text:
        return []
    # Split on two or more newlines
    paragraphs = re.split(r"\n\s*\n", text)
    cleaned = [p.strip() for p in paragraphs if p.strip()]
    return cleaned


def create_chunks_from_paragraphs(
    paragraphs: List[str],
    target_size: int = DEFAULT_TARGET_CHUNK_CHARS,
    overlap_size: int = DEFAULT_OVERLAP_CHARS,
) -> List[str]:
    """Combine paragraphs into chunks respecting target size and overlap."""
    if not paragraphs:
        return []

    chunks: List[str] = []
    current_chunk: List[str] = []
    current_length = 0

    for para in paragraphs:
        para_len = len(para)

        # If a single paragraph is larger than target_size, split by sentences or lines
        if para_len > target_size * 1.5:
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_length = 0

            # Subdivide long paragraph by sentences or newlines
            sub_parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+", para) if s.strip()]
            sub_accum: List[str] = []
            sub_len = 0
            for part in sub_parts:
                if sub_len + len(part) > target_size and sub_accum:
                    chunks.append(" ".join(sub_accum))
                    # Retain last part for overlap
                    sub_accum = [sub_accum[-1], part] if len(sub_accum[-1]) < overlap_size else [part]
                    sub_len = sum(len(x) for x in sub_accum)
                else:
                    sub_accum.append(part)
                    sub_len += len(part)
            if sub_accum:
                chunks.append(" ".join(sub_accum))
            continue

        if current_length + para_len > target_size and current_chunk:
            combined = "\n\n".join(current_chunk)
            chunks.append(combined)

            # Keep the last paragraph for overlap if it doesn't exceed overlap_size
            last_para = current_chunk[-1]
            if len(last_para) <= overlap_size:
                current_chunk = [last_para, para]
                current_length = len(last_para) + para_len
            else:
                current_chunk = [para]
                current_length = para_len
        else:
            current_chunk.append(para)
            current_length += para_len

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    # Filter out any tiny fragments
    return [c for c in chunks if len(c) >= MIN_CHUNK_CHARS]


def chunk_document_data(
    document_id: str,
    user_id: str,
    folder_id: str,
    source_filename: str,
    json_path: Optional[Path] = None,
    fallback_text: Optional[str] = None,
    target_size: int = DEFAULT_TARGET_CHUNK_CHARS,
    overlap_size: int = DEFAULT_OVERLAP_CHARS,
) -> List[Dict[str, Any]]:
    """Generate structured chunks with complete metadata from extractor JSON or fallback text.

    Each returned chunk dictionary contains:
    - chunk_id: Unique deterministic identifier
    - document_id: ID of parent document
    - user_id: Owner user ID
    - folder_id: Associated folder ID
    - source_filename: Name of the original file
    - page_number: Exact page number (1-indexed) or None
    - section_title: Extracted section/heading or None
    - chunk_index: Global chunk index in document
    - text_hash: SHA-256 hash of the chunk text
    - text: Internal text representation for embedding (hidden from frontend)
    """
    raw_pages: List[Dict[str, Any]] = []

    # Attempt to load structured pages from extractor JSON
    if json_path and json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict) and "pages" in data and isinstance(data["pages"], list):
                    raw_pages = data["pages"]
        except Exception as exc:
            logger.warning("Failed to parse JSON file %s for chunking: %s", json_path, exc)

    all_chunks: List[Dict[str, Any]] = []
    current_section: Optional[str] = None
    global_chunk_idx = 0

    if raw_pages:
        # Process page by page to guarantee accurate page citations
        for page_data in sorted(raw_pages, key=lambda p: p.get("page", 0)):
            page_num = page_data.get("page")
            page_text = page_data.get("text", "") or ""

            # Check if page contains any tables with markdown representation
            tables = page_data.get("tables", []) or []
            table_md_snippets = [t.get("markdown", "") for t in tables if t.get("markdown")]
            if table_md_snippets:
                page_text += "\n\n" + "\n\n".join(table_md_snippets)

            paragraphs = split_text_into_paragraphs(page_text)
            page_chunks = create_chunks_from_paragraphs(paragraphs, target_size, overlap_size)

            for chunk_str in page_chunks:
                detected_heading = detect_section_heading(chunk_str)
                if detected_heading:
                    current_section = detected_heading

                chunk_id = f"{document_id}_p{page_num}_{global_chunk_idx}"
                all_chunks.append({
                    "chunk_id": chunk_id,
                    "document_id": document_id,
                    "user_id": user_id,
                    "folder_id": folder_id,
                    "source_filename": source_filename,
                    "page_number": page_num,
                    "section_title": current_section,
                    "chunk_index": global_chunk_idx,
                    "text_hash": compute_sha256(chunk_str),
                    "text": chunk_str,
                })
                global_chunk_idx += 1
    elif fallback_text:
        # Fallback when JSON is unavailable
        paragraphs = split_text_into_paragraphs(fallback_text)
        text_chunks = create_chunks_from_paragraphs(paragraphs, target_size, overlap_size)
        for chunk_str in text_chunks:
            detected_heading = detect_section_heading(chunk_str)
            if detected_heading:
                current_section = detected_heading

            chunk_id = f"{document_id}_c{global_chunk_idx}"
            all_chunks.append({
                "chunk_id": chunk_id,
                "document_id": document_id,
                "user_id": user_id,
                "folder_id": folder_id,
                "source_filename": source_filename,
                "page_number": None,
                "section_title": current_section,
                "chunk_index": global_chunk_idx,
                "text_hash": compute_sha256(chunk_str),
                "text": chunk_str,
            })
            global_chunk_idx += 1

    return all_chunks
