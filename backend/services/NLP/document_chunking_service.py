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
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger("nalanda.chunking")

# Constants for text splitting
DEFAULT_TARGET_CHUNK_CHARS = 1400  # ~400 tokens (allows complete multi-step sections)
DEFAULT_OVERLAP_CHARS = 150        # ~45 tokens
MIN_CHUNK_CHARS = 15


def compute_sha256(text: str) -> str:
    """Compute SHA-256 hash of a string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def detect_section_heading(text_block: str) -> Optional[str]:
    """Detect section heading or title in a text block with high precision."""
    candidates: List[Tuple[str, int]] = []
    for idx, line in enumerate(text_block.splitlines()):
        line = line.strip()
        if not line or len(line) < 3 or len(line) > 120:
            continue
        # Reject table rows, bullets, code, quotes, examples
        if "|" in line or line.startswith(("-", "•", "*", "+", ">", "Example", "grunt>", "$", "http")):
            continue
        # Reject formulas, equations, code brackets
        if any(ch in line for ch in ["=", "<", ">", "→", "{", "}", "[", "]", "\\"]):
            continue
        # If line has a colon or dash/hyphen with explanation, only consider title part before separator
        for sep in [": ", " - ", " – ", " — "]:
            if sep in line:
                parts = line.split(sep, 1)
                if len(parts[1].strip().split()) >= 2:
                    potential_title = parts[0].strip()
                    if potential_title and len(potential_title) >= 3:
                        line = potential_title
                        break

        words = line.split()
        if len(words) > 10:
            continue

        # Major all-caps section header at first line (e.g. 'DOCKER CONTAINER LIFECYCLE', 'HIVE ARCHITECTURE')
        if idx == 0 and line.isupper() and any(c.isalpha() for c in line) and 2 <= len(words) <= 8 and not line.endswith((".", ";", ",")):
            candidates.append((line.strip(), 7))
            continue

        # Major numbered headings (e.g. '8. HIVE JOB EXECUTION FLOW', '25. Data Processing Operators', '7. Execution Modes of Apache Pig')
        num_major = re.match(r"^(?:\d+[\.\)]|\d+(?:\.\d+)*\.?)\s+([A-Za-z][\w\s\-/:,\?]{2,80})$", line)
        if num_major and not line.endswith("."):
            title_part = num_major.group(1).strip()
            if (title_part.isupper() or
                re.search(r"\b(?:architecture|working|execution|flow|workflow|operators?|versus|\bvs\b|difference|comparison|mechanisms?|models?|components?|features?|types?|lifecycle|overview|introduction)\b", title_part, re.I)):
                candidates.append((line.strip(), 7))
                continue

        # Major structural headings (e.g. 'Working / Architecture of Apache Pig', 'Steps for Creating a Visualization', '9. Pig vs MapReduce')
        if (re.search(r"\b(?:architecture|working\s*/\s*architecture|working\s+of|versus|\bvs\b|difference|comparison|execution\s+modes?|execution\s+flow)\b", line, flags=re.I) or
            re.match(r"^(?:steps?|phases?|stages?|process|procedure|methods?|techniques?|workflow|lifecycle|components?|characteristics?|features?|types?|advantages?|disadvantages?|overview|introduction|modes?)\b", line, flags=re.I)) and len(words) <= 10:
            candidates.append((line.strip(), 6))
            continue

        # 1. Markdown headings (e.g. '# Title', '## Section')
        md_match = re.match(r"^#{1,4}\s+(.+)$", line)
        if md_match:
            candidates.append((md_match.group(1).strip(), 5))
            continue

        # 2. Numbered headings (e.g., '1. Introduction', '1. Need for Visualization', '3.7 Box Plot')
        num_match = re.match(r"^(?:\d+[\.\)]|\d+(?:\.\d+)*\.?)\s+([A-Za-z][\w\s\-/:,\?]{2,70})$", line)
        if num_match and not line.endswith("."):
            candidates.append((line.strip(), 5))
            continue

        # 3. Question titles (with or without '?', e.g. 'What is a Container', 'Why Pig is needed?')
        if (line.endswith("?") or re.match(r"^(?:what|why|how|when|where|which|who)\b", line, flags=re.I)) and len(words) <= 9:
            candidates.append((line.strip(), 4))
            continue

        # 4. Unit / Module prefix lines (e.g. 'UNIT 5 – CONTAINERIZATION & DOCKER')
        unit_match = re.match(r"^(?:UNIT|MODULE|CHAPTER|PART)\s+\d+[\s:–-]+(.+)$", line, flags=re.I)
        if unit_match and len(unit_match.group(1).strip().split()) <= 8:
            candidates.append((unit_match.group(1).strip(), 3))
            continue

        # 5. Standalone short acronyms (e.g. 'APDD', 'BDA') - low priority fallback
        if len(words) == 1 and len(line) <= 5 and line.isupper():
            candidates.append((line.strip(), 1))
            continue

        # 6. Short title-case or all-caps lines (e.g. 'APACHE PIG', 'Main Components of Apache Pig')
        if not line.endswith((".", ";", ",")):
            if line.isupper() and any(c.isalpha() for c in line) and len(line) > 3:
                candidates.append((line.strip(), 4))
                continue
            content_words = [w for w in words if w.lower() not in {"a", "an", "the", "in", "on", "of", "for", "to", "and", "is", "are"}]
            if content_words and len(content_words) >= 2:
                cap_content = [w for w in content_words if w[0].isupper()]
                if len(cap_content) >= len(content_words) * 0.7:
                    candidates.append((line.strip(), 4))
                    continue

    if not candidates:
        return None
    # Sort by priority score descending, preserving order among equal priority
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0]


def split_text_into_paragraphs(text: str) -> List[str]:
    """Split text into distinct non-empty paragraphs and heading blocks."""
    if not text:
        return []
    # Split on two or more newlines
    raw_blocks = re.split(r"\n\s*\n", text)
    cleaned: List[str] = []
    for block in raw_blocks:
        lines = block.splitlines()
        curr_lines = []
        for line in lines:
            line_str = line.strip()
            # Only split on high-confidence structural/numbered headings (avoiding table column headers)
            h = detect_section_heading(line_str)
            is_strong_heading = bool(h and (re.match(r"^(?:#{1,4}\s+|\d+[\.\)]|\d+(?:\.\d+)+)", line_str) or re.search(r"\b(vs|versus|difference|comparison|architecture|working)\b", line_str, re.I)))
            if curr_lines and is_strong_heading and not line_str.startswith(("-", "•", "*", "+")):
                cleaned.append("\n".join(curr_lines).strip())
                curr_lines = [line]
            else:
                curr_lines.append(line)
        if curr_lines:
            cleaned.append("\n".join(curr_lines).strip())
    return [p for p in cleaned if p.strip()]


def create_chunks_from_paragraphs(
    paragraphs: List[str],
    target_size: int = DEFAULT_TARGET_CHUNK_CHARS,
    overlap_size: int = DEFAULT_OVERLAP_CHARS,
) -> List[str]:
    """Combine paragraphs into chunks respecting target size, section boundaries, and overlap."""
    if not paragraphs:
        return []

    chunks: List[str] = []
    current_chunk: List[str] = []
    current_length = 0

    for para in paragraphs:
        para_len = len(para)
        para_heading = detect_section_heading(para)

        # Check if this paragraph is a numbered item continuing a list in the current chunk
        is_list_continuation = False
        if para_heading and current_chunk:
            is_num_heading = bool(re.match(r"^(?:\d+[\.\)]|\d+(?:\.\d+)+)", para.strip()))
            is_major_topic = bool(re.search(r"\b(vs|versus|difference|comparison|architecture|working|overview|introduction|modes?|execution|models?|components?|features?|applications?|techniques?|mechanisms?|types?|evolution|need)\b", para_heading, flags=re.I))
            if is_num_heading and not is_major_topic:
                all_nums = [int(m.group(1)) for m in re.finditer(r"(?:^|\n)\s*(\d+)[\.\)]", "\n".join(current_chunk))]
                curr_num_m = re.match(r"^(\d+)[\.\)]", para.strip())
                curr_num = int(curr_num_m.group(1)) if curr_num_m else None
                if curr_num is not None:
                    if all_nums:
                        last_num = all_nums[-1]
                        if 1 <= (curr_num - last_num) <= 2 and curr_num <= 20:
                            is_list_continuation = True
                    elif curr_num <= 2:
                        is_list_continuation = True

        # If this paragraph starts a brand new section heading and we already have content,
        # flush the previous chunk to ensure section isolation (unless continuing a list within target_size).
        # Never flush short banner titles (< 90 chars) into isolated empty chunks.
        if para_heading and not is_list_continuation and current_chunk and current_length >= 90:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = []
            current_length = 0

        # If a single paragraph is larger than target_size * 1.5, split by sentences or lines
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

        max_allowed_size = int(target_size * 1.5) if is_list_continuation else target_size
        if current_length + para_len > max_allowed_size and current_chunk:
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

            # Check if page contains any figures/diagrams with meaningful captions
            figures = page_data.get("figures", []) or []
            meaningful_captions = [
                f"Figure: {f.get('caption', '')}"
                for f in figures
                if f.get("caption") and not re.match(r"^figure\s+on\s+page\s+\d+$", f.get("caption", "").strip(), re.I)
            ]
            if meaningful_captions:
                page_text += "\n\n" + "\n\n".join(meaningful_captions)

            paragraphs = split_text_into_paragraphs(page_text)
            page_chunks = create_chunks_from_paragraphs(paragraphs, target_size, overlap_size)

            for chunk_str in page_chunks:
                detected_heading = detect_section_heading(chunk_str)
                if detected_heading:
                    is_decimal_sec = bool(re.match(r"^\d+\.\d+", detected_heading.strip()))
                    is_major_topic = bool(re.search(r"\b(vs|versus|difference|comparison|architecture|working|overview|introduction|modes?|execution|models?|components?|features?|applications?|techniques?|mechanisms?|types?|evolution|lifecycle|need|operators?|commands?|tables?|functions?|udf|big sql)\b", detected_heading, re.I))
                    is_step_syntax = bool(re.match(r"^\s*(?:Step|Phase|Stage)\s+\d+", detected_heading, re.I))

                    is_cont_step = False
                    if not is_decimal_sec and not is_major_topic:
                        if is_step_syntax and current_section and re.search(r"\b(steps?|workflow|flow|procedure|phases?|stages?|working)\b", current_section, re.I):
                            is_cont_step = True
                        elif re.match(r"^\s*[4-9][\.\)]\s+", detected_heading.strip()) and current_section and re.search(r"\b(steps?|procedure|phases?|stages?)\b", current_section, re.I):
                            is_cont_step = True

                    if is_cont_step and current_section and not re.match(r"^\d+[\.\)]", current_section):
                        base_sec = current_section.replace(" (Contd.)", "")
                        current_section = f"{base_sec} (Contd.)"
                    elif not is_cont_step:
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
                    "figures": figures,
                    "tables": tables,
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
