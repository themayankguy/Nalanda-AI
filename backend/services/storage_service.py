"""Storage management service for Nalanda AI original documents and extraction directories."""
import os
import re
from pathlib import Path
from typing import Tuple

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BASE_STORAGE_DIR = PROJECT_ROOT / "storage"

# Supported formats matching nalanda_extractor.py
SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
}

MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB limit


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal and unsafe characters."""
    if not filename:
        return "unnamed_document"
    # Normalize both forward and backward slashes
    normalized = filename.strip().replace("\\", "/")
    # Take only the final component of any path
    clean = os.path.basename(normalized)
    # Replace dangerous or control characters
    clean = re.sub(r'[\x00-\x1f\x7f<>:"/\\|?*]', "_", clean)
    # Prevent leading dots or whitespace
    clean = clean.lstrip(". ")
    if not clean:
        clean = "document"
    return clean[:255]


def validate_file(filename: str, file_size: int) -> Tuple[bool, str, str]:
    """Validate file extension and size.
    
    Returns (is_valid, file_type, error_message).
    """
    safe_name = sanitize_filename(filename)
    ext = Path(safe_name).suffix.lower()

    if not ext or ext not in SUPPORTED_EXTENSIONS:
        supported_list = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        return False, "", f"Unsupported file extension '{ext}'. Supported types: {supported_list}"

    if file_size > MAX_FILE_SIZE_BYTES:
        max_mb = MAX_FILE_SIZE_BYTES // (1024 * 1024)
        return False, "", f"File size exceeds the {max_mb} MB limit."

    file_type = "pdf" if ext == ".pdf" else "image"
    return True, file_type, ""


def get_document_paths(user_id: str, folder_id: str, document_id: str, filename: str) -> dict:
    """Generate isolated storage paths for original document and extraction outputs."""
    safe_name = sanitize_filename(filename)
    doc_base = BASE_STORAGE_DIR / "users" / user_id / "folders" / folder_id / "documents" / document_id
    
    original_dir = doc_base / "original"
    extracted_dir = doc_base / "extracted"
    
    original_dir.mkdir(parents=True, exist_ok=True)
    extracted_dir.mkdir(parents=True, exist_ok=True)
    
    original_file_path = original_dir / safe_name
    
    return {
        "original_dir": original_dir,
        "extracted_dir": extracted_dir,
        "original_file_path": original_file_path,
        "safe_filename": safe_name,
    }
