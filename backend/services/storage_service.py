"""Storage management service for Nalanda AI original documents and extraction directories."""
import os
import re
from pathlib import Path
from typing import Tuple

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BASE_STORAGE_DIR = PROJECT_ROOT / "storage"

# Supported formats: .pptx, .heic, .heif, .png, .jpg, .jpeg, and preserved .pdf
SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".pptx",
    ".heic",
    ".heif",
    ".png",
    ".jpg",
    ".jpeg",
}

# Supported MIME types per extension
SUPPORTED_MIME_TYPES = {
    ".pdf": {
        "application/pdf",
        "application/x-pdf",
        "application/acrobat",
        "applications/vnd.pdf",
        "text/pdf",
        "text/x-pdf",
    },
    ".pptx": {
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "application/x-pptx",
        "application/vnd.ms-powerpoint",
        "application/vnd.openxmlformats-officedocument.presentationml",
        "application/zip",
        "application/x-zip-compressed",
    },
    ".heic": {
        "image/heic",
        "image/heif",
        "image/heic-sequence",
    },
    ".heif": {
        "image/heif",
        "image/heic",
        "image/heif-sequence",
    },
    ".png": {
        "image/png",
        "image/x-png",
    },
    ".jpg": {
        "image/jpeg",
        "image/pjpeg",
        "image/jpg",
    },
    ".jpeg": {
        "image/jpeg",
        "image/pjpeg",
        "image/jpg",
    },
}

GENERIC_MIME_TYPES = {
    "application/octet-stream",
    "binary/octet-stream",
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


def validate_file(
    filename: str,
    file_size: int,
    content_type: str | None = None,
) -> Tuple[bool, str, str]:
    """Validate file extension, size, and MIME content type.
    
    Returns (is_valid, file_type, error_message).
    """
    if not filename or not filename.strip():
        return False, "", "Filename is required."

    safe_name = sanitize_filename(filename)
    ext = Path(safe_name).suffix.lower()

    if not ext:
        return False, "", "File has no extension or is invalid."

    if ext not in SUPPORTED_EXTENSIONS:
        supported_list = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        return False, "", f"Unsupported file extension '{ext}'. Supported types: {supported_list}"

    if file_size <= 0:
        return False, "", "File is empty (0 bytes)."

    if file_size > MAX_FILE_SIZE_BYTES:
        max_mb = MAX_FILE_SIZE_BYTES // (1024 * 1024)
        return False, "", f"File size exceeds the {max_mb} MB limit."

    if content_type:
        clean_mime = content_type.split(";")[0].strip().lower()
        if clean_mime:
            valid_mimes = SUPPORTED_MIME_TYPES.get(ext, set())
            if clean_mime not in valid_mimes and clean_mime not in GENERIC_MIME_TYPES:
                return (
                    False,
                    "",
                    f"Invalid MIME type '{clean_mime}' for file extension '{ext}'.",
                )

    if ext == ".pdf":
        file_type = "pdf"
    elif ext == ".pptx":
        file_type = "pptx"
    else:
        file_type = "image"

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


def get_figure_file_path(user_id: str, folder_id: str, document_id: str, filename: str) -> Path | None:
    """Resolve and validate path for an extracted figure file."""
    safe_name = sanitize_filename(filename)
    fig_path = (
        BASE_STORAGE_DIR
        / "users"
        / user_id
        / "folders"
        / folder_id
        / "documents"
        / document_id
        / "extracted"
        / "figures"
        / safe_name
    )
    if fig_path.exists() and fig_path.is_file():
        return fig_path
    return None
