"""Tests for storage path sanitization, file validation, and database models."""
import os
import pytest
from pathlib import Path
from backend.services.storage_service import (
    sanitize_filename,
    validate_file,
    get_document_paths,
    MAX_FILE_SIZE_BYTES,
)


def test_sanitize_filename_traversal():
    """Ensure path traversal attacks are neutralized."""
    assert sanitize_filename("../../../etc/passwd") == "passwd"
    assert sanitize_filename("..\\..\\windows\\system32\\cmd.exe") == "cmd.exe"
    assert sanitize_filename("folder/subfolder/file.pdf") == "file.pdf"
    assert sanitize_filename("   .hidden.pdf   ") == "hidden.pdf"
    assert sanitize_filename("") == "unnamed_document"


def test_validate_file_extensions():
    """Verify supported and unsupported extensions."""
    # Supported: .pdf, .pptx, .heic, .heif, .png, .jpg, .jpeg
    for ext in [".pdf", ".pptx", ".heic", ".heif", ".png", ".jpg", ".jpeg"]:
        valid, ftype, err = validate_file(f"doc{ext}", 1024)
        assert valid is True
        assert err == ""
        if ext == ".pdf":
            assert ftype == "pdf"
        elif ext == ".pptx":
            assert ftype == "pptx"
        else:
            assert ftype == "image"

    # Unsupported formats (including previous or non-supported types)
    for bad_ext in [".webp", ".bmp", ".tif", ".tiff", ".exe", ".sh", ".py", ".docx", ".zip", ".html"]:
        valid, ftype, err = validate_file(f"test{bad_ext}", 1024)
        assert valid is False
        assert "Unsupported file extension" in err


def test_validate_mime_types():
    """Verify MIME type validation for supported formats."""
    # Valid explicit MIME types
    assert validate_file("test.pdf", 1024, "application/pdf")[0] is True
    assert validate_file("test.pptx", 1024, "application/vnd.openxmlformats-officedocument.presentationml.presentation")[0] is True
    assert validate_file("test.heic", 1024, "image/heic")[0] is True
    assert validate_file("test.heif", 1024, "image/heif")[0] is True
    assert validate_file("test.png", 1024, "image/png")[0] is True
    assert validate_file("test.jpg", 1024, "image/jpeg")[0] is True
    assert validate_file("test.jpeg", 1024, "image/jpeg")[0] is True

    # Generic octet-stream MIME types are accepted as fallback
    assert validate_file("test.heic", 1024, "application/octet-stream")[0] is True
    assert validate_file("test.pptx", 1024, "application/octet-stream")[0] is True

    # Invalid / mismatched MIME types are rejected
    valid, _, err = validate_file("test.pdf", 1024, "text/x-python")
    assert valid is False
    assert "Invalid MIME type" in err

    valid, _, err = validate_file("test.png", 1024, "application/pdf")
    assert valid is False
    assert "Invalid MIME type" in err


def test_validate_empty_and_invalid_files():
    """Verify empty files or missing extensions are rejected."""
    valid, _, err = validate_file("empty.pdf", 0)
    assert valid is False
    assert "empty" in err

    valid, _, err = validate_file("noextension", 1024)
    assert valid is False
    assert "no extension" in err

    valid, _, err = validate_file("", 1024)
    assert valid is False
    assert "required" in err


def test_validate_file_size_limit():
    """Verify file size enforcement."""
    valid, _, err = validate_file("test.pdf", MAX_FILE_SIZE_BYTES - 1)
    assert valid is True

    valid, _, err = validate_file("test.pdf", MAX_FILE_SIZE_BYTES + 100)
    assert valid is False
    assert "exceeds" in err


def test_get_document_paths():
    """Ensure document paths are organized according to Nalanda conventions."""
    paths = get_document_paths("user123", "folder456", "doc789", "lecture_notes.pdf")
    assert "users/user123/folders/folder456/documents/doc789/original" in str(paths["original_dir"]).replace("\\", "/")
    assert "users/user123/folders/folder456/documents/doc789/extracted" in str(paths["extracted_dir"]).replace("\\", "/")
    assert paths["original_file_path"].name == "lecture_notes.pdf"
    assert paths["original_dir"].exists()
    assert paths["extracted_dir"].exists()
