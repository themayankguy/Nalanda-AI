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
    # Supported
    for ext in [".pdf", ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"]:
        valid, ftype, err = validate_file(f"doc{ext}", 1024)
        assert valid is True
        assert err == ""
        if ext == ".pdf":
            assert ftype == "pdf"
        else:
            assert ftype == "image"

    # Unsupported
    for bad_ext in [".exe", ".sh", ".py", ".docx", ".zip", ".html"]:
        valid, ftype, err = validate_file(f"test{bad_ext}", 1024)
        assert valid is False
        assert "Unsupported file extension" in err


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
