"""OCR and text extraction utility service for document figures.

Provides persistent caching of figure OCR text to enable accurate
cross-modal figure matching, caption resolution, and diagram discovery.
"""
import json
import logging
import threading
from pathlib import Path
from typing import Dict, Optional

from backend.services.storage_service import BASE_STORAGE_DIR, get_figure_file_path

logger = logging.getLogger("nalanda.figure_ocr")

_easyocr_reader = None
_ocr_lock = threading.Lock()


def get_ocr_reader():
    """Lazily initialize and return the EasyOCR reader singleton."""
    global _easyocr_reader
    if _easyocr_reader is None:
        with _ocr_lock:
            if _easyocr_reader is None:
                try:
                    import easyocr
                    _easyocr_reader = easyocr.Reader(["en"], gpu=False)
                except Exception as exc:
                    logger.warning("EasyOCR could not be initialized: %s", exc)
                    _easyocr_reader = False
    return _easyocr_reader if _easyocr_reader is not False else None


def get_figure_ocr_cache_path(user_id: str, folder_id: str, document_id: str) -> Path:
    """Return path to figure_ocr.json for the specified document."""
    return (
        BASE_STORAGE_DIR
        / "users"
        / user_id
        / "folders"
        / folder_id
        / "documents"
        / document_id
        / "extracted"
        / "figures"
        / "figure_ocr.json"
    )


def load_figure_ocr_cache(user_id: str, folder_id: str, document_id: str) -> Dict[str, str]:
    """Load cached figure OCR strings for a document."""
    cache_path = get_figure_ocr_cache_path(user_id, folder_id, document_id)
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
        except Exception as exc:
            logger.warning("Failed to read figure OCR cache %s: %s", cache_path, exc)
    return {}


def save_figure_ocr_cache(user_id: str, folder_id: str, document_id: str, cache: Dict[str, str]) -> None:
    """Save figure OCR cache atomically."""
    cache_path = get_figure_ocr_cache_path(user_id, folder_id, document_id)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(".tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
        tmp_path.replace(cache_path)
    except Exception as exc:
        logger.warning("Failed to save figure OCR cache %s: %s", cache_path, exc)
        if tmp_path.exists():
            tmp_path.unlink()


def get_figure_ocr_text(
    user_id: str,
    folder_id: str,
    document_id: str,
    filename: str,
    auto_extract: bool = True,
) -> str:
    """Retrieve OCR text for a figure, using disk cache or running EasyOCR lazily."""
    if not filename:
        return ""

    cache = load_figure_ocr_cache(user_id, folder_id, document_id)
    if filename in cache:
        return cache[filename]

    if not auto_extract:
        return ""

    # Run OCR on the figure file
    fig_file = get_figure_file_path(user_id, folder_id, document_id, filename)
    if not fig_file or not fig_file.exists():
        return ""

    reader = get_ocr_reader()
    if not reader:
        return ""

    try:
        lines = reader.readtext(str(fig_file), detail=0)
        extracted_text = " ".join(lines).strip()
    except Exception as exc:
        logger.warning("OCR extraction failed on %s: %s", filename, exc)
        extracted_text = ""

    # Update cache
    cache[filename] = extracted_text
    save_figure_ocr_cache(user_id, folder_id, document_id, cache)
    return extracted_text
