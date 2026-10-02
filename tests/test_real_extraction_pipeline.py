"""Integration test verifying actual end-to-end execution of nalanda_extractor.py."""
import uuid
from pathlib import Path
import fitz  # PyMuPDF
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.database import SessionLocal, Base, engine
from backend.models import Document, DocumentExtraction
from backend.services.nalanda_extraction_service import process_document_extraction


@pytest.fixture
def client():
    return TestClient(app)


def test_real_pdf_extraction_pipeline(client, tmp_path):
    """Create a real digital PDF, upload it, execute extraction, and verify internal storage and status."""
    # 1. Create a minimal 1-page digital PDF using PyMuPDF
    test_pdf_path = tmp_path / "lecture1.pdf"
    doc_fitz = fitz.open()
    page = doc_fitz.new_page()
    page.insert_text(
        fitz.Point(72, 72),
        "Introduction to Big Data\n\nBig data refers to data that is so large, fast or complex that it's difficult or impossible to process using traditional methods.\n\nKey Characteristics of Big Data:\n• Volume: Large amount of data\n• Velocity: High speed of data generation\n• Variety: Different types of data\n"
    )
    doc_fitz.save(str(test_pdf_path))
    doc_fitz.close()

    # 2. Register user & create folder
    uid = uuid.uuid4().hex[:8]
    res_user = client.post(
        "/api/auth/register",
        json={"email": f"prof_{uid}@nalanda.ai", "username": f"prof_{uid}", "password": "password123"},
    )
    assert res_user.status_code == 201
    token = res_user.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    res_folder = client.post("/api/folders", json={"name": "Big Data Systems"}, headers=headers)
    assert res_folder.status_code == 201
    folder_id = res_folder.json()["id"]

    # 3. Upload document
    with open(test_pdf_path, "rb") as f:
        res_upload = client.post(
            f"/api/folders/{folder_id}/documents/upload",
            files={"file": ("lecture1.pdf", f, "application/pdf")},
            headers=headers,
        )
    assert res_upload.status_code == 202
    doc_id = res_upload.json()["id"]

    # 4. Run the extraction service on this document
    process_document_extraction(doc_id)

    # 5. Check document status via API
    res_status = client.get(f"/api/documents/{doc_id}/status", headers=headers)
    assert res_status.status_code == 200
    status_payload = res_status.json()
    assert status_payload["status"] == "completed"
    assert status_payload["error_message"] is None

    # 6. Verify that document list endpoint returns ONLY metadata (no text)
    res_docs = client.get(f"/api/folders/{folder_id}/documents", headers=headers)
    assert res_docs.status_code == 200
    doc_meta = res_docs.json()[0]
    assert doc_meta["id"] == doc_id
    assert doc_meta["status"] == "completed"
    assert "full_text" not in doc_meta
    assert "text" not in doc_meta
    assert "markdown" not in doc_meta

    # 7. Verify internal backend database storage contains the extracted content
    db = SessionLocal()
    try:
        db_doc = db.query(Document).filter_by(id=doc_id).first()
        assert db_doc is not None
        assert db_doc.status == "completed"

        extraction = db.query(DocumentExtraction).filter_by(document_id=doc_id).first()
        assert extraction is not None
        assert "Introduction to Big Data" in extraction.full_text
        assert "Key Characteristics" in extraction.full_text
        assert extraction.total_pages == 1
        assert extraction.total_characters > 50
        assert Path(extraction.json_storage_path).exists()
    finally:
        db.close()
