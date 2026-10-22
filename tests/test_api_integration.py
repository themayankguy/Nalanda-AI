import uuid
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.database import Base, engine


@pytest.fixture(autouse=True)
def setup_db():
    Base.metadata.create_all(bind=engine)
    yield


@pytest.fixture
def client():
    return TestClient(app)


def test_auth_and_user_isolation(client):
    """Test user registration, login, and authorization boundaries."""
    uid = uuid.uuid4().hex[:8]
    # 1. Register User A
    res_a = client.post(
        "/api/auth/register",
        json={"email": f"alice_{uid}@nalanda.ai", "username": f"alice_{uid}", "password": "password123"},
    )
    assert res_a.status_code == 201
    token_a = res_a.json()["access_token"]
    headers_a = {"Authorization": f"Bearer {token_a}"}

    # 2. Register User B
    res_b = client.post(
        "/api/auth/register",
        json={"email": f"bob_{uid}@nalanda.ai", "username": f"bob_{uid}", "password": "password123"},
    )
    assert res_b.status_code == 201
    token_b = res_b.json()["access_token"]
    headers_b = {"Authorization": f"Bearer {token_b}"}

    # 3. User A creates a folder
    res_folder_a = client.post(
        "/api/folders",
        json={"name": "Machine Learning"},
        headers=headers_a,
    )
    assert res_folder_a.status_code == 201
    folder_a_id = res_folder_a.json()["id"]

    # 4. User B cannot see User A's folder in list
    res_b_folders = client.get("/api/folders", headers=headers_b)
    assert res_b_folders.status_code == 200
    b_folder_ids = [f["id"] for f in res_b_folders.json()]
    assert folder_a_id not in b_folder_ids

    # 5. User B cannot access User A's folder directly
    res_b_access = client.get(f"/api/folders/{folder_a_id}", headers=headers_b)
    assert res_b_access.status_code == 404

    # 6. User B cannot upload documents into User A's folder
    dummy_file = ("notes.pdf", b"%PDF-1.4 dummy content", "application/pdf")
    res_b_upload = client.post(
        f"/api/folders/{folder_a_id}/documents/upload",
        files={"file": dummy_file},
        headers=headers_b,
    )
    assert res_b_upload.status_code == 404


def test_document_upload_and_metadata_isolation(client):
    """Test document upload, metadata responses, and ensure NO extracted content is leaked."""
    uid = uuid.uuid4().hex[:8]
    # Register user
    res_user = client.post(
        "/api/auth/register",
        json={"email": f"student_{uid}@nalanda.ai", "username": f"student_{uid}", "password": "password123"},
    )
    assert res_user.status_code == 201
    token = res_user.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # Create folder
    res_folder = client.post("/api/folders", json={"name": "Deep Learning"}, headers=headers)
    assert res_folder.status_code == 201
    folder_id = res_folder.json()["id"]

    # Test upload unsupported file
    bad_file = ("script.py", b"print('hello')", "text/x-python")
    res_bad = client.post(
        f"/api/folders/{folder_id}/documents/upload",
        files={"file": bad_file},
        headers=headers,
    )
    assert res_bad.status_code == 400
    assert "Unsupported file extension" in res_bad.json()["detail"]

    # Test upload supported file (PDF)
    pdf_content = b"%PDF-1.4 mock pdf content for ingestion testing"
    good_file = ("syllabus.pdf", pdf_content, "application/pdf")
    res_upload = client.post(
        f"/api/folders/{folder_id}/documents/upload",
        files={"file": good_file},
        headers=headers,
    )
    assert res_upload.status_code == 202
    upload_data = res_upload.json()
    assert upload_data["status"] == "pending"
    doc_id = upload_data["id"]

    # CRITICAL CHECK: Verify upload response contains NO extracted text or internal paths
    assert "full_text" not in upload_data
    assert "text" not in upload_data
    assert "markdown" not in upload_data
    assert "storage_path" not in upload_data

    # List documents in folder
    res_list = client.get(f"/api/folders/{folder_id}/documents", headers=headers)
    assert res_list.status_code == 200
    doc_list = res_list.json()
    assert len(doc_list) >= 1
    target_doc = next(d for d in doc_list if d["id"] == doc_id)
    assert target_doc["original_filename"] == "syllabus.pdf"
    assert target_doc["file_type"] == "pdf"
    assert target_doc["file_size_bytes"] == len(pdf_content)

    # CRITICAL CHECK: Verify list response contains NO extracted text or internal paths
    for item in doc_list:
        assert "full_text" not in item
        assert "text" not in item
        assert "markdown" not in item
        assert "storage_path" not in item
        assert "extracted" not in item

    # Check status endpoint
    res_status = client.get(f"/api/documents/{doc_id}/status", headers=headers)
    assert res_status.status_code == 200
    status_data = res_status.json()
    assert status_data["id"] == doc_id
    assert status_data["status"] in ["pending", "processing", "completed", "failed"]
    assert "full_text" not in status_data


def test_document_upload_supported_formats_and_validations(client):
    """Test uploads for .pptx, .heic, .heif, .png, .jpg, .jpeg and error validations."""
    uid = uuid.uuid4().hex[:8]
    res_user = client.post(
        "/api/auth/register",
        json={"email": f"formats_{uid}@nalanda.ai", "username": f"user_{uid}", "password": "password123"},
    )
    assert res_user.status_code == 201
    token = res_user.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    res_folder = client.post("/api/folders", json={"name": "Formats Folder"}, headers=headers)
    assert res_folder.status_code == 201
    folder_id = res_folder.json()["id"]

    # 1. PPTX upload
    pptx_bytes = b"PK\x03\x04mock_pptx_data"
    res_pptx = client.post(
        f"/api/folders/{folder_id}/documents/upload",
        files={"file": ("slides.pptx", pptx_bytes, "application/vnd.openxmlformats-officedocument.presentationml.presentation")},
        headers=headers,
    )
    assert res_pptx.status_code == 202
    pptx_id = res_pptx.json()["id"]

    # 2. HEIC and HEIF uploads (accepted inputs without decoding/OCR yet)
    heic_bytes = b"\x00\x00\x00\x18ftypheicmock"
    res_heic = client.post(
        f"/api/folders/{folder_id}/documents/upload",
        files={"file": ("photo.heic", heic_bytes, "image/heic")},
        headers=headers,
    )
    assert res_heic.status_code == 202

    heif_bytes = b"\x00\x00\x00\x18ftypmif1mock"
    res_heif = client.post(
        f"/api/folders/{folder_id}/documents/upload",
        files={"file": ("photo.heif", heif_bytes, "image/heif")},
        headers=headers,
    )
    assert res_heif.status_code == 202

    # 3. PNG, JPG, JPEG uploads
    png_bytes = b"\x89PNG\r\n\x1a\nmock_png_data"
    res_png = client.post(
        f"/api/folders/{folder_id}/documents/upload",
        files={"file": ("chart.png", png_bytes, "image/png")},
        headers=headers,
    )
    assert res_png.status_code == 202

    jpg_bytes = b"\xff\xd8\xffmock_jpg_data"
    res_jpg = client.post(
        f"/api/folders/{folder_id}/documents/upload",
        files={"file": ("diagram.jpg", jpg_bytes, "image/jpeg")},
        headers=headers,
    )
    assert res_jpg.status_code == 202

    jpeg_bytes = b"\xff\xd8\xffmock_jpeg_data"
    res_jpeg = client.post(
        f"/api/folders/{folder_id}/documents/upload",
        files={"file": ("figure.jpeg", jpeg_bytes, "image/jpeg")},
        headers=headers,
    )
    assert res_jpeg.status_code == 202

    # Verify all uploaded documents exist with correct file_type and file_size
    res_docs = client.get(f"/api/folders/{folder_id}/documents", headers=headers)
    assert res_docs.status_code == 200
    docs = {d["original_filename"]: d for d in res_docs.json()}
    assert docs["slides.pptx"]["file_type"] == "pptx"
    assert docs["slides.pptx"]["file_size_bytes"] == len(pptx_bytes)
    assert docs["photo.heic"]["file_type"] == "image"
    assert docs["photo.heif"]["file_type"] == "image"
    assert docs["chart.png"]["file_type"] == "image"
    assert docs["diagram.jpg"]["file_type"] == "image"
    assert docs["figure.jpeg"]["file_type"] == "image"

    # 4. Error validation: Empty file (0 bytes)
    res_empty = client.post(
        f"/api/folders/{folder_id}/documents/upload",
        files={"file": ("empty.png", b"", "image/png")},
        headers=headers,
    )
    assert res_empty.status_code == 400
    assert "empty" in res_empty.json()["detail"].lower()

    # 5. Error validation: Invalid / mismatched MIME type
    res_bad_mime = client.post(
        f"/api/folders/{folder_id}/documents/upload",
        files={"file": ("notes.pdf", b"%PDF-1.4 mock", "text/x-python")},
        headers=headers,
    )
    assert res_bad_mime.status_code == 400
    assert "Invalid MIME type" in res_bad_mime.json()["detail"]

    # 6. Error validation: Unsupported format (e.g. .webp, .docx)
    res_unsupported = client.post(
        f"/api/folders/{folder_id}/documents/upload",
        files={"file": ("notes.docx", b"mock docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
        headers=headers,
    )
    assert res_unsupported.status_code == 400
    assert "Unsupported file extension" in res_unsupported.json()["detail"]

