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
