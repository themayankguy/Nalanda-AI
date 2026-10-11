"""Comprehensive test suite for multimodal local RAG:
- Intent classification (visual vs tabular vs conceptual)
- Sentence-level salience scoring (direct answer extraction)
- Multimodal chunking with figure & table bindings
- Figure retrieval endpoint security
- End-to-end multimodal search response
"""
import uuid
import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.database import Base, engine, SessionLocal
from backend.models import User, Folder, Document, DocumentExtraction
from backend.services.storage_service import BASE_STORAGE_DIR
from backend.services.NLP.semantic_search_service import (
    detect_query_intent,
    extract_direct_answer,
    sync_folder_index,
)
from backend.services.NLP.document_chunking_service import chunk_document_data
from backend.services.NLP.vector_index_service import get_embedding_model


@pytest.fixture(autouse=True)
def setup_db():
    Base.metadata.create_all(bind=engine)
    yield


@pytest.fixture
def client():
    return TestClient(app)


# --- 1. Intent Detection Tests ---

def test_intent_detection():
    """Verify intent classifier detects visual, tabular, and conceptual queries accurately."""
    # Visual intent
    assert detect_query_intent("Show me the diagram of the transformer") == "visual"
    assert detect_query_intent("What is the architecture flowchart of CNN?") == "visual"
    assert detect_query_intent("Display the schematic plot") == "visual"

    # Tabular intent
    assert detect_query_intent("Compare supervised versus unsupervised learning") == "tabular"
    assert detect_query_intent("Show the table of evaluation metrics") == "tabular"
    assert detect_query_intent("What are the differences between SGD and Adam?") == "tabular"

    # Conceptual intent
    assert detect_query_intent("What is gradient descent?") == "conceptual"
    assert detect_query_intent("Explain how backpropagation calculates gradients") == "conceptual"


# --- 2. Sentence Salience Scoring Tests ---

def test_extract_direct_answer():
    """Verify sentence salience scoring extracts the most relevant sentence answering the query."""
    text = (
        "Convolutional neural networks are specialized for grid data. "
        "Max pooling reduces the spatial dimensions of feature maps to achieve translation invariance. "
        "Fully connected layers perform final classification at the output."
    )
    model = get_embedding_model()

    # Query focused on pooling
    ans = extract_direct_answer("What does max pooling do in a neural network?", text, model=model)
    assert ans is not None
    assert "pooling" in ans.lower()

    # Query focused on classification
    ans2 = extract_direct_answer("Which layer performs classification?", text, model=model)
    assert ans2 is not None
    assert "classification" in ans2.lower()


# --- 3. Multimodal Chunking Tests ---

def test_multimodal_chunking_with_figures_and_tables(tmp_path):
    """Verify chunk_document_data binds figures and tables and indexes figure captions."""
    json_path = tmp_path / "test_doc.json"
    dummy_data = {
        "filename": "deep_learning.pdf",
        "file_type": "pdf",
        "total_pages": 1,
        "pages": [{
            "page": 1,
            "text": "Deep learning models learn hierarchical representations of input data.",
            "figures": [{
                "figure_index": 1,
                "page": 1,
                "filename": "deep_learning_p1_fig_1.png",
                "caption": "Figure 1: Deep neural network architecture with three hidden layers.",
                "source": "PyMuPDF",
                "width": 600,
                "height": 400,
            }],
            "tables": [{
                "table_index": 1,
                "page": 1,
                "html": "<table><tr><th>Layer</th><th>Units</th></tr><tr><td>Hidden 1</td><td>128</td></tr></table>",
                "markdown": "| Layer | Units |\n|---|---|\n| Hidden 1 | 128 |",
            }],
        }],
    }
    json_path.write_text(json.dumps(dummy_data), encoding="utf-8")

    chunks = chunk_document_data(
        document_id="doc_123",
        user_id="user_abc",
        folder_id="folder_xyz",
        source_filename="deep_learning.pdf",
        json_path=json_path,
    )

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk["page_number"] == 1
    assert len(chunk["figures"]) == 1
    assert chunk["figures"][0]["filename"] == "deep_learning_p1_fig_1.png"
    assert len(chunk["tables"]) == 1
    assert "Figure 1: Deep neural network architecture" in chunk["text"]
    assert "| Layer | Units |" in chunk["text"]


# --- 4. Figure Retrieval Endpoint Security Tests ---

def test_figure_endpoint_authorization(client, tmp_path):
    """Verify /api/documents/{id}/figures/{filename} enforces user isolation."""
    uid = uuid.uuid4().hex[:8]

    # Register User A
    res_a = client.post(
        "/api/auth/register",
        json={"email": f"alice_{uid}@nalanda.ai", "username": f"alice_{uid}", "password": "pass123"},
    )
    token_a = res_a.json()["access_token"]
    user_a_id = res_a.json()["user"]["id"]
    headers_a = {"Authorization": f"Bearer {token_a}"}

    # Register User B
    res_b = client.post(
        "/api/auth/register",
        json={"email": f"bob_{uid}@nalanda.ai", "username": f"bob_{uid}", "password": "pass123"},
    )
    token_b = res_b.json()["access_token"]
    headers_b = {"Authorization": f"Bearer {token_b}"}

    # User A creates a folder
    res_folder = client.post("/api/folders", json={"name": "Vision Course"}, headers=headers_a)
    folder_id = res_folder.json()["id"]

    # Create dummy document and figure in storage for User A
    doc_id = str(uuid.uuid4())
    fig_dir = BASE_STORAGE_DIR / "users" / user_a_id / "folders" / folder_id / "documents" / doc_id / "extracted" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_file = fig_dir / "sample_diagram.png"
    fig_file.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4")

    # Add document record to DB
    db = SessionLocal()
    doc = Document(
        id=doc_id,
        folder_id=folder_id,
        user_id=user_a_id,
        original_filename="vision.pdf",
        file_type="pdf",
        file_size_bytes=100,
        storage_path=str(fig_file),
        status="completed",
    )
    db.add(doc)
    db.commit()
    db.close()

    # User A can access figure
    res_fig_a = client.get(f"/api/documents/{doc_id}/figures/sample_diagram.png", headers=headers_a)
    assert res_fig_a.status_code == 200
    assert res_fig_a.headers["content-type"] == "image/png"

    # User B cannot access User A's figure
    res_fig_b = client.get(f"/api/documents/{doc_id}/figures/sample_diagram.png", headers=headers_b)
    assert res_fig_b.status_code == 404

    # Non-existent figure returns 404
    res_missing = client.get(f"/api/documents/{doc_id}/figures/non_existent.png", headers=headers_a)
    assert res_missing.status_code == 404


# --- 5. End-to-End Multimodal Search & Chat API Tests ---

def test_multimodal_semantic_search_api(client):
    """Test full multimodal semantic search endpoint returning direct answers and figures."""
    uid = uuid.uuid4().hex[:8]

    # Register user
    res_user = client.post(
        "/api/auth/register",
        json={"email": f"carol_{uid}@nalanda.ai", "username": f"carol_{uid}", "password": "pass123"},
    )
    token = res_user.json()["access_token"]
    user_id = res_user.json()["user"]["id"]
    headers = {"Authorization": f"Bearer {token}"}

    # Create folder
    res_f = client.post("/api/folders", json={"name": "Neural Networks"}, headers=headers)
    folder_id = res_f.json()["id"]

    # Create document
    doc_id = str(uuid.uuid4())
    db = SessionLocal()
    doc = Document(
        id=doc_id,
        folder_id=folder_id,
        user_id=user_id,
        original_filename="cnn_lecture.pdf",
        file_type="pdf",
        file_size_bytes=500,
        storage_path="dummy_path",
        status="completed",
    )
    db.add(doc)
    db.commit()

    # Create extraction with structured JSON containing figures and tables
    doc_extracted_dir = BASE_STORAGE_DIR / "users" / user_id / "folders" / folder_id / "documents" / doc_id / "extracted"
    (doc_extracted_dir / "json").mkdir(parents=True, exist_ok=True)
    json_path = doc_extracted_dir / "json" / "cnn_lecture.json"

    json_payload = {
        "filename": "cnn_lecture.pdf",
        "file_type": "pdf",
        "total_pages": 1,
        "pages": [{
            "page": 4,
            "text": (
                "## Convolutional Architecture\n\n"
                "Convolution layers apply spatial filter kernels over input images. "
                "Pooling layers perform spatial downsampling to reduce parameters."
            ),
            "figures": [{
                "figure_index": 1,
                "page": 4,
                "filename": "cnn_arch_p4_fig_1.png",
                "caption": "Figure 4: Complete Convolutional Neural Network flowchart and architecture.",
                "source": "PyMuPDF",
                "width": 800,
                "height": 500,
            }],
            "tables": [{
                "table_index": 1,
                "page": 4,
                "html": "<table><tr><th>Kernel</th><th>Stride</th></tr><tr><td>3x3</td><td>1</td></tr></table>",
                "markdown": "| Kernel | Stride |\n|---|---|\n| 3x3 | 1 |",
            }],
        }],
    }
    json_path.write_text(json.dumps(json_payload), encoding="utf-8")

    extraction = DocumentExtraction(
        document_id=doc_id,
        full_text=json_payload["pages"][0]["text"],
        json_storage_path=str(json_path),
        output_directory=str(doc_extracted_dir),
    )
    db.add(extraction)
    db.commit()
    db.close()

    # Search query requesting diagram
    search_res = client.post(
        "/api/search/semantic",
        json={"query": "Show diagram of convolutional architecture flowchart", "folder_id": folder_id, "top_k": 3},
        headers=headers,
    )
    assert search_res.status_code == 200
    data = search_res.json()

    assert data["folder_id"] == folder_id
    assert data["detected_intent"] == "visual"
    assert data["direct_answer"] is not None
    assert "convolution" in data["direct_answer"].lower() or "filter" in data["direct_answer"].lower()

    # Primary figure resolved
    assert data["primary_figure"] is not None
    assert data["primary_figure"]["filename"] == "cnn_arch_p4_fig_1.png"
    assert "/api/documents/" in data["primary_figure"]["figure_url"]

    # Results list contains figures and tables
    assert len(data["results"]) >= 1
    top_result = data["results"][0]
    assert top_result["page_number"] == 4
    assert len(top_result["figures"]) == 1
    assert len(top_result["tables"]) == 1
