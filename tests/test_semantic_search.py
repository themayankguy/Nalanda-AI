"""Comprehensive test suite for local semantic search, chunking, and strict folder isolation."""
import json
import uuid
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.database import Base, engine, SessionLocal
from backend.models import User, Folder, Document, DocumentExtraction
from backend.services.document_chunking_service import (
    chunk_document_data,
    create_chunks_from_paragraphs,
    detect_section_heading,
)
from backend.services.vector_index_service import (
    FolderVectorIndex,
    get_embedding_model,
)
from backend.services.semantic_search_service import sync_folder_index


@pytest.fixture(autouse=True)
def setup_db():
    Base.metadata.create_all(bind=engine)
    yield


@pytest.fixture
def client():
    return TestClient(app)


# --- 1. Unit Tests: Chunking & Edge Cases ---

def test_chunking_short_and_long_paragraphs():
    """Verify chunking handles both short text and large paragraphs without data loss."""
    short_text = ["This is a short paragraph.", "Another brief paragraph."]
    short_chunks = create_chunks_from_paragraphs(short_text, target_size=500, overlap_size=50)
    assert len(short_chunks) == 1
    assert "This is a short paragraph." in short_chunks[0]

    # Large paragraph that exceeds target size
    long_para = "Sentence number " + ". Sentence number ".join(str(i) for i in range(100)) + "."
    long_chunks = create_chunks_from_paragraphs([long_para], target_size=200, overlap_size=40)
    assert len(long_chunks) > 1
    for c in long_chunks:
        assert len(c) > 0


def test_chunking_heading_detection():
    """Verify heading and section detection in markdown and plain text."""
    assert detect_section_heading("## 3.7 Box Plot\nA box plot is used...") == "3.7 Box Plot"
    assert detect_section_heading("# Data Visualization Overview") == "Data Visualization Overview"
    assert detect_section_heading("Just normal text without any heading.") is None


def test_chunking_empty_and_malformed_json(tmp_path):
    """Verify chunking handles empty, non-existent, or malformed JSON without crashing."""
    # 1. Non-existent JSON
    chunks_missing = chunk_document_data(
        document_id="doc_missing",
        user_id="u1",
        folder_id="f1",
        source_filename="missing.pdf",
        json_path=tmp_path / "non_existent.json",
        fallback_text="Fallback content for missing file.",
    )
    assert len(chunks_missing) == 1
    assert "Fallback content" in chunks_missing[0]["text"]

    # 2. Malformed JSON
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("invalid { json content [", encoding="utf-8")
    chunks_bad = chunk_document_data(
        document_id="doc_bad",
        user_id="u1",
        folder_id="f1",
        source_filename="bad.pdf",
        json_path=bad_json,
        fallback_text="Safe fallback text.",
    )
    assert len(chunks_bad) == 1
    assert "Safe fallback" in chunks_bad[0]["text"]


# --- 2. Unit Tests: Vector Index & Local Embeddings ---

def test_local_embedding_model_shape_and_norm():
    """Verify all-MiniLM-L6-v2 loads locally and outputs normalized 384-dimensional vectors."""
    model = get_embedding_model()
    vectors = model.encode(["Hello Nalanda AI", "Statistical data visualization"], normalize_embeddings=True)
    assert vectors.shape == (2, 384)
    # Check unit norm
    import numpy as np
    norm_0 = np.linalg.norm(vectors[0])
    assert abs(norm_0 - 1.0) < 1e-4


def test_vector_index_persistence_and_update(tmp_path):
    """Verify vector index persists to disk, supports updates, and avoids redundant re-indexing."""
    index_dir = tmp_path / "test_vec_index"
    index = FolderVectorIndex(index_dir)

    sample_chunks = [
        {
            "chunk_id": "c1",
            "document_id": "d1",
            "user_id": "u1",
            "folder_id": "f1",
            "source_filename": "stats.pdf",
            "page_number": 5,
            "section_title": "Box Plot",
            "chunk_index": 0,
            "text_hash": "hash_c1",
            "text": "Box plots display the distribution and identify statistical outliers.",
        },
        {
            "chunk_id": "c2",
            "document_id": "d1",
            "user_id": "u1",
            "folder_id": "f1",
            "source_filename": "stats.pdf",
            "page_number": 4,
            "section_title": "Pie Chart",
            "chunk_index": 1,
            "text_hash": "hash_c2",
            "text": "Pie charts represent expenditure and proportions of company expenses.",
        },
    ]

    added = index.add_document_chunks(sample_chunks)
    assert added == 2
    assert len(index.chunks) == 2

    # Query: natural language for outliers
    results = index.search("Which chart helps identify outliers in data?", top_k=2)
    assert len(results) == 2
    top_match, score = results[0]
    assert top_match["section_title"] == "Box Plot"
    assert score > 0.4

    # Query: natural language for company expenses
    results_expenses = index.search("How can I compare company expenses?", top_k=2)
    top_match_exp, score_exp = results_expenses[0]
    assert top_match_exp["section_title"] == "Pie Chart"
    assert score_exp > 0.4

    # Verify disk persistence across new instance reload
    reloaded_index = FolderVectorIndex(index_dir)
    assert len(reloaded_index.chunks) == 2
    results_reloaded = reloaded_index.search("identify outliers", top_k=1)
    assert results_reloaded[0][0]["section_title"] == "Box Plot"

    # Test document deletion
    reloaded_index.remove_document("d1")
    assert len(reloaded_index.chunks) == 0
    assert len(reloaded_index.search("identify outliers")) == 0


# --- 3. Integration Tests: API & Strict User/Folder Isolation ---

def test_semantic_search_api_and_strict_isolation(client, tmp_path):
    """Test full semantic search API with strict multi-user and multi-folder isolation."""
    db = SessionLocal()

    # Create User A with Folder A1 (Machine Learning) and Folder A2 (History)
    uid_a = uuid.uuid4().hex[:8]
    user_a = User(id=str(uuid.uuid4()), email=f"user_a_{uid_a}@nalanda.ai", username=f"user_a_{uid_a}", password_hash="hash")
    folder_a1 = Folder(id=str(uuid.uuid4()), name="Machine Learning", user_id=user_a.id)
    folder_a2 = Folder(id=str(uuid.uuid4()), name="World History", user_id=user_a.id)

    # Create User B with Folder B1 (Deep Learning)
    uid_b = uuid.uuid4().hex[:8]
    user_b = User(id=str(uuid.uuid4()), email=f"user_b_{uid_b}@nalanda.ai", username=f"user_b_{uid_b}", password_hash="hash")
    folder_b1 = Folder(id=str(uuid.uuid4()), name="Deep Learning", user_id=user_b.id)

    db.add_all([user_a, folder_a1, folder_a2, user_b, folder_b1])
    db.commit()

    # Document 1 in User A's Folder A1: Statistics & Box plots
    doc_a1 = Document(
        id=str(uuid.uuid4()),
        folder_id=folder_a1.id,
        user_id=user_a.id,
        original_filename="stats_unit.pdf",
        file_type="pdf",
        file_size_bytes=1024,
        storage_path=str(tmp_path / "stats.pdf"),
        status="completed",
    )
    ext_a1 = DocumentExtraction(
        id=str(uuid.uuid4()),
        document_id=doc_a1.id,
        full_text="3.7 Box Plot: A box plot is used to represent numerical data distributions and identify outliers. Median, quartiles, and outliers are clearly visualized.",
        total_pages=1,
    )

    # Document 2 in User A's Folder A2: History document
    doc_a2 = Document(
        id=str(uuid.uuid4()),
        folder_id=folder_a2.id,
        user_id=user_a.id,
        original_filename="roman_empire.pdf",
        file_type="pdf",
        file_size_bytes=1024,
        storage_path=str(tmp_path / "history.pdf"),
        status="completed",
    )
    ext_a2 = DocumentExtraction(
        id=str(uuid.uuid4()),
        document_id=doc_a2.id,
        full_text="The Roman Republic expanded rapidly across the Mediterranean basin and developed complex civil administration.",
        total_pages=1,
    )

    # Document 3 in User B's Folder B1: Neural networks
    doc_b1 = Document(
        id=str(uuid.uuid4()),
        folder_id=folder_b1.id,
        user_id=user_b.id,
        original_filename="cnn_notes.pdf",
        file_type="pdf",
        file_size_bytes=1024,
        storage_path=str(tmp_path / "cnn.pdf"),
        status="completed",
    )
    ext_b1 = DocumentExtraction(
        id=str(uuid.uuid4()),
        document_id=doc_b1.id,
        full_text="Convolutional layers extract spatial feature maps from input images using kernel filters and pooling.",
        total_pages=1,
    )

    user_a_id = user_a.id
    user_b_id = user_b.id
    folder_a1_id = folder_a1.id
    folder_a2_id = folder_a2.id
    folder_b1_id = folder_b1.id

    db.add_all([doc_a1, ext_a1, doc_a2, ext_a2, doc_b1, ext_b1])
    db.commit()
    db.close()

    # Generate session tokens for API authentication
    from backend.auth import create_session_token
    db_session = SessionLocal()
    token_a = create_session_token(db_session, user_a_id).token
    token_b = create_session_token(db_session, user_b_id).token
    db_session.close()

    headers_a = {"Authorization": f"Bearer {token_a}"}
    headers_b = {"Authorization": f"Bearer {token_b}"}

    # --- Test 1: User A searches Folder A1 for outliers ---
    res_a1 = client.post(
        "/api/search/semantic",
        json={"query": "Which graph helps identify outliers?", "folder_id": folder_a1_id, "top_k": 3},
        headers=headers_a,
    )
    assert res_a1.status_code == 200
    data_a1 = res_a1.json()
    assert data_a1["folder_id"] == folder_a1_id
    assert len(data_a1["results"]) >= 1
    top_res = data_a1["results"][0]
    assert top_res["source_filename"] == "stats_unit.pdf"
    assert top_res["relevance_score"] > 0.4
    # CRITICAL CHECK: Verify response contains NO raw extracted text or internal paths
    assert "full_text" not in top_res
    assert "text" not in top_res
    assert "content" not in top_res
    assert "storage_path" not in top_res

    # --- Test 2: User A querying Folder A1 CANNOT retrieve content from User A's Folder A2 ---
    res_history = client.post(
        "/api/search/semantic",
        json={"query": "Tell me about the Roman Republic", "folder_id": folder_a1_id, "top_k": 3},
        headers=headers_a,
    )
    assert res_history.status_code == 200
    for item in res_history.json()["results"]:
        assert item["source_filename"] != "roman_empire.pdf"

    # --- Test 3: User A querying Folder A1 CANNOT retrieve content from User B's Folder B1 ---
    res_cnn = client.post(
        "/api/search/semantic",
        json={"query": "Convolutional feature maps and pooling", "folder_id": folder_a1_id, "top_k": 3},
        headers=headers_a,
    )
    assert res_cnn.status_code == 200
    for item in res_cnn.json()["results"]:
        assert item["source_filename"] != "cnn_notes.pdf"

    # --- Test 4: Supplying another user's folder ID is rejected with 404 ---
    res_unauth = client.post(
        "/api/search/semantic",
        json={"query": "Identify outliers", "folder_id": folder_b1_id, "top_k": 3},
        headers=headers_a,  # User A trying to access User B's folder
    )
    assert res_unauth.status_code == 404
    assert "Target folder not found" in res_unauth.json()["detail"]

    # --- Test 5: Omitting folder_id is rejected by validation ---
    res_no_folder = client.post(
        "/api/search/semantic",
        json={"query": "Identify outliers", "folder_id": "", "top_k": 3},
        headers=headers_a,
    )
    assert res_no_folder.status_code == 422 or res_no_folder.status_code == 400

    # --- Test 6: Empty query is rejected ---
    res_empty_q = client.post(
        "/api/search/semantic",
        json={"query": "   ", "folder_id": folder_a1_id, "top_k": 3},
        headers=headers_a,
    )
    assert res_empty_q.status_code == 400
