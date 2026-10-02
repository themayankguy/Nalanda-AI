"""Unit tests for the Nalanda extraction integration service."""
import json
import uuid
import pytest
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.database import Base
from backend.models import User, Folder, Document, DocumentExtraction
from backend.services.nalanda_extraction_service import (
    parse_and_store_extraction_output,
    run_extraction_subprocess,
    EXTRACTOR_SCRIPT,
)


@pytest.fixture
def db_session(tmp_path):
    """Create a temporary in-memory or SQLite database session."""
    db_file = tmp_path / "test_extraction.db"
    engine = create_engine(f"sqlite:///{db_file}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    TestingSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSession()
    yield session
    session.close()


def test_extractor_script_exists():
    """Verify that nalanda_extractor.py exists at project root and is unchanged."""
    assert EXTRACTOR_SCRIPT.exists()
    assert EXTRACTOR_SCRIPT.is_file()


def test_parse_and_store_extraction_output(db_session, tmp_path):
    """Test parsing outputs produced by nalanda_extractor.py and storing internally."""
    # 1. Setup mock user, folder, and document in DB
    user = User(id=str(uuid.uuid4()), email="test@nalanda.ai", username="extractor_user", password_hash="hash")
    folder = Folder(id=str(uuid.uuid4()), name="Machine Learning", user_id=user.id)
    doc = Document(
        id=str(uuid.uuid4()),
        folder_id=folder.id,
        user_id=user.id,
        original_filename="sample_unit.pdf",
        file_type="pdf",
        file_size_bytes=10240,
        storage_path=str(tmp_path / "sample_unit.pdf"),
        status="processing",
    )
    db_session.add_all([user, folder, doc])
    db_session.commit()

    # 2. Setup mock extraction directory matching real nalanda_extractor.py output structure
    output_dir = tmp_path / "extracted"
    (output_dir / "json").mkdir(parents=True)
    (output_dir / "markdown").mkdir(parents=True)
    (output_dir / "txt").mkdir(parents=True)
    (output_dir / "tables").mkdir(parents=True)

    stem = "sample_unit"
    sample_json = {
        "filename": "sample_unit.pdf",
        "file_type": "pdf",
        "file_sha256": "abcdef1234567890",
        "total_pages": 2,
        "pages": [
            {
                "page": 1,
                "method": "PyMuPDF",
                "ocr": False,
                "characters": 25,
                "processing_time": 0.05,
                "text": "Chapter 1: Neural Networks",
                "tables": [
                    {
                        "table_index": 1,
                        "source": "PyMuPDF",
                        "data": [["Layer", "Units"], ["Input", "128"]],
                        "markdown": "| Layer | Units |\n| --- | --- |\n| Input | 128 |",
                        "html": "<table><tr><th>Layer</th><th>Units</th></tr><tr><td>Input</td><td>128</td></tr></table>"
                    }
                ]
            },
            {
                "page": 2,
                "method": "PyMuPDF",
                "ocr": False,
                "characters": 22,
                "processing_time": 0.03,
                "text": "Backpropagation rules",
                "tables": []
            }
        ],
        "total_direct_time": 0.08,
        "total_ocr_time": 0.0,
        "total_processing_time": 0.09,
    }

    (output_dir / "json" / f"{stem}.json").write_text(json.dumps(sample_json), encoding="utf-8")
    (output_dir / "markdown" / f"{stem}.md").write_text("# Chapter 1: Neural Networks\n\nBackpropagation rules\n", encoding="utf-8")
    (output_dir / "txt" / f"{stem}.txt").write_text("Chapter 1: Neural Networks\n\nBackpropagation rules\n", encoding="utf-8")
    (output_dir / "tables" / f"{stem}_page_1_table_1.html").write_text("<table>test</table>", encoding="utf-8")
    (output_dir / "processing_summary.json").write_text(json.dumps({"files_processed": 1, "failures": []}), encoding="utf-8")

    # 3. Ingest output
    extraction = parse_and_store_extraction_output(
        db_session,
        doc,
        Path(doc.storage_path),
        output_dir,
    )
    db_session.commit()

    # 4. Verify internal storage
    assert extraction is not None
    assert extraction.document_id == doc.id
    assert "Chapter 1: Neural Networks" in extraction.full_text
    assert extraction.total_pages == 2
    assert extraction.total_tables == 1
    assert extraction.total_characters > 0
    assert str(output_dir) in extraction.output_directory
    assert len(json.loads(extraction.table_files_json)) == 1

    # Verify query from DB
    stored = db_session.query(DocumentExtraction).filter_by(document_id=doc.id).first()
    assert stored is not None
    assert stored.total_pages == 2


def test_missing_json_raises_error(db_session, tmp_path):
    """Verify that missing extractor outputs raise an appropriate error."""
    doc = Document(
        id=str(uuid.uuid4()),
        folder_id="f1",
        user_id="u1",
        original_filename="missing.pdf",
        file_type="pdf",
        file_size_bytes=100,
        storage_path=str(tmp_path / "missing.pdf"),
        status="processing",
    )
    empty_output_dir = tmp_path / "empty_extracted"
    empty_output_dir.mkdir()

    with pytest.raises(FileNotFoundError):
        parse_and_store_extraction_output(db_session, doc, Path(doc.storage_path), empty_output_dir)
