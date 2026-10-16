"""Unit tests for ATML Automated Comprehensive Notes and Study Guide Generation.

Verifies:
- Multi-section study guide extraction from lecture slides
- Fact-anchor and hypernymic definition compilation with exact citations
- Comparative matrix table formatting
- Procedural workflow step extraction
- High-yield exam revision takeaway formulation
- Pydantic schema and API endpoint integrity
"""
import pytest
from backend.services.atml.notes_generator import NotesGenerator, NotesSection, NotesResponse


class TestNotesGeneratorExtraction:
    """Tests for notes concept and workflow extraction."""

    def test_extract_fact_anchors_in_notes(self):
        generator = NotesGenerator()
        chunks = [
            {
                "text": (
                    "Tableau is defined as an interactive data visualization software focused on business intelligence.\n"
                    "Data Visualization is the process of converting raw data into a visual representation."
                ),
                "document_id": "doc-1",
                "source_filename": "visualization.pdf",
                "page_number": 2,
                "section_title": "Overview",
            }
        ]
        anchors = generator._extract_fact_anchors(chunks)
        assert len(anchors) >= 1
        concepts = [a["concept"] for a in anchors]
        assert "Tableau" in concepts or "Data Visualization" in concepts
        for a in anchors:
            assert len(a["definition"]) > 10
            assert a["chunk"]["source_filename"] == "visualization.pdf"

    def test_extract_workflows_in_notes(self):
        generator = NotesGenerator()
        chunks = [
            {
                "text": (
                    "1. Understand the data schema and attribute types.\n"
                    "2. Select appropriate chart type based on cardinality.\n"
                    "3. Render visual display and verify clarity.\n"
                    "4. Publish dashboard for end-user interaction."
                ),
                "document_id": "doc-2",
                "source_filename": "workflow.pdf",
                "page_number": 5,
                "section_title": "Visualization Pipeline",
            }
        ]
        workflows = generator._extract_workflows(chunks)
        assert len(workflows) == 1
        wf = workflows[0]
        assert len(wf["steps"]) >= 3
        assert "Understand" in wf["steps"][0]


class TestNotesDataStructures:
    """Tests for NotesSection and NotesResponse data models."""

    def test_notes_section_dataclass(self):
        sec = NotesSection(
            id="sec-overview",
            title="Executive Overview",
            icon="📋",
            content_markdown="This unit covers Big Data Visualization.",
            citations=["lecture.pdf (Page 1)"],
        )
        d = sec.to_dict()
        assert d["id"] == "sec-overview"
        assert d["icon"] == "📋"
        assert "Big Data" in d["content_markdown"]
        assert len(d["citations"]) == 1

    def test_notes_response_dataclass(self):
        sec = NotesSection(
            id="sec-1",
            title="Overview",
            icon="📋",
            content_markdown="Summary text.",
            citations=["doc.pdf (p. 1)"],
        )
        res = NotesResponse(
            folder_id="f-123",
            folder_name="Big Data Analytics",
            generated_at="2026-10-16 12:00 UTC",
            documents_analyzed=["doc.pdf"],
            total_sections=1,
            markdown_content="# Study Guide\nSummary text.",
            sections=[sec],
        )
        d = res.to_dict()
        assert d["folder_id"] == "f-123"
        assert d["folder_name"] == "Big Data Analytics"
        assert d["total_sections"] == 1
        assert len(d["sections"]) == 1
        assert "# Study Guide" in d["markdown_content"]


class TestNotesAPIEndpoint:
    """Test API endpoint authorization and schema contract."""

    def test_notes_api_unauthorized_without_token(self):
        from fastapi.testclient import TestClient
        from backend.main import app

        client = TestClient(app)
        response = client.get("/api/folders/fake-id/notes")
        assert response.status_code in [401, 403, 404]

    def test_multi_document_notes_synthesis_includes_all_documents(self, monkeypatch):
        """Verify that when multiple documents are present, all appear in sections and citations."""
        from unittest.mock import MagicMock
        from backend.models import Folder, User, Document

        generator = NotesGenerator()

        mock_user = MagicMock(spec=User)
        mock_user.id = "user-123"

        mock_folder = MagicMock(spec=Folder)
        mock_folder.id = "folder-456"
        mock_folder.name = "Distributed Computing"
        mock_folder.user_id = "user-123"

        doc1 = MagicMock(spec=Document)
        doc1.id = "doc-1"
        doc1.original_filename = "Unit_1_HDFS.pdf"
        doc1.status = "completed"
        doc1.extraction = None

        doc2 = MagicMock(spec=Document)
        doc2.id = "doc-2"
        doc2.original_filename = "Unit_2_Spark.pdf"
        doc2.status = "completed"
        doc2.extraction = None

        mock_db = MagicMock()
        mock_query = mock_db.query.return_value
        # First query is for folder, second for completed documents
        mock_query.filter.return_value.first.return_value = mock_folder
        mock_query.filter.return_value.all.return_value = [doc1, doc2]

        sample_chunks = [
            {
                "document_id": "doc-1",
                "source_filename": "Unit_1_HDFS.pdf",
                "page_number": 1,
                "section_title": "HDFS Overview",
                "text": "HDFS is defined as a distributed file system designed to run on commodity hardware.",
            },
            {
                "document_id": "doc-2",
                "source_filename": "Unit_2_Spark.pdf",
                "page_number": 3,
                "section_title": "Spark RDDs",
                "text": "Apache Spark refers to a unified analytics engine for large-scale data processing.",
            },
        ]

        # Patch FolderVectorIndex to return sample_chunks
        class MockIndex:
            def __init__(self, *args, **kwargs):
                self.chunks = sample_chunks

        monkeypatch.setattr("backend.services.atml.notes_generator.FolderVectorIndex", MockIndex)

        res = generator.generate_notes_for_folder(mock_db, mock_user, "folder-456")

        assert len(res.documents_analyzed) == 2
        assert "Unit_1_HDFS.pdf" in res.documents_analyzed
        assert "Unit_2_Spark.pdf" in res.documents_analyzed

        # Verify that both documents are cited in the guide
        all_citations = []
        for s in res.sections:
            all_citations.extend(s.citations)

        has_doc1 = any("Unit_1_HDFS.pdf" in cit for cit in all_citations)
        has_doc2 = any("Unit_2_Spark.pdf" in cit for cit in all_citations)
        assert has_doc1, "Expected Unit_1_HDFS.pdf citations"
        assert has_doc2, "Expected Unit_2_Spark.pdf citations"

        # Verify markdown content mentions both documents
        assert "Unit_1_HDFS.pdf" in res.markdown_content
        assert "Unit_2_Spark.pdf" in res.markdown_content


class TestNotesPdfExport:
    def test_pdf_exporter_generates_valid_pdf_bytes(self):
        """Verify that NotesPdfExporter produces valid multi-page PDF bytes from markdown."""
        import fitz
        from backend.services.atml.pdf_exporter import NotesPdfExporter

        exporter = NotesPdfExporter()
        sample_md = """
# Comprehensive Study Guide: Big Data
## 1. Overview
This is a comprehensive study guide for testing PDF generation.

## 2. Core Concepts
### 2.1 HDFS Architecture
- **Formal Definition:** Distributed filesystem for commodity clusters.
- **Key Characteristics:** 128MB block size, 3x replication factor.
> Provenance: Unit 2 (Page 1)

## 3. Comparative Matrix
| System | Role | Execution |
| :--- | :--- | :--- |
| **HDFS** | Storage | Block replication |
| **Pig** | Processing | MapReduce / Tez |
"""
        pdf_bytes = exporter.export_markdown_to_pdf_bytes(sample_md, "Big Data")

        assert isinstance(pdf_bytes, bytes)
        assert len(pdf_bytes) > 1000
        assert pdf_bytes.startswith(b"%PDF")

        # Verify PyMuPDF can parse it cleanly
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        assert len(doc) >= 1
        page_text = doc[0].get_text()
        assert "Comprehensive Study Guide" in page_text
        assert "HDFS Architecture" in page_text

    def test_pdf_export_api_unauthorized(self):
        """Ensure /api/folders/{id}/notes/export-pdf rejects unauthenticated requests with 401."""
        from fastapi.testclient import TestClient
        from backend.main import app

        client = TestClient(app)
        response = client.post("/api/folders/some-folder-id/notes/export-pdf")
        assert response.status_code == 401


