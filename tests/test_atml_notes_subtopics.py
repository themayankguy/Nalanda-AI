"""
Comprehensive unit and integration tests for ATML Notes Generator subtopic extraction,
validation, source grounding, and exam-readiness.
"""

import pytest
from typing import List, Dict, Any
from backend.services.atml.notes_generator import NotesGenerator, SubtopicUnit


class TestNotesSubtopicExtractionAndValidation:
    """Tests for discrete subtopic extraction, deduplication, and source-grounding."""

    def setup_method(self):
        self.generator = NotesGenerator()

    def test_distinct_definitions_docker_vs_docker_engine(self):
        """Docker and Docker Engine must have distinct, technically accurate definitions."""
        pages = [
            {
                "page": 5,
                "text": """
                Docker
                Docker is a containerization platform used to package applications and their dependencies into containers.
                Features of Docker
                - Portable across environments
                - Lightweight resource footprint

                Docker Engine
                Docker Engine is the core technology responsible for building and running Docker containers. It follows a client-server architecture.
                Components of Docker Engine
                - Docker Daemon (dockerd)
                - Docker CLI
                - REST API
                """,
                "tables": [],
            },
            {
                "page": 6,
                "text": """
                Docker Container
                A Docker Container is a lightweight, runnable instance of a Docker image.
                - Shares host kernel
                - Isolated process space
                """,
                "tables": [],
            }
        ]

        subtopics = self.generator._extract_document_subtopics("APDD UNIT 5.pdf", pages)
        subtopic_map = {s.title.lower(): s for s in subtopics}

        assert "docker" in subtopic_map
        assert "docker engine" in subtopic_map
        assert "docker container" in subtopic_map

        docker_def = subtopic_map["docker"].definition.lower()
        engine_def = subtopic_map["docker engine"].definition.lower()
        container_def = subtopic_map["docker container"].definition.lower()

        # Definitions must be distinct
        assert docker_def != engine_def
        assert engine_def != container_def
        assert docker_def != container_def

        # No neural computation hallucination
        assert "neural computation" not in docker_def
        assert "neural computation" not in engine_def
        assert "neural computation" not in container_def

        # Correct technical grounding
        assert "containerization platform" in docker_def or "package" in docker_def
        assert "engine" in engine_def or "client-server" in engine_def or "building" in engine_def or "technology" in engine_def
        assert "runnable instance" in container_def or "instance" in container_def or "lightweight" in container_def

    def test_no_generic_exam_traps_or_invariant_filler(self):
        """Generated exam points must omit generic buzzwords, invariants, and filler."""
        pages = [
            {
                "page": 1,
                "text": """
                Containerization
                Containerization is an approach in software engineering to bundle code and dependencies.
                - Solves dependency hell
                - Ensures portability
                """,
                "tables": []
            }
        ]

        subtopics = self.generator._extract_document_subtopics("Course_Module.pdf", pages)
        assert len(subtopics) >= 1

        for sub in subtopics:
            rendered = self.generator._render_subtopic_markdown(sub, 1, 1)

            # Assert complete absence of previous filler phrases
            assert "buzzword" not in rendered.lower()
            assert "isolated buzzword" not in rendered.lower()
            assert "maintains determinism under standard operating constraints" not in rendered.lower()
            assert "specialized neural computation layer" not in rendered.lower()
            assert "engineering trade-offs" not in rendered.lower()

    def test_page_reference_bounds_and_verification(self):
        """Page citations must accurately reflect source pages and never fabricate invalid ranges."""
        pages = [
            {"page": 3, "text": "Virtual Machines vs Containers\n- Containers share OS kernel.", "tables": []},
            {"page": 4, "text": "Reusing Container\n- Reusing a container means using an existing container instance.", "tables": []},
        ]

        subtopics = self.generator._extract_document_subtopics("Tech_Overview.pdf", pages)
        for s in subtopics:
            assert 1 <= s.page_start <= 2
            assert 1 <= s.page_end <= 2
            assert s.page_start <= s.page_end

            rendered = self.generator._render_subtopic_markdown(s, 1, 1)
            assert f"**Source:** Tech_Overview.pdf, Page" in rendered

    def test_duplicate_definition_sanitization(self):
        """Validation pass must detect duplicate definitions across different topics and sanitize them."""
        # Manually create two SubtopicUnit instances with identical definitions
        sub1 = SubtopicUnit(
            title="Topic Alpha",
            doc_name="Notes.pdf",
            page_start=1,
            page_end=1,
            definition="A standardized protocol for managing state transitions.",
            explanation="Detailed explanation of Alpha.",
            key_points=["Key point A1", "Key point A2"],
            exam_points=["Exam point Alpha"]
        )
        sub2 = SubtopicUnit(
            title="Topic Beta",
            doc_name="Notes.pdf",
            page_start=2,
            page_end=2,
            definition="A standardized protocol for managing state transitions.",
            explanation="Detailed explanation of Beta.",
            key_points=["Key point B1", "Key point B2"],
            exam_points=["Exam point Beta"]
        )

        validated = self.generator._validate_and_sanitize_subtopics([sub1, sub2], "Notes.pdf", total_pages=2)
        assert len(validated) == 2
        # Topic Beta definition must have been regenerated/sanitized to avoid duplication
        assert validated[0].definition != validated[1].definition

    def test_coverage_not_artificially_truncated(self):
        """The generator must process all subtopics without an arbitrary 8-concept cap."""
        # Create 15 distinct subtopics in the pages
        pages = []
        for i in range(1, 16):
            pages.append({
                "page": i,
                "text": f"Subtopic {i}\nSubtopic {i} is a distinct component in the course syllabus.\n- Property {i}\n- Mechanism {i}",
                "tables": []
            })

        subtopics = self.generator._extract_document_subtopics("Big_Syllabus.pdf", pages)
        # In the old code, `if count >= 8: break` artificially truncated to 8. Now it must capture all.
        assert len(subtopics) >= 15

    def test_selective_unit_filtering_single_document(self):
        """User selecting 1 specific unit must generate notes containing only that unit."""
        from unittest.mock import MagicMock
        from backend.models import User, Folder, Document

        mock_db = MagicMock()
        user = User(id="u-123", email="student@nalanda.edu")
        folder = Folder(id="f-456", user_id="u-123", name="Operating Systems")

        doc1 = Document(id="d-1", folder_id="f-456", original_filename="Unit_1_Processes.pdf", status="completed")
        doc2 = Document(id="d-2", folder_id="f-456", original_filename="Unit_2_Memory.pdf", status="completed")
        doc3 = Document(id="d-3", folder_id="f-456", original_filename="Unit_3_FileSystems.pdf", status="completed")

        mock_db.query().filter().first.return_value = folder
        mock_db.query().filter().all.return_value = [doc1, doc2, doc3]
        mock_db.query().filter().delete.return_value = 1

        # Mock page loading for each doc
        self.generator._load_document_pages = MagicMock(side_effect=lambda d, db: [
            {"page": 1, "text": f"{d.original_filename}\n{d.original_filename} is a core syllabus module.\n- Concept {d.id}", "tables": []}
        ])

        # Test selecting only Unit 2
        result = self.generator.generate_notes_for_folder(
            db=mock_db,
            current_user=user,
            folder_id="f-456",
            force_refresh=True,
            document_ids=["d-2"]
        )

        assert result.documents_analyzed == ["Unit_2_Memory.pdf"]
        assert "Unit_1_Processes.pdf" not in result.documents_analyzed
        assert "Unit_3_FileSystems.pdf" not in result.documents_analyzed

    def test_selective_unit_filtering_subset_documents(self):
        """User selecting a subset of units must generate notes containing all selected units."""
        from unittest.mock import MagicMock
        from backend.models import User, Folder, Document

        mock_db = MagicMock()
        user = User(id="u-123", email="student@nalanda.edu")
        folder = Folder(id="f-456", user_id="u-123", name="Operating Systems")

        doc1 = Document(id="d-1", folder_id="f-456", original_filename="Unit_1_Processes.pdf", status="completed")
        doc2 = Document(id="d-2", folder_id="f-456", original_filename="Unit_2_Memory.pdf", status="completed")
        doc3 = Document(id="d-3", folder_id="f-456", original_filename="Unit_3_FileSystems.pdf", status="completed")

        mock_db.query().filter().first.return_value = folder
        mock_db.query().filter().all.return_value = [doc1, doc2, doc3]
        mock_db.query().filter().delete.return_value = 1

        self.generator._load_document_pages = MagicMock(side_effect=lambda d, db: [
            {"page": 1, "text": f"{d.original_filename}\n{d.original_filename} is a core syllabus module.\n- Concept {d.id}", "tables": []}
        ])

        # Test selecting Unit 1 and Unit 3
        result = self.generator.generate_notes_for_folder(
            db=mock_db,
            current_user=user,
            folder_id="f-456",
            force_refresh=True,
            document_ids=["d-1", "d-3"]
        )

        assert set(result.documents_analyzed) == {"Unit_1_Processes.pdf", "Unit_3_FileSystems.pdf"}
        assert "Unit_2_Memory.pdf" not in result.documents_analyzed

    def test_selective_unit_invalid_selection_raises_value_error(self):
        """Selecting non-existent or uncompleted documents raises a clear descriptive ValueError."""
        from unittest.mock import MagicMock
        from backend.models import User, Folder, Document

        mock_db = MagicMock()
        user = User(id="u-123", email="student@nalanda.edu")
        folder = Folder(id="f-456", user_id="u-123", name="Operating Systems")
        doc1 = Document(id="d-1", folder_id="f-456", original_filename="Unit_1.pdf", status="completed")

        mock_db.query().filter().first.return_value = folder
        mock_db.query().filter().all.return_value = [doc1]

        with pytest.raises(ValueError, match="None of the selected unit files were found"):
            self.generator.generate_notes_for_folder(
                db=mock_db,
                current_user=user,
                folder_id="f-456",
                force_refresh=True,
                document_ids=["d-nonexistent"]
            )

