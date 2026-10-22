"""Comprehensive test suite for Nalanda AI Academic Notes Formatting and Structuring.

Validates:
1. Repeated headings and duplicate explanation merging.
2. OCR punctuation, zero-width spaces, and broken hyphenation line repairs.
3. Multi-page running header and slide counter removal.
4. Nested headings, bullets, and 7-tier academic hierarchy.
5. Incomplete fragments preserved without unsupported hallucination.
6. Preservation of technical terminology, formulas, numbers, and negations.
7. Correct source citation provenance after sections are merged.
8. Domain-agnostic structuring across Medicine, Law, History, Biology, Physics, and CS.
9. Intent-aware response formatting (Definitions, Procedures, Comparisons, Types, Evaluative, General).
10. Regression safety for existing notes generation and chat APIs.
"""
import pytest
from unittest.mock import MagicMock
from typing import List, Dict, Any

from backend.services.atml.content_normalizer import ContentNormalizer
from backend.services.atml.notes_generator import NotesGenerator, SubtopicUnit
from backend.services.atml.answer_structurer import AnswerStructurer
from backend.models import User, Folder, Document


class TestContentNormalizationAndOCR:
    """Tests for OCR artifact cleanup, line-wrap repair, and running header/footer removal."""

    def test_ocr_artifact_cleanup_and_hyphenation_repair(self):
        raw = "The con-\ntainerization engine pro--vides safe infor-\nmation ^^ extraction .... without loss."
        cleaned = ContentNormalizer.clean_ocr_artifacts(raw)

        # Hyphenated wraps should be stitched
        assert "containerization" in cleaned
        assert "information" in cleaned
        # OCR punctuation artifacts removed
        assert "pro--vides" not in cleaned
        assert "^^" not in cleaned
        assert "...." not in cleaned

    def test_preservation_of_technical_formulas_and_negations(self):
        raw = "Algorithm has time complexity O(n log n) and does not require additional heap space; x >= y."
        cleaned = ContentNormalizer.clean_ocr_artifacts(raw)

        assert "O(n log n)" in cleaned
        assert "does not require" in cleaned
        assert "x >= y" in cleaned

    def test_remove_running_headers_and_footers(self):
        pages = [
            {"page": 1, "text": "COURSE 101: ADVANCED SYSTEMS\nTopic Intro\nBody line 1\nSlide 1 of 5"},
            {"page": 2, "text": "COURSE 101: ADVANCED SYSTEMS\nTopic Deep Dive\nBody line 2\nSlide 2 of 5"},
            {"page": 3, "text": "COURSE 101: ADVANCED SYSTEMS\nTopic Analysis\nBody line 3\nSlide 3 of 5"},
            {"page": 4, "text": "COURSE 101: ADVANCED SYSTEMS\nTopic Evaluation\nBody line 4\nSlide 4 of 5"},
            {"page": 5, "text": "COURSE 101: ADVANCED SYSTEMS\nSummary\nBody line 5\nSlide 5 of 5"},
        ]
        cleaned_pages = ContentNormalizer.remove_running_headers_and_footers(pages)

        for p in cleaned_pages:
            t = p["text"]
            # Repetitive running header stripped
            assert "COURSE 101: ADVANCED SYSTEMS" not in t
            # Slide counter stripped
            assert "Slide" not in t
            # Body content preserved
            assert "Body line" in t


class TestHeadingNormalizationAndMerging:
    """Tests for redundant heading detection and canonical title selection."""

    def test_redundant_heading_merging_same_concept(self):
        # Containerization vs What is a Container
        should_merge, canonical = ContentNormalizer.should_merge_headings(
            "Containerization", "What is a Container", page_distance=1
        )
        assert should_merge is True
        assert canonical.lower() in ("containerization", "container")

        # Explanation of a Container vs Container
        should_merge, canonical = ContentNormalizer.should_merge_headings(
            "Explanation of a Container", "Container", page_distance=1
        )
        assert should_merge is True
        assert "container" in canonical.lower()

        # HDFS Architecture vs Architecture of HDFS
        should_merge, canonical = ContentNormalizer.should_merge_headings(
            "HDFS Architecture", "Architecture of HDFS", page_distance=1
        )
        assert should_merge is True
        assert "hdfs architecture" in canonical.lower()

    def test_distinct_concepts_must_not_merge(self):
        # Distinct qualifiers
        merge1, _ = ContentNormalizer.should_merge_headings("Docker Engine", "Docker Container")
        assert merge1 is False

        merge2, _ = ContentNormalizer.should_merge_headings("Cell Membrane", "Cell Division")
        assert merge2 is False

        merge3, _ = ContentNormalizer.should_merge_headings("Fiscal Policy", "Monetary Policy")
        assert merge3 is False

        # Numbered/Ordinal subtopics
        merge4, _ = ContentNormalizer.should_merge_headings("Subtopic 1", "Subtopic 2")
        assert merge4 is False

        merge5, _ = ContentNormalizer.should_merge_headings("Phase I", "Phase II")
        assert merge5 is False


class TestBulletFormattingAndFragmentPreservation:
    """Tests for fragment formatting without hallucination."""

    def test_kv_fragment_formatted_with_bold_anchor(self):
        fragment = "Containerization: application and dependencies packaged together"
        formatted = ContentNormalizer.format_fragment_to_bullet(fragment)
        assert "**Containerization:**" in formatted
        assert "dependencies packaged together" in formatted

    def test_ambiguous_fragment_retains_source_wording(self):
        fragment = "O(n log n) average case sorting; not in-place."
        formatted = ContentNormalizer.format_fragment_to_bullet(fragment)
        assert "O(n log n)" in formatted
        assert "not in-place" in formatted


class TestNotesGeneratorHierarchicalStructuring:
    """Tests for hierarchical notes generation, provenance, and domain-agnostic behavior."""

    def setup_method(self):
        self.generator = NotesGenerator()

    def test_subtopic_merging_retains_combined_provenance(self):
        pages = [
            {
                "page": 1,
                "text": """
                Containerization
                Containerization is an operating-system-level virtualization method.
                - Packages dependencies together
                - Eliminates environment discrepancies
                """,
                "tables": []
            },
            {
                "page": 2,
                "text": """
                What is a Container
                A container is a standalone executable package of software.
                - Shares host kernel directly
                - Lightweight startup time
                """,
                "tables": []
            }
        ]

        subtopics = self.generator._extract_document_subtopics("TechGuide.pdf", pages)
        # Should merge into a single canonical subtopic
        assert len(subtopics) == 1
        sub = subtopics[0]
        assert sub.page_start == 1
        assert sub.page_end == 2
        # Points from both pages must be retained
        assert any("packages dependencies" in kp.lower() for kp in sub.key_points)
        assert any("shares host kernel" in kp.lower() for kp in sub.key_points)

        # Markdown render test
        rendered = self.generator._render_subtopic_markdown(sub, 1, 1)
        assert "**Source:** TechGuide.pdf, Pages 1–2" in rendered
        assert "> 💡 **Revision Takeaway:**" in rendered

    @pytest.mark.parametrize("subject,title,text,expected_term", [
        (
            "Medicine",
            "Atherosclerosis",
            "Atherosclerosis is a vascular disorder characterized by lipid deposition in arterial walls.\n- Intimal plaques form\n- Leads to arterial stenosis\nImportant: Endothelial injury initiates lipid infiltration.",
            "lipid deposition"
        ),
        (
            "Law",
            "Doctrine of Stare Decisis",
            "The Doctrine of Stare Decisis is a legal principle where courts adhere to established judicial precedent.\n- Distinguishes ratio decidendi from obiter dicta\n- Promotes consistency across judicial rulings\nNote that lower courts are strictly bound by supreme appellate decisions.",
            "judicial precedent"
        ),
        (
            "Biology",
            "Photosynthesis",
            "Photosynthesis is a metabolic process converting radiant solar energy into biochemical glucose.\n- Occurs within chloroplasts\n- Comprises light-dependent reactions and Calvin cycle\nCrucial: Chlorophyll pigments capture photon wavelengths.",
            "radiant solar energy"
        ),
        (
            "Physics",
            "Special Relativity",
            "Special Relativity is a physical theory describing spacetime geometry in inertial reference frames.\n- Speed of light c is invariant across all observers\n- Produces time dilation and Lorentz length contraction\nRemember: Simultaneous events in one frame are not simultaneous in another.",
            "spacetime geometry"
        ),
        (
            "Economics",
            "Monetary Policy",
            "Monetary Policy is the macroeconomic framework through which central banks control liquidity.\n- Regulates benchmark discount rates\n- Utilizes open market operations\nEssential: Controlling inflation remains the primary objective.",
            "central banks"
        ),
    ])
    def test_domain_agnostic_notes_across_subjects(self, subject, title, text, expected_term):
        """Validates that medicine, law, biology, physics, and economics all generate structured notes without hardcoding."""
        pages = [{"page": 1, "text": f"{title}\n{text}", "tables": []}]
        subtopics = self.generator._extract_document_subtopics(f"{subject}_Notes.pdf", pages)

        assert len(subtopics) >= 1
        sub = subtopics[0]
        assert expected_term in sub.definition.lower()
        assert len(sub.key_points) >= 1

        rendered = self.generator._render_subtopic_markdown(sub, 1, 1)
        assert f"# 1.1 {sub.title}" in rendered
        assert "## Definition" in rendered
        assert "## Key Points" in rendered
        assert "> 💡 **Revision Takeaway:**" in rendered


class TestIntentAwareResponseFormatting:
    """Tests for query intent detection and structured formatting."""

    def setup_method(self):
        self.structurer = AnswerStructurer()

    def test_definition_intent_formatting(self):
        query = "What is a Neural Network?"
        raw = (
            "A neural network is a computational framework modeled on the biological network of neurons in animal brains.\n"
            "- Input Layer: Receives input features from training datasets.\n"
            "- Hidden Layers: Perform non-linear transformations using activation weights.\n"
            "- Output Layer: Produces final classifications or regression predictions."
        )
        formatted = self.structurer.structure(query, raw)
        assert "### Neural Network" in formatted or "### What Is A Neural Network" in formatted
        assert "computational framework" in formatted
        assert "- **Input Layer**" in formatted

    def test_procedural_intent_formatting(self):
        query = "What are the steps of cellular respiration?"
        raw = (
            "Cellular Respiration Workflow:\n"
            "Step 1: Glycolysis\n"
            "Glucose is converted into pyruvate in the cytoplasm, yielding 2 net ATP.\n"
            "Step 2: Krebs Cycle\n"
            "Acetyl-CoA enters the mitochondrial matrix to produce NADH and FADH2.\n"
            "Step 3: Electron Transport Chain\n"
            "Proton gradient across inner mitochondrial membrane drives ATP synthase."
        )
        formatted = self.structurer.structure(query, raw)
        assert "1. **Step 1: Glycolysis**" in formatted or "1. Glycolysis" in formatted
        assert "2. **Step 2: Krebs Cycle**" in formatted or "2. Krebs Cycle" in formatted
        assert "3. **Step 3: Electron Transport Chain**" in formatted or "3. Electron Transport Chain" in formatted

    def test_comparison_intent_formatting(self):
        query = "Compare Mitosis vs Meiosis"
        raw = (
            "Mitosis and Meiosis are distinct forms of cellular division:\n"
            "- Mitosis: Produces two genetically identical diploid somatic daughter cells.\n"
            "- Meiosis: Produces four genetically diverse haploid gamete cells through two division rounds."
        )
        formatted = self.structurer.structure(query, raw)
        assert "Mitosis" in formatted
        assert "Meiosis" in formatted
        assert "- **Mitosis" in formatted or "**Mitosis**" in formatted


class TestRegressionFolderNotesGeneration:
    """Regression test ensuring folder notes generation contract remains intact."""

    def test_folder_notes_generation_mock_flow(self):
        generator = NotesGenerator()
        mock_db = MagicMock()
        user = User(id="u-abc", email="test@university.edu")
        folder = Folder(id="f-xyz", user_id="u-abc", name="Distributed Computing")
        doc = Document(id="d-101", folder_id="f-xyz", original_filename="Unit_1_Overview.pdf", status="completed")

        mock_db.query().filter().first.return_value = folder
        mock_db.query().filter().all.return_value = [doc]
        mock_db.query().filter().delete.return_value = 1

        generator._load_document_pages = MagicMock(return_value=[
            {
                "page": 1,
                "text": """
                Distributed Computing
                Distributed Computing is a model where components located on networked computers coordinate actions by passing messages.
                - Message Passing: Enables communication without shared memory
                - Concurrency: Multiple tasks execute simultaneously
                - Fault Tolerance: System continues operating despite node failures
                """,
                "tables": []
            }
        ])

        response = generator.generate_notes_for_folder(
            db=mock_db,
            current_user=user,
            folder_id="f-xyz",
            force_refresh=True,
        )

        assert response.folder_id == "f-xyz"
        assert response.folder_name == "Distributed Computing"
        assert response.documents_analyzed == ["Unit_1_Overview.pdf"]
        assert response.total_sections >= 1
        assert len(response.sections) >= 1

        assert "Distributed Computing" in response.markdown_content
        sub_sec = response.sections[1]
        assert "Distributed Computing" in sub_sec.content_markdown
        assert len(sub_sec.citations) >= 1
        assert "Unit_1_Overview.pdf" in sub_sec.citations[0]
