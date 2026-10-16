"""Unit tests for ATML Quiz and Question Generation Engine.

Verifies:
- Educational fact anchor mining from course passages
- Inverted Seq2Seq question stem formulation
- Latent vector distractor mining and uniqueness guarantees
- Correct option positioning and citation preservation
- Graceful handling of empty or non-existent folders
"""
import pytest
from backend.services.atml.quiz_generator import QuizGenerator, QuizQuestion, QuizResponse
from backend.services.atml.config import ATMLConfig


class TestQuizGeneratorAnchorsAndStems:
    """Tests for fact-anchor extraction and question generation."""

    def test_extract_fact_anchors_definitions(self):
        generator = QuizGenerator()
        chunks = [
            {
                "text": (
                    "Data Visualization is the process of representing data in visual form such as charts and graphs.\n"
                    "MapReduce refers to a software framework for easily writing applications which process vast amounts of data."
                ),
                "document_id": "doc-1",
                "source_filename": "lecture.pdf",
                "page_number": 2,
                "section_title": "Overview",
            }
        ]
        anchors = generator.extract_fact_anchors(chunks)
        assert len(anchors) >= 1
        concepts = [a["concept"] for a in anchors]
        assert "Data Visualization" in concepts or "MapReduce" in concepts

    def test_extract_fact_anchors_bold_and_steps(self):
        generator = QuizGenerator()
        chunks = [
            {
                "text": (
                    "**HDFS NameNode**: Manages the file system namespace and maintains the metadata tree.\n"
                    "1. Understand the Data: Examine available numerical and categorical columns."
                ),
                "document_id": "doc-2",
                "source_filename": "hadoop.pdf",
                "page_number": 5,
                "section_title": "Architecture",
            }
        ]
        anchors = generator.extract_fact_anchors(chunks)
        assert len(anchors) >= 1
        concepts = [a["concept"] for a in anchors]
        assert any("HDFS NameNode" in c or "Understand the Data" in c for c in concepts)

    def test_generate_question_stem_phrasing(self):
        generator = QuizGenerator()
        stem = generator.generate_question_stem(
            concept="Data Visualization",
            definition="the process of representing data in graphical form such as charts and diagrams.",
            statement="Data Visualization is the process of representing data in graphical form.",
            model=None,
            tokenizer=None,
        )
        assert stem.endswith("?")
        assert len(stem) > 20
        # Concept name should not appear as an obvious giveaway inside the question stem
        assert "Data Visualization" not in stem

    def test_mine_semantic_distractors_uniqueness(self):
        generator = QuizGenerator()
        correct_concept = "Supervised Learning"
        pool = [
            "Supervised Learning",  # Target itself
            "Unsupervised Learning",
            "Reinforcement Learning",
            "Semi-supervised Learning",
            "Feature Engineering",
            "Gradient Descent",
        ]
        distractors = generator.mine_semantic_distractors(
            correct_concept=correct_concept,
            candidate_pool=pool,
            embedding_model=None,
            top_k=3,
        )
        assert len(distractors) == 3
        assert correct_concept not in distractors
        # All distractors must be unique
        assert len(set(distractors)) == 3

    def test_distractor_fallback_when_pool_is_small(self):
        generator = QuizGenerator()
        distractors = generator.mine_semantic_distractors(
            correct_concept="Hadoop",
            candidate_pool=["Hadoop"],  # Empty other candidates
            embedding_model=None,
            top_k=3,
        )
        assert len(distractors) == 3
        assert "Hadoop" not in distractors

    def test_rejection_of_stopwords_and_fillers(self):
        from backend.services.atml.quiz_generator import clean_academic_concept, is_valid_definition
        # Meta fillers and slide outlines must be rejected
        assert clean_academic_concept("Command") is None
        assert clean_academic_concept("Command Used") is None
        assert clean_academic_concept("Required Action") is None
        assert clean_academic_concept("Features") is None
        assert clean_academic_concept("Step 1") is None
        assert clean_academic_concept("Minimal Errors Explanation") is None
        assert clean_academic_concept("Working") is None
        assert clean_academic_concept("Components") is None

        # Industry application examples and out-of-context hardware words must be rejected
        assert clean_academic_concept("Airlines") is None
        assert clean_academic_concept("Manufacturing") is None
        assert clean_academic_concept("Sports") is None
        assert clean_academic_concept("Rack") is None
        assert clean_academic_concept("Driver") is None
        assert clean_academic_concept("Capacity") is None
        assert clean_academic_concept("Important Component") is None

        # Hyphenated academic concepts must be preserved
        assert clean_academic_concept("Drill-down") == "Drill-down"

        # Core CS single-word concepts from OS, Networks, DBMS must pass
        assert clean_academic_concept("Semaphore") == "Semaphore"
        assert clean_academic_concept("Deadlock") == "Deadlock"
        assert clean_academic_concept("Router") == "Router"
        assert clean_academic_concept("Normalization") == "Normalization"

        # Multi-word and acronym concepts from other domains must pass
        assert clean_academic_concept("Virtual Memory") == "Virtual Memory"
        assert clean_academic_concept("Packet Switching") == "Packet Switching"
        assert clean_academic_concept("TCP") == "TCP"

        # CLI command lines must be rejected as definitions
        assert not is_valid_definition("docker run -d -p 8080:80 nginx")
        assert not is_valid_definition("docker create --name <container-name> <image-name>")
        assert not is_valid_definition("t <image-name> .")

        # Valid definitions must pass
        assert is_valid_definition("A lightweight, standalone, and executable package that includes application dependencies.")

    def test_options_are_domain_related_concepts(self):
        generator = QuizGenerator()
        distractors = generator.mine_semantic_distractors(
            correct_concept="Docker Container",
            candidate_pool=["Docker Container"],
            embedding_model=None,
            top_k=3,
        )
        assert len(distractors) == 3
        for d in distractors:
            # Distractors must not be stopwords or meta fillers
            assert d.lower() not in {"command", "features", "step 1", "required action", "airlines", "manufacturing", "rack"}
            # Must be valid length
            assert len(d) > 3

    def test_visualization_options_are_sibling_charts(self):
        generator = QuizGenerator()
        distractors = generator.mine_semantic_distractors(
            correct_concept="Line Chart",
            candidate_pool=["Line Chart"],
            embedding_model=None,
            top_k=3,
        )
        assert len(distractors) == 3
        # Should contain visual sibling charts like Bar Chart, Scatter Plot, Histogram, etc.
        assert all(any(w in d.lower() for w in ["chart", "plot", "histogram", "distribution", "map"]) for d in distractors)

    def test_os_options_are_sibling_os_concepts(self):
        generator = QuizGenerator()
        distractors = generator.mine_semantic_distractors(
            correct_concept="Virtual Memory",
            candidate_pool=["Virtual Memory"],
            context_text="Operating Systems Course Unit 3 Memory Management",
            embedding_model=None,
            top_k=3,
        )
        assert len(distractors) == 3
        assert "Virtual Memory" not in distractors
        # Options must come from OS domain
        assert any(term in distractors for term in [
            "CPU Scheduling", "Deadlock Prevention", "Mutual Exclusion", "Page Replacement", "Context Switching"
        ])

    def test_networks_options_are_sibling_network_concepts(self):
        generator = QuizGenerator()
        distractors = generator.mine_semantic_distractors(
            correct_concept="Packet Switching",
            candidate_pool=["Packet Switching"],
            context_text="Computer Networks Lecture 4 OSI Layer Routing",
            embedding_model=None,
            top_k=3,
        )
        assert len(distractors) == 3
        assert "Packet Switching" not in distractors
        # Options must come from Networks domain
        assert any(term in distractors for term in [
            "Circuit Switching", "Routing Protocol", "TCP/IP Protocol", "OSI Model", "Network Gateway"
        ])


class TestQuizEndToEndAssembly:
    """Tests for complete QuizQuestion serialization and citation integrity."""

    def test_quiz_question_dataclass_to_dict(self):
        q = QuizQuestion(
            id="q-001",
            question="Which framework divides workloads into map and reduce phases?",
            options=["A) Spark", "B) MapReduce", "C) Kafka", "D) Flink"],
            correct_option_index=1,
            correct_answer="MapReduce",
            explanation="MapReduce divides distributed tasks into map and reduce phases.",
            source_document="bigdata.pdf",
            page_number=8,
            section_title="MapReduce Execution",
            concept="MapReduce",
        )
        data = q.to_dict()
        assert data["id"] == "q-001"
        assert data["correct_option_index"] == 1
        assert data["correct_answer"] == "MapReduce"
        assert data["options"][1].endswith("MapReduce")
        assert data["page_number"] == 8
        assert data["source_document"] == "bigdata.pdf"


def test_quiz_api_endpoint(monkeypatch):
    """Test the /api/folders/{id}/quiz API endpoint."""
    import uuid
    from fastapi.testclient import TestClient
    from backend.main import app

    client = TestClient(app)
    uid = uuid.uuid4().hex[:8]

    # Register and create folder
    res_reg = client.post(
        "/api/auth/register",
        json={"email": f"quiz_user_{uid}@nalanda.ai", "username": f"quizer_{uid}", "password": "pass1234password"},
    )
    assert res_reg.status_code == 201
    token = res_reg.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    res_folder = client.post(
        "/api/folders",
        json={"name": "Big Data Course"},
        headers=headers,
    )
    assert res_folder.status_code == 201
    folder_id = res_folder.json()["id"]

    # Query quiz endpoint for empty folder (should return 0 questions gracefully)
    res_quiz = client.get(f"/api/folders/{folder_id}/quiz", headers=headers)
    assert res_quiz.status_code == 200
    quiz_data = res_quiz.json()
    assert quiz_data["folder_id"] == folder_id
    assert "questions" in quiz_data
    assert quiz_data["total_questions"] == 0
