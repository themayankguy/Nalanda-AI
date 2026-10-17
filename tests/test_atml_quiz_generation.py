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

    # Query with doc_ids query parameter on empty folder
    res_quiz_filtered = client.get(f"/api/folders/{folder_id}/quiz?doc_ids=doc-fake-1", headers=headers)
    assert res_quiz_filtered.status_code == 404


def test_quiz_generator_ten_questions_and_selective_filtering():
    """Verify quiz generator can synthesize up to 10 questions and respects document_ids filtering."""
    from unittest.mock import MagicMock
    from backend.services.atml.quiz_generator import QuizGenerator

    generator = QuizGenerator()

    # Create 12 distinct academic concept definitions across two documents
    doc1_chunks = [
        {"document_id": "doc-unit-1", "source_filename": "Unit 1.pdf", "page_number": i + 1, "section_title": f"Concept {i}", "text": f"ConceptAlpha{i} is defined as a specialized architectural component that handles data pipelines in system tier {i}."}
        for i in range(6)
    ]
    doc2_chunks = [
        {"document_id": "doc-unit-2", "source_filename": "Unit 2.pdf", "page_number": i + 1, "section_title": f"Concept {i+6}", "text": f"ConceptBeta{i} is defined as an algorithmic optimization technique that accelerates query convergence in phase {i}."}
        for i in range(6)
    ]
    all_chunks = doc1_chunks + doc2_chunks

    # Mock DB and Folder
    mock_db = MagicMock()
    mock_user = MagicMock(id="user-123")
    mock_folder = MagicMock(id="folder-123", name="Distributed Computing", user_id="user-123")
    mock_db.query.return_value.filter.return_value.first.return_value = mock_folder

    # Mock Document records
    mock_doc1 = MagicMock(id="doc-unit-1", original_filename="Unit 1.pdf", status="completed", extraction=None)
    mock_doc2 = MagicMock(id="doc-unit-2", original_filename="Unit 2.pdf", status="completed", extraction=None)

    def mock_query_filter(*args, **kwargs):
        mock_q = MagicMock()
        mock_q.all.return_value = [mock_doc1, mock_doc2]
        mock_q.filter.return_value.all.return_value = [mock_doc1]
        return mock_q

    # Test 1: Synthesize 10 questions across both documents
    import backend.services.atml.quiz_generator as qg_mod
    original_vector_index = qg_mod.FolderVectorIndex

    class MockVectorIndex:
        def __init__(self, index_dir):
            self.chunks = all_chunks

    qg_mod.FolderVectorIndex = MockVectorIndex
    try:
        # Generate 10 questions
        res_10 = generator.generate_quiz_for_folder(
            db=mock_db,
            current_user=mock_user,
            folder_id="folder-123",
            num_questions=10,
        )
        assert len(res_10.questions) == 10
        assert res_10.total_questions == 10

        # Test 2: Selective filtering on doc-unit-1 only
        res_filtered = generator.generate_quiz_for_folder(
            db=mock_db,
            current_user=mock_user,
            folder_id="folder-123",
            num_questions=10,
            document_ids=["doc-unit-1"],
        )
        # All questions must strictly come from Unit 1.pdf
        assert len(res_filtered.questions) > 0
        assert all(q.source_document == "Unit 1.pdf" for q in res_filtered.questions)
        assert not any(q.source_document == "Unit 2.pdf" for q in res_filtered.questions)
    finally:
        qg_mod.FolderVectorIndex = original_vector_index


class TestQuizQuestionQualityValidationAndContextPreservation:
    """Rigorous tests for Nalanda AI Quiz Quality: Self-containment, validation, and distractor anti-collision."""

    def test_rejection_of_incomplete_not_required_without_context(self):
        """Verify candidate anchor is discarded when mode/subject cannot be established from the source."""
        from backend.services.atml.quiz_generator import QuizGenerator, validate_quiz_question
        generator = QuizGenerator()

        orphan_chunks = [
            {
                "document_id": "doc-orphan",
                "source_filename": "untitled.pdf",
                "page_number": 1,
                "section_title": "",
                "text": "In general workloads, Hadoop and HDFS are not required in this mode.",
            }
        ]
        anchors = generator.extract_fact_anchors(orphan_chunks)
        assert not any(a.get("concept") == "Hadoop and HDFS" for a in anchors)

        bad_q = QuizQuestion(
            id="bad-1",
            question='Which concept or technology is defined as: "Not required in this mode"?',
            options=["A) Hadoop Cluster", "B) Data in Hadoop", "C) Hadoop and HDFS", "D) Hadoop"],
            correct_option_index=2,
            correct_answer="Hadoop and HDFS",
            explanation="According to BDA UNIT 4.pdf (Page 3), Hadoop and HDFS is: not required in this mode.",
            source_document="BDA UNIT 4.pdf",
            page_number=3,
            section_title="Modes",
            concept="Hadoop and HDFS",
        )
        is_valid, failures = validate_quiz_question(bad_q)
        assert not is_valid
        assert any("sentence-completion" in f or "too similar" in f for f in failures)

    def test_bda_unit_4_mode_context_retrieval_and_complete_regeneration(self):
        """Test the exact BDA UNIT 4.pdf Page 3 example."""
        from backend.services.atml.quiz_generator import QuizGenerator, validate_quiz_question, are_options_too_similar
        generator = QuizGenerator()

        bda_chunk = {
            "document_id": "doc-bda4",
            "source_filename": "BDA UNIT 4.pdf",
            "page_number": 3,
            "section_title": "7. Execution Modes of Apache Pig",
            "text": (
                "7. Execution Modes of Apache Pig\n"
                "Apache Pig can be executed in two modes:\n"
                "A. Local Mode\n"
                "- In Local Mode, all files are processed using the local file system of the machine. "
                "Hadoop and HDFS are not required in this mode. It is mainly used for testing and development purposes.\n"
                "- Example: pig -x local\n"
                "B. MapReduce Mode / HDFS Mode\n"
                "- In MapReduce Mode, Pig processes data that is stored in HDFS."
            ),
        }

        anchors = generator.extract_fact_anchors([bda_chunk])
        mode_anchors = [a for a in anchors if a.get("anchor_type") == "mode_requirement"]
        assert len(mode_anchors) >= 1
        ma = mode_anchors[0]
        assert ma["concept"] == "Hadoop and HDFS"
        assert ma["mode"] == "Local Mode"
        assert ma["subject"] == "Apache Pig"
        assert ma["requirement"] == "not_required"

        stem = generator.generate_question_stem(
            concept=ma["concept"],
            statement=ma["statement"],
            chunk=bda_chunk,
            anchor=ma,
            model=None,
            tokenizer=None,
        )
        assert "Local Mode" in stem
        assert "Apache Pig" in stem
        assert "not being required" in stem or "not required" in stem
        assert '"Not required in this mode"' not in stem

        distractors = generator.mine_semantic_distractors(
            correct_concept=ma["concept"],
            candidate_pool=["Hadoop", "Hadoop Cluster", "Data in Hadoop", "Local File System", "Java Virtual Machine", "Grunt Shell"],
            embedding_model=None,
            anchor=ma,
            top_k=3,
        )
        assert not any("hadoop" in d.lower() for d in distractors)
        for d in distractors:
            assert not are_options_too_similar(d, ma["concept"])

        all_options = [ma["concept"]] + distractors[:3]
        clean_opts = [f"{chr(65+i)}) {opt}" for i, opt in enumerate(all_options)]
        full_q = QuizQuestion(
            id="q-bda-verified",
            question=stem,
            options=clean_opts,
            correct_option_index=0,
            correct_answer=ma["concept"],
            explanation="According to BDA UNIT 4.pdf (Page 3), in Local Mode of Apache Pig, Hadoop and HDFS are not required because files are processed using the local file system.",
            source_document="BDA UNIT 4.pdf",
            page_number=3,
            section_title="7. Execution Modes of Apache Pig",
            concept=ma["concept"],
        )
        is_valid, failures = validate_quiz_question(full_q, bda_chunk["text"])
        assert is_valid, f"Validation failed with: {failures}"

    def test_distractor_anti_collision_rules(self):
        """Verify strict pairwise collision rejection."""
        from backend.services.atml.quiz_generator import are_options_too_similar

        assert are_options_too_similar("Hadoop", "Hadoop and HDFS")
        assert are_options_too_similar("Hadoop Cluster", "Hadoop and HDFS")
        assert are_options_too_similar("Hadoop", "Hadoop Cluster")
        assert are_options_too_similar("Data in Hadoop", "Hadoop")

        assert not are_options_too_similar("Hadoop and HDFS", "Local File System")
        assert not are_options_too_similar("Hadoop and HDFS", "Java Virtual Machine")
        assert not are_options_too_similar("Local File System", "Grunt Shell")
        assert not are_options_too_similar("Tuple", "Bag")
        assert not are_options_too_similar("Atom", "Map")

    def test_elimination_of_literal_sentence_completions(self):
        """Ensure no question stem produces quoted completions or dangling phrases."""
        from backend.services.atml.quiz_generator import QuizGenerator
        generator = QuizGenerator()

        stem = generator.generate_question_stem(
            concept="Tuple",
            definition="an ordered collection of fields in a relation.",
            statement="A tuple is an ordered collection of fields.",
            chunk={"section_title": "6. Pig Data Model", "source_filename": "BDA UNIT 4.pdf"},
            model=None,
            tokenizer=None,
        )
        assert '"' not in stem
        assert "defined as:" not in stem
        assert stem.endswith("?")
        assert "ordered collection of fields" in stem

    def test_validation_rejects_missing_mode_context(self):
        """Verify question that mentions 'mode' without specifying which mode is rejected."""
        from backend.services.atml.quiz_generator import QuizQuestion, validate_quiz_question

        ambiguous_q = QuizQuestion(
            id="q-ambig",
            question="Which component is not required in this mode?",
            options=["A) Hadoop and HDFS", "B) Local File System", "C) JVM", "D) Grunt"],
            correct_option_index=0,
            correct_answer="Hadoop and HDFS",
            explanation="According to the notes, Hadoop and HDFS are not required.",
            source_document="BDA UNIT 4.pdf",
            page_number=3,
            section_title="Modes",
            concept="Hadoop and HDFS",
        )
        is_valid, failures = validate_quiz_question(ambiguous_q)
        assert not is_valid
        assert any("mode" in f.lower() or "anaphoric" in f.lower() for f in failures)

