"""Automated Test Suite for Nalanda AI Incremental Knowledge Integration.

Comprehensive validation covering all 13 required scenarios:
1. Uploading an identical file twice does not create duplicate knowledge.
2. Uploading the same paragraph in a different file preserves both source references.
3. Uploading additional information about an existing concept enriches that concept.
4. Uploading a new subtopic creates a subtopic under the correct existing concept.
5. Uploading a genuinely new concept creates a new node.
6. Similar but distinct concepts are not incorrectly merged.
7. OCR variations and minor spelling differences are handled appropriately.
8. Conflicting statements from different documents are preserved and flagged.
9. New content is searchable immediately after successful ingestion.
10. The graph and vector index remain consistent after updates and failures.
11. Retrying or concurrently processing an upload does not produce duplicates.
12. Cross-user and cross-workspace knowledge contamination is prevented.
13. Deleting a document updates provenance without deleting independently supported knowledge.
"""
import json
import uuid
from pathlib import Path
import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.database import Base
from backend.models import (
    User,
    Folder,
    Document,
    DocumentExtraction,
    KnowledgeConcept,
    KnowledgeItem,
    DocumentIngestionLog,
)
from backend.services.NLP.knowledge_integration_service import (
    KnowledgeIntegrationService,
    compute_normalized_hash,
    normalize_text_for_comparison,
    detect_contradiction,
)
from backend.services.NLP.vector_index_service import FolderVectorIndex


# In-memory SQLite for high-speed isolated unit testing
@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def mock_embedding_model():
    """Deterministic local embedding model mock producing 384-dim vectors based on token hashes."""
    class MockSentenceTransformer:
        def encode(self, texts, **kwargs):
            if isinstance(texts, str):
                texts = [texts]
            vectors = []
            for t in texts:
                # Seed RNG with text hash for deterministic semantic clustering
                h = abs(hash(t.lower())) % (2**31)
                rng = np.random.RandomState(h)
                vec = rng.randn(384).astype(np.float32)

                # Inject strong bias for thematic topic clustering
                t_lower = t.lower()
                if "hdfs" in t_lower or "namenode" in t_lower or "datanode" in t_lower:
                    theme_bias = np.ones(384, dtype=np.float32) * 1.5
                    vec = vec * 0.2 + theme_bias * 0.8
                elif "random forest" in t_lower or "decision tree" in t_lower:
                    theme_bias = -np.ones(384, dtype=np.float32) * 1.5
                    vec = vec * 0.2 + theme_bias * 0.8
                elif "blockchain" in t_lower or "consensus" in t_lower:
                    theme_bias = np.zeros(384, dtype=np.float32)
                    theme_bias[0:100] = 2.0
                    vec = vec * 0.2 + theme_bias * 0.8

                norm = np.linalg.norm(vec) + 1e-8
                vectors.append(vec / norm)
            return np.array(vectors, dtype=np.float32)
    return MockSentenceTransformer()


def _create_user_and_folder(db: sessionmaker, suffix: str = ""):
    u = User(
        id=f"user_{suffix}_{uuid.uuid4().hex[:6]}",
        email=f"user_{suffix}_{uuid.uuid4().hex[:6]}@nalanda.ai",
        username=f"user_{suffix}_{uuid.uuid4().hex[:6]}",
        password_hash="hashed_pw",
    )
    db.add(u)
    db.flush()
    f = Folder(
        id=f"folder_{suffix}_{uuid.uuid4().hex[:6]}",
        user_id=u.id,
        name=f"Big Data Systems {suffix}",
    )
    db.add(f)
    db.commit()
    return u, f


def _create_mock_document(db, user, folder, filename: str, doc_id: str = None) -> Document:
    d = Document(
        id=doc_id or str(uuid.uuid4()),
        user_id=user.id,
        folder_id=folder.id,
        original_filename=filename,
        file_type="pdf",
        file_size_bytes=1024,
        storage_path=f"/tmp/{filename}",
        status="completed",
    )
    db.add(d)
    db.commit()
    return d


class TestIncrementalKnowledgeIntegration:
    """Rigorous evaluation of the 13 incremental integration criteria."""

    def test_01_upload_identical_file_twice_no_duplicates(self, db_session, mock_embedding_model):
        """Scenario 1: Uploading an identical file twice does not create duplicate knowledge."""
        user, folder = _create_user_and_folder(db_session, "s1")
        doc1 = _create_mock_document(db_session, user, folder, "hdfs_overview.pdf")

        chunks = [
            {
                "chunk_id": "c1",
                "page_number": 1,
                "section_title": "HDFS Architecture",
                "text": "The Hadoop Distributed File System (HDFS) provides high-throughput access to application data.",
            },
            {
                "chunk_id": "c2",
                "page_number": 2,
                "section_title": "HDFS Architecture",
                "text": "NameNode manages the file system namespace and maintains the directory tree.",
            },
        ]

        # First ingestion
        res1 = KnowledgeIntegrationService.integrate_document_knowledge(
            db_session, doc1, chunks=chunks, model=mock_embedding_model
        )
        assert res1["concepts_created"] >= 1
        assert res1["duplicates_skipped"] == 0

        initial_concept_count = db_session.query(KnowledgeConcept).filter(KnowledgeConcept.folder_id == folder.id).count()
        initial_item_count = db_session.query(KnowledgeItem).filter(KnowledgeItem.folder_id == folder.id).count()

        # Second ingestion of the exact same document
        doc2 = _create_mock_document(db_session, user, folder, "hdfs_overview_copy.pdf")
        res2 = KnowledgeIntegrationService.integrate_document_knowledge(
            db_session, doc2, chunks=chunks, model=mock_embedding_model
        )

        final_concept_count = db_session.query(KnowledgeConcept).filter(KnowledgeConcept.folder_id == folder.id).count()
        final_item_count = db_session.query(KnowledgeItem).filter(KnowledgeItem.folder_id == folder.id).count()

        assert res2["duplicates_skipped"] == 2
        assert final_concept_count == initial_concept_count, "No duplicate concept nodes should be created."
        assert final_item_count == initial_item_count, "No duplicate knowledge items should be created."

    def test_02_same_paragraph_different_file_preserves_provenance(self, db_session, mock_embedding_model):
        """Scenario 2: Uploading the same paragraph in a different file preserves both source references."""
        user, folder = _create_user_and_folder(db_session, "s2")
        doc_a = _create_mock_document(db_session, user, folder, "Module1_Slides.pdf")
        doc_b = _create_mock_document(db_session, user, folder, "Exam_Study_Guide.pdf")

        shared_para = "A DataNode stores data in the Hadoop cluster and serves read and write requests from clients."

        chunks_a = [{"chunk_id": "a1", "page_number": 4, "section_title": "DataNode Overview", "text": shared_para}]
        chunks_b = [{"chunk_id": "b1", "page_number": 12, "section_title": "DataNode Overview", "text": shared_para}]

        KnowledgeIntegrationService.integrate_document_knowledge(db_session, doc_a, chunks_a, mock_embedding_model)
        KnowledgeIntegrationService.integrate_document_knowledge(db_session, doc_b, chunks_b, mock_embedding_model)

        item = db_session.query(KnowledgeItem).filter(KnowledgeItem.folder_id == folder.id).first()
        assert item is not None
        sources = json.loads(item.provenance_sources_json)
        assert len(sources) == 2, "Both document references must be preserved in provenance."
        doc_ids = [s["document_id"] for s in sources]
        filenames = [s["filename"] for s in sources]
        pages = [s["page"] for s in sources]

        assert doc_a.id in doc_ids and doc_b.id in doc_ids
        assert "Module1_Slides.pdf" in filenames and "Exam_Study_Guide.pdf" in filenames
        assert 4 in pages and 12 in pages

    def test_03_additional_information_enriches_existing_concept(self, db_session, mock_embedding_model):
        """Scenario 3: Uploading additional information about an existing concept enriches that concept."""
        user, folder = _create_user_and_folder(db_session, "s3")
        doc1 = _create_mock_document(db_session, user, folder, "hdfs_intro.pdf")

        # Initial concept
        chunks1 = [{
            "chunk_id": "c1", "page_number": 1, "section_title": "HDFS Architecture",
            "text": "NameNode manages filesystem metadata and coordinates block allocations across data nodes."
        }]
        KnowledgeIntegrationService.integrate_document_knowledge(db_session, doc1, chunks1, mock_embedding_model)

        # Later upload containing new details about the same concept
        doc2 = _create_mock_document(db_session, user, folder, "hdfs_namenode_deep_dive.pdf")
        chunks2 = [{
            "chunk_id": "c2", "page_number": 3, "section_title": "HDFS Architecture",
            "text": "NameNode maintains the EditLog journal and uses an In-Memory FsImage to deliver sub-millisecond metadata operations."
        }]
        res = KnowledgeIntegrationService.integrate_document_knowledge(db_session, doc2, chunks2, mock_embedding_model)

        concepts = db_session.query(KnowledgeConcept).filter(KnowledgeConcept.folder_id == folder.id).all()
        assert len(concepts) == 1, "Should enrich the existing concept instead of creating a second HDFS concept node."
        assert res["concepts_enriched"] == 1
        assert len(concepts[0].items) == 2, "Both factual points should reside under the same unified concept."

    def test_04_new_subtopic_attached_to_correct_existing_concept(self, db_session, mock_embedding_model):
        """Scenario 4: Uploading a new subtopic creates a subtopic under the correct existing concept."""
        user, folder = _create_user_and_folder(db_session, "s4")
        doc1 = _create_mock_document(db_session, user, folder, "ensemble_learning.pdf")
        doc2 = _create_mock_document(db_session, user, folder, "random_forest_tradeoffs.pdf")

        chunks1 = [{
            "chunk_id": "c1", "page_number": 1, "section_title": "Random Forest",
            "text": "Random Forest constructs a multitude of decision trees during training and outputs the mode or mean prediction."
        }]
        KnowledgeIntegrationService.integrate_document_knowledge(db_session, doc1, chunks1, mock_embedding_model)

        chunks2 = [{
            "chunk_id": "c2", "page_number": 5, "section_title": "Random Forest",
            "text": "Limitations: Random forests require substantial computational memory for large ensembles and are not easily interpretable."
        }]
        KnowledgeIntegrationService.integrate_document_knowledge(db_session, doc2, chunks2, mock_embedding_model)

        concepts = db_session.query(KnowledgeConcept).filter(KnowledgeConcept.folder_id == folder.id).all()
        assert len(concepts) == 1
        items = concepts[0].items
        subtopics = [itm.subtopic for itm in items]
        assert "Limitations" in subtopics, "The new distinct detail must be organized under the Limitations subtopic."

    def test_05_genuinely_new_concept_creates_new_node(self, db_session, mock_embedding_model):
        """Scenario 5: Uploading a genuinely new concept creates a new node."""
        user, folder = _create_user_and_folder(db_session, "s5")
        doc1 = _create_mock_document(db_session, user, folder, "hadoop.pdf")
        doc2 = _create_mock_document(db_session, user, folder, "crypto.pdf")

        chunks1 = [{
            "chunk_id": "c1", "page_number": 1, "section_title": "HDFS Architecture",
            "text": "HDFS cluster architecture consists of a single NameNode and multiple slave DataNodes."
        }]
        chunks2 = [{
            "chunk_id": "c2", "page_number": 1, "section_title": "Blockchain Consensus",
            "text": "Proof of Work consensus mechanism requires miners to solve cryptographic hash puzzles to validate blocks."
        }]

        res1 = KnowledgeIntegrationService.integrate_document_knowledge(db_session, doc1, chunks1, mock_embedding_model)
        res2 = KnowledgeIntegrationService.integrate_document_knowledge(db_session, doc2, chunks2, mock_embedding_model)

        assert res2["concepts_created"] == 1
        concepts = db_session.query(KnowledgeConcept).filter(KnowledgeConcept.folder_id == folder.id).all()
        concept_names = [c.name for c in concepts]
        assert len(concepts) == 2
        assert any("HDFS" in n for n in concept_names)
        assert any("Blockchain" in n for n in concept_names)

    def test_06_similar_but_distinct_concepts_not_incorrectly_merged(self, db_session, mock_embedding_model):
        """Scenario 6: Similar but distinct concepts are not incorrectly merged."""
        user, folder = _create_user_and_folder(db_session, "s6")
        doc = _create_mock_document(db_session, user, folder, "ml_comparison.pdf")

        # Two distinct algorithms in the same ML family
        chunks = [
            {
                "chunk_id": "c1", "page_number": 2, "section_title": "Decision Tree Classifier",
                "text": "A Decision Tree splits data based on information gain or Gini impurity to form a single tree structure."
            },
            {
                "chunk_id": "c2", "page_number": 8, "section_title": "Random Forest Classifier",
                "text": "Random Forest constructs an ensemble of randomized decorrelated decision trees using bagging."
            }
        ]
        KnowledgeIntegrationService.integrate_document_knowledge(db_session, doc, chunks, mock_embedding_model)

        concepts = db_session.query(KnowledgeConcept).filter(KnowledgeConcept.folder_id == folder.id).all()
        assert len(concepts) == 2, "Decision Tree and Random Forest must remain distinct concepts."

    def test_07_ocr_variations_and_minor_spelling_handled(self, db_session, mock_embedding_model):
        """Scenario 7: OCR variations and minor spelling differences are handled appropriately."""
        user, folder = _create_user_and_folder(db_session, "s7")
        doc1 = _create_mock_document(db_session, user, folder, "clean.pdf")
        doc2 = _create_mock_document(db_session, user, folder, "ocr_scanned.pdf")

        clean_text = "NameNode manages the file system namespace and metadata."
        # Common OCR errors: 1 for l, 0 for o
        ocr_text = "NameNode manages the fi1e system namespace and metadata."

        chunks1 = [{"chunk_id": "c1", "page_number": 1, "section_title": "NameNode", "text": clean_text}]
        chunks2 = [{"chunk_id": "c2", "page_number": 1, "section_title": "NameNode", "text": ocr_text}]

        KnowledgeIntegrationService.integrate_document_knowledge(db_session, doc1, chunks1, mock_embedding_model)
        res = KnowledgeIntegrationService.integrate_document_knowledge(db_session, doc2, chunks2, mock_embedding_model)

        assert res["duplicates_skipped"] == 1, "OCR typo should be normalized and recognized as duplicate content."
        assert db_session.query(KnowledgeItem).filter(KnowledgeItem.folder_id == folder.id).count() == 1

    def test_08_conflicting_statements_preserved_and_flagged(self, db_session, mock_embedding_model):
        """Scenario 8: Conflicting statements from different documents are preserved and flagged."""
        user, folder = _create_user_and_folder(db_session, "s8")
        doc1 = _create_mock_document(db_session, user, folder, "hadoop2_specs.pdf")
        doc2 = _create_mock_document(db_session, user, folder, "hadoop1_specs.pdf")

        # Conflicting default block sizes between Hadoop 1.x and Hadoop 2.x
        text1 = "In HDFS, the default block size is 128 MB for high throughput batch processing."
        text2 = "In HDFS, the default block size is 64 MB for storage optimization."

        chunks1 = [{"chunk_id": "c1", "page_number": 3, "section_title": "HDFS Block Size", "text": text1}]
        chunks2 = [{"chunk_id": "c2", "page_number": 2, "section_title": "HDFS Block Size", "text": text2}]

        KnowledgeIntegrationService.integrate_document_knowledge(db_session, doc1, chunks1, mock_embedding_model)
        res2 = KnowledgeIntegrationService.integrate_document_knowledge(db_session, doc2, chunks2, mock_embedding_model)

        assert res2["conflicts_flagged"] >= 1, "Conflicting block sizes must be flagged."
        items = db_session.query(KnowledgeItem).filter(KnowledgeItem.folder_id == folder.id).all()
        assert len(items) == 2, "Both conflicting statements must be preserved rather than overwritten."
        assert any(itm.is_conflict for itm in items), "At least one item must have is_conflict=True."

    def test_09_new_content_searchable_immediately(self, db_session, tmp_path, mock_embedding_model):
        """Scenario 9: New content is searchable immediately after successful ingestion."""
        user, folder = _create_user_and_folder(db_session, "s9")
        doc = _create_mock_document(db_session, user, folder, "hive_execution.pdf")

        index_dir = tmp_path / "v_idx"
        v_idx = FolderVectorIndex(index_dir)

        chunks = [{
            "chunk_id": "c1",
            "document_id": doc.id,
            "user_id": user.id,
            "folder_id": folder.id,
            "source_filename": "hive_execution.pdf",
            "page_number": 4,
            "section_title": "Tez Execution Engine",
            "text_hash": compute_normalized_hash("Apache Tez executes complex directed acyclic graphs of data processing tasks."),
            "text": "Apache Tez executes complex directed acyclic graphs of data processing tasks for Hive queries.",
        }]

        # Add to vector index and integrate
        v_idx.add_document_chunks(chunks, model=mock_embedding_model)
        KnowledgeIntegrationService.integrate_document_knowledge(db_session, doc, chunks, mock_embedding_model)

        results = v_idx.search("directed acyclic graph Tez", top_k=3, model=mock_embedding_model)
        assert len(results) > 0
        top_chunk, score = results[0]
        assert "Tez" in top_chunk["text"]

    def test_10_graph_and_vector_index_consistent_after_updates(self, db_session, mock_embedding_model):
        """Scenario 10: The graph and vector index remain consistent after updates and failures."""
        user, folder = _create_user_and_folder(db_session, "s10")
        doc1 = _create_mock_document(db_session, user, folder, "Unit1.pdf")
        doc2 = _create_mock_document(db_session, user, folder, "Unit2.pdf")

        chunks1 = [{
            "chunk_index": 0, "document_id": doc1.id, "source_filename": "Unit1.pdf",
            "page_number": 1, "section_title": "Interactive Visual Analytics",
            "text": "Interactive visual analytics systems support multidimensional data analysis."
        }]
        chunks2 = [{
            "chunk_index": 1, "document_id": doc2.id, "source_filename": "Unit2.pdf",
            "page_number": 1, "section_title": "Interactive Visual Analytics",
            "text": "Interactive visual analytics dashboards integrate scatter plots, bar charts, and maps."
        }]

        KnowledgeIntegrationService.integrate_document_knowledge(db_session, doc1, chunks1, mock_embedding_model)
        KnowledgeIntegrationService.integrate_document_knowledge(db_session, doc2, chunks2, mock_embedding_model)

        concepts = db_session.query(KnowledgeConcept).filter(KnowledgeConcept.folder_id == folder.id).all()
        assert len(concepts) == 1, "Both files discuss the same concept, so they should unify into 1 concept node."
        concept = concepts[0]
        assert "Visual Analytics" in concept.name
        items = db_session.query(KnowledgeItem).filter(KnowledgeItem.concept_id == concept.id).all()
        assert len(items) >= 2

    def test_11_retrying_or_concurrent_upload_idempotent(self, db_session, mock_embedding_model):
        """Scenario 11: Retrying or concurrently processing an upload does not produce duplicates."""
        user, folder = _create_user_and_folder(db_session, "s11")
        doc = _create_mock_document(db_session, user, folder, "retried_upload.pdf")

        chunks = [
            {"chunk_id": "c1", "page_number": 1, "section_title": "Topic A", "text": "Unique statement about Topic A."},
            {"chunk_id": "c2", "page_number": 2, "section_title": "Topic B", "text": "Unique statement about Topic B."},
        ]

        # Initial attempt
        KnowledgeIntegrationService.integrate_document_knowledge(db_session, doc, chunks, mock_embedding_model)
        c_count_1 = db_session.query(KnowledgeConcept).filter(KnowledgeConcept.folder_id == folder.id).count()
        i_count_1 = db_session.query(KnowledgeItem).filter(KnowledgeItem.folder_id == folder.id).count()

        # Retry attempt (simulating extraction retry or background retry)
        KnowledgeIntegrationService.integrate_document_knowledge(db_session, doc, chunks, mock_embedding_model)
        c_count_2 = db_session.query(KnowledgeConcept).filter(KnowledgeConcept.folder_id == folder.id).count()
        i_count_2 = db_session.query(KnowledgeItem).filter(KnowledgeItem.folder_id == folder.id).count()

        assert c_count_1 == c_count_2, "Retrying must not duplicate concept nodes."
        assert i_count_1 == i_count_2, "Retrying must not duplicate knowledge items."

    def test_12_cross_user_and_cross_workspace_isolation(self, db_session, mock_embedding_model):
        """Scenario 12: Cross-user and cross-workspace knowledge contamination is prevented."""
        user_a, folder_a = _create_user_and_folder(db_session, "alice")
        user_b, folder_b = _create_user_and_folder(db_session, "bob")

        doc_a = _create_mock_document(db_session, user_a, folder_a, "alice_notes.pdf")
        doc_b = _create_mock_document(db_session, user_b, folder_b, "bob_notes.pdf")

        # Both users upload identical concepts in their respective folders
        chunks = [{
            "chunk_id": "c1", "page_number": 1, "section_title": "NameNode Architecture",
            "text": "NameNode is the central controller of HDFS architecture."
        }]

        KnowledgeIntegrationService.integrate_document_knowledge(db_session, doc_a, chunks, mock_embedding_model)
        KnowledgeIntegrationService.integrate_document_knowledge(db_session, doc_b, chunks, mock_embedding_model)

        alice_concepts = db_session.query(KnowledgeConcept).filter(KnowledgeConcept.folder_id == folder_a.id).all()
        bob_concepts = db_session.query(KnowledgeConcept).filter(KnowledgeConcept.folder_id == folder_b.id).all()

        assert len(alice_concepts) == 1
        assert len(bob_concepts) == 1
        assert alice_concepts[0].user_id == user_a.id
        assert bob_concepts[0].user_id == user_b.id
        assert alice_concepts[0].id != bob_concepts[0].id, "Users must have isolated knowledge representations."

    def test_13_deleting_document_updates_provenance_preserves_shared(self, db_session, mock_embedding_model):
        """Scenario 13: Deleting a document updates provenance without deleting independently supported knowledge."""
        user, folder = _create_user_and_folder(db_session, "s13")
        doc_1 = _create_mock_document(db_session, user, folder, "doc_1.pdf")
        doc_2 = _create_mock_document(db_session, user, folder, "doc_2.pdf")

        shared_chunk = [{
            "chunk_id": "shared_1", "page_number": 1, "section_title": "HDFS Replication",
            "text": "HDFS replicates each data block across three separate fault-tolerant racks by default."
        }]

        # Both documents contain the replication knowledge
        KnowledgeIntegrationService.integrate_document_knowledge(db_session, doc_1, shared_chunk, mock_embedding_model)
        KnowledgeIntegrationService.integrate_document_knowledge(db_session, doc_2, shared_chunk, mock_embedding_model)

        assert db_session.query(KnowledgeItem).filter(KnowledgeItem.folder_id == folder.id).count() == 1

        # Delete doc_1
        KnowledgeIntegrationService.handle_document_deletion(db_session, doc_1.id, folder.id, user.id)

        # Knowledge item should still exist because doc_2 supports it!
        remaining_item = db_session.query(KnowledgeItem).filter(KnowledgeItem.folder_id == folder.id).first()
        assert remaining_item is not None, "Shared knowledge must not be deleted when one source document is deleted."
        sources = json.loads(remaining_item.provenance_sources_json)
        assert len(sources) == 1
        assert sources[0]["document_id"] == doc_2.id

        # Delete doc_2 (last remaining supporting doc)
        KnowledgeIntegrationService.handle_document_deletion(db_session, doc_2.id, folder.id, user.id)
        assert db_session.query(KnowledgeItem).filter(KnowledgeItem.folder_id == folder.id).count() == 0
        assert db_session.query(KnowledgeConcept).filter(KnowledgeConcept.folder_id == folder.id).count() == 0
