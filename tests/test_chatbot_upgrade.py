"""Comprehensive test suite for Nalanda AI Chatbot Upgrade:
- Multi-chunk context aggregation
- Hierarchical concept gathering (e.g. all types of charts)
- Text normalization (OCR cleanup, line unwrapping, Markdown tables)
- Multi-turn context and pronoun resolution
- Persistent conversation storage and security isolation
"""
import json
import uuid
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.database import Base, engine, SessionLocal
from backend.models import User, Folder, Document, DocumentExtraction, Conversation, ChatMessage
from backend.services.NLP.text_normalizer import (
    normalize_ocr_text,
    format_types_response,
    format_definition_response,
    format_comparison_table,
    deduplicate_sentences_across_chunks,
    detect_query_presentation_type,
)
from backend.services.NLP.hierarchical_concept_gatherer import HierarchicalConceptGatherer
from backend.services.NLP.chat_context_service import ChatContextService
from backend.services.NLP.chat_service import (
    create_conversation,
    list_conversations,
    get_conversation_detail,
    delete_conversation,
    post_user_message_and_respond,
)


@pytest.fixture(autouse=True)
def setup_db():
    Base.metadata.create_all(bind=engine)
    yield


@pytest.fixture
def client():
    return TestClient(app)


# =========================================================================
# 1. Tests: Text Normalization and Markdown Formatting
# =========================================================================

def test_ocr_hyphenation_and_mid_sentence_line_unwrapping():
    """Verify that mid-sentence line breaks and hyphenated words are healed without destroying structure."""
    raw_ocr = (
        "Data visual-\n"
        "ization is the pro-\n"
        "cess of converting raw or processed data into a visual\n"
        "representation such as charts and graphs.\n\n"
        "### Types of Visualizations\n"
        "- Line Chart: Shows\n"
        "trends over time.\n"
        "- Bar Chart: Compares\n"
        "categories of data.\n"
    )

    cleaned = normalize_ocr_text(raw_ocr)

    # Hyphenated words should be unified
    assert "visualization" in cleaned
    assert "process" in cleaned
    # Headers and list items should retain their structure
    assert "### Types of Visualizations" in cleaned
    assert "- Line Chart: Shows trends over time." in cleaned
    assert "- Bar Chart: Compares categories of data." in cleaned


def test_markdown_tables_and_code_blocks_preserved_during_normalization():
    """Ensure tables and code blocks remain completely unaltered by line unwrapping."""
    raw = (
        "Here is the comparison table:\n\n"
        "| Feature | HDFS | Local FS |\n"
        "| :--- | :--- | :--- |\n"
        "| Redundancy | Triple replication | None |\n"
        "| Scaling | Horizontal | Vertical |\n\n"
        "Run this command:\n"
        "```bash\n"
        "docker run -d -p 8080:80 nginx\n"
        "```\n"
    )
    cleaned = normalize_ocr_text(raw)
    assert "| Feature | HDFS | Local FS |" in cleaned
    assert "| Redundancy | Triple replication | None |" in cleaned
    assert "```bash\ndocker run -d -p 8080:80 nginx\n```" in cleaned


def test_deduplicate_sentences_across_chunks():
    """Verify duplicate sentences across multiple retrieved chunks are pruned."""
    chunk1 = "Apache Spark is a distributed processing framework. It supports in-memory computing."
    chunk2 = "Apache Spark is a distributed processing framework. It provides high-level APIs in Python."
    deduped = deduplicate_sentences_across_chunks([chunk1, chunk2])

    assert len(deduped) == 2
    # Second chunk should retain unique sentence but drop verbatim first sentence
    assert "It provides high-level APIs in Python." in deduped[1]
    assert deduped[1].count("Apache Spark is a distributed processing framework") == 0


def test_question_presentation_classification():
    """Verify query presentation classification into definition, types, comparison, procedure."""
    assert detect_query_presentation_type("What is a Scatter Plot?") == "definition"
    assert detect_query_presentation_type("What are the types of charts?") == "types"
    assert detect_query_presentation_type("Compare HDFS vs Local FS") == "comparison"
    assert detect_query_presentation_type("What are the steps to create a visualization?") == "procedure"


# =========================================================================
# 2. Tests: Hierarchical Concept Gathering
# =========================================================================

def test_hierarchical_gatherer_identifies_all_chart_types():
    """Verify gather_sibling_concepts extracts all charts across distinct chunks."""
    chunks = [
        {
            "chunk_index": 0,
            "section_title": "1. Need for Visualization",
            "page_number": 1,
            "source_filename": "UNIT 5 BDA.pdf",
            "text": "Data visualization converts data into visual formats like charts. For example, a line chart shows changes over time.",
        },
        {
            "chunk_index": 3,
            "section_title": "3.1 Line Chart",
            "page_number": 3,
            "source_filename": "UNIT 5 BDA.pdf",
            "text": "A line chart represents data points connected by lines. It is mainly used to show trends.",
        },
        {
            "chunk_index": 4,
            "section_title": "3.2 Histogram",
            "page_number": 3,
            "source_filename": "UNIT 5 BDA.pdf",
            "text": "A histogram represents the frequency distribution of numerical data.",
        },
        {
            "chunk_index": 5,
            "section_title": "3.3 Bar Chart",
            "page_number": 4,
            "source_filename": "UNIT 5 BDA.pdf",
            "text": "A bar chart represents categorical data using rectangular bars.",
        },
        {
            "chunk_index": 6,
            "section_title": "3.4 Pie Chart",
            "page_number": 4,
            "source_filename": "UNIT 5 BDA.pdf",
            "text": "A pie chart represents data as slices of a circle.",
        },
        {
            "chunk_index": 7,
            "section_title": "3.5 Heat Map",
            "page_number": 5,
            "source_filename": "UNIT 5 BDA.pdf",
            "text": "A heat map uses colors to represent values or intensity.",
        },
        {
            "chunk_index": 8,
            "section_title": "3.6 Scatter Plot",
            "page_number": 5,
            "source_filename": "UNIT 5 BDA.pdf",
            "text": "A scatter plot represents relationships between two numerical variables.",
        },
        {
            "chunk_index": 9,
            "section_title": "3.7 Box Plot",
            "page_number": 6,
            "source_filename": "UNIT 5 BDA.pdf",
            "text": "A box plot summarizes numerical data using quartiles and outliers.",
        },
    ]

    is_hier = HierarchicalConceptGatherer.is_hierarchical_query("What are the types of charts?")
    assert is_hier is True

    parent = HierarchicalConceptGatherer.extract_parent_target("What are the types of charts?")
    assert "chart" in parent

    siblings = HierarchicalConceptGatherer.gather_sibling_concepts(parent, chunks)
    sibling_names = {s["name"].lower() for s in siblings}

    # Must contain not just Line and Bar, but all 7 distinct chart types!
    assert "line chart" in sibling_names
    assert "histogram" in sibling_names
    assert "bar chart" in sibling_names
    assert "pie chart" in sibling_names
    assert "heat map" in sibling_names
    assert "scatter plot" in sibling_names
    assert "box plot" in sibling_names

    formatted_response = format_types_response(parent.title(), siblings)
    assert "### Types of" in formatted_response
    assert "Line Chart" in formatted_response
    assert "Histogram" in formatted_response
    assert "Box Plot" in formatted_response
    assert "Page 3" in formatted_response


# =========================================================================
# 3. Tests: Multi-Turn Context and Pronoun Resolution
# =========================================================================

def test_multiturn_pronoun_resolution():
    """Verify 'its advantages' resolves to antecedent topic 'Hadoop Distributed File System'."""
    history = [
        {"role": "user", "content": "Explain Hadoop Distributed File System."},
        {"role": "assistant", "content": "**Hadoop Distributed File System (HDFS)** is a distributed file system designed to run on commodity hardware."},
    ]

    resolved_q, antecedent = ChatContextService.resolve_followup_query("What are its advantages?", history)
    assert "hadoop distributed file system" in resolved_q.lower() or "hdfs" in resolved_q.lower()
    assert "advantages" in resolved_q.lower()


def test_multiturn_ordinal_point_resolution():
    """Verify 'Explain the second point' extracts point 2 from previous assistant response."""
    history = [
        {"role": "user", "content": "What are the components of HDFS?"},
        {
            "role": "assistant",
            "content": (
                "Here are the primary components:\n"
                "1. **NameNode** - The master server that manages the file system namespace.\n"
                "2. **DataNode** - The worker node that stores actual data blocks.\n"
                "3. **Secondary NameNode** - Performs periodic checkpoints.\n"
            ),
        },
    ]

    resolved_q, antecedent = ChatContextService.resolve_followup_query("Explain the second point", history)
    assert "datanode" in resolved_q.lower()


# =========================================================================
# 4. Tests: Persistent Conversation Storage & Security Isolation
# =========================================================================

def test_conversation_persistence_and_user_isolation(client):
    """Verify conversation lifecycle and user access isolation."""
    db = SessionLocal()
    try:
        # Create user 1 and user 2
        u1 = User(
            id=str(uuid.uuid4()),
            email=f"user1_{uuid.uuid4()}@nalanda.test",
            username=f"u1_{uuid.uuid4().hex[:6]}",
            password_hash="fakehash",
        )
        u2 = User(
            id=str(uuid.uuid4()),
            email=f"user2_{uuid.uuid4()}@nalanda.test",
            username=f"u2_{uuid.uuid4().hex[:6]}",
            password_hash="fakehash",
        )
        db.add_all([u1, u2])
        db.commit()

        # Create folder for user 1
        folder1 = Folder(
            id=str(uuid.uuid4()),
            name="BDA Unit 5",
            user_id=u1.id,
        )
        db.add(folder1)
        db.commit()

        # 1. Create conversation for user 1
        conv_summary = create_conversation(db, u1, folder1.id, title="Unit 5 Visualizations")
        assert conv_summary.id is not None
        assert conv_summary.title == "Unit 5 Visualizations"

        # 2. List conversations for user 1
        conv_list = list_conversations(db, u1, folder1.id)
        assert len(conv_list) == 1
        assert conv_list[0].id == conv_summary.id

        # 3. Security Isolation: User 2 cannot access User 1's conversation
        with pytest.raises(Exception):
            get_conversation_detail(db, u2, conv_summary.id)

        # 4. Security Isolation: User 2 cannot list User 1's folder conversations
        with pytest.raises(Exception):
            list_conversations(db, u2, folder1.id)

        # 5. Delete conversation
        del_res = delete_conversation(db, u1, conv_summary.id)
        assert del_res["status"] == "deleted"

        # After deletion, list should be empty
        conv_list_after = list_conversations(db, u1, folder1.id)
        assert len(conv_list_after) == 0

    finally:
        db.close()
