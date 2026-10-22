"""SQLAlchemy models for Nalanda AI."""
import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Integer, DateTime, ForeignKey, Text, Boolean
from sqlalchemy.orm import relationship
from backend.database import Base


def utcnow():
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(100), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=utcnow, nullable=False)

    folders = relationship("Folder", back_populates="user", cascade="all, delete-orphan")
    documents = relationship("Document", back_populates="user", cascade="all, delete-orphan")
    tokens = relationship("SessionToken", back_populates="user", cascade="all, delete-orphan")


class SessionToken(Base):
    __tablename__ = "session_tokens"

    token = Column(String(64), primary_key=True)
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    created_at = Column(DateTime, default=utcnow, nullable=False)
    expires_at = Column(DateTime, nullable=True)

    user = relationship("User", back_populates="tokens")


class Folder(Base):
    __tablename__ = "folders"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False)
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    created_at = Column(DateTime, default=utcnow, nullable=False)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow, nullable=False)

    user = relationship("User", back_populates="folders")
    documents = relationship("Document", back_populates="folder", cascade="all, delete-orphan")
    concepts = relationship("KnowledgeConcept", back_populates="folder", cascade="all, delete-orphan")


class Document(Base):
    __tablename__ = "documents"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    folder_id = Column(String(36), ForeignKey("folders.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    original_filename = Column(String(255), nullable=False)
    file_type = Column(String(20), nullable=False)  # "pdf", "image", etc.
    file_size_bytes = Column(Integer, nullable=False, default=0)
    storage_path = Column(String(1024), nullable=False)  # Absolute or relative path to original file
    status = Column(String(30), nullable=False, default="pending")  # pending, processing, completed, failed
    error_message = Column(String(500), nullable=True)
    created_at = Column(DateTime, default=utcnow, nullable=False)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow, nullable=False)

    user = relationship("User", back_populates="documents")
    folder = relationship("Folder", back_populates="documents")
    extraction = relationship("DocumentExtraction", back_populates="document", uselist=False, cascade="all, delete-orphan")
    ingestion_log = relationship("DocumentIngestionLog", back_populates="document", uselist=False, cascade="all, delete-orphan")


class DocumentExtraction(Base):
    """Internal backend storage for extracted content and metadata.
    
    CRITICAL CONSTRAINT: Extracted text, markdown, and raw JSON outputs are kept
    strictly internal to backend storage and are NEVER returned in public document APIs.
    """
    __tablename__ = "document_extractions"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id = Column(String(36), ForeignKey("documents.id", ondelete="CASCADE"), unique=True, nullable=False, index=True)
    full_text = Column(Text, nullable=True)
    markdown_content = Column(Text, nullable=True)
    json_storage_path = Column(String(1024), nullable=True)
    output_directory = Column(String(1024), nullable=True)
    table_files_json = Column(Text, nullable=True)  # JSON-encoded list of table HTML paths
    metadata_json = Column(Text, nullable=True)  # JSON-encoded raw metadata
    total_pages = Column(Integer, default=0)
    total_characters = Column(Integer, default=0)
    total_tables = Column(Integer, default=0)
    created_at = Column(DateTime, default=utcnow, nullable=False)

    document = relationship("Document", back_populates="extraction")


class FolderNoteCache(Base):
    """Caches generated academic study guides per folder for instantaneous retrieval."""
    __tablename__ = "folder_note_caches"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    folder_id = Column(String(36), ForeignKey("folders.id", ondelete="CASCADE"), unique=True, nullable=False, index=True)
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    markdown_content = Column(Text, nullable=False)
    sections_json = Column(Text, nullable=False)  # JSON-encoded list of section dicts
    documents_hash = Column(String(64), nullable=False)  # SHA-256 fingerprint of completed document states
    total_sections = Column(Integer, default=0)
    created_at = Column(DateTime, default=utcnow, nullable=False)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow, nullable=False)

    folder = relationship("Folder", backref="note_cache")


class Conversation(Base):
    """Represents a persistent multi-turn chat session scoped to a user and folder."""
    __tablename__ = "conversations"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    folder_id = Column(String(36), ForeignKey("folders.id", ondelete="CASCADE"), nullable=False, index=True)
    title = Column(String(255), nullable=False, default="New Conversation")
    created_at = Column(DateTime, default=utcnow, nullable=False)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow, nullable=False)

    user = relationship("User", backref="conversations")
    folder = relationship("Folder", backref="conversations")
    messages = relationship("ChatMessage", back_populates="conversation", cascade="all, delete-orphan", order_by="ChatMessage.created_at")


class ChatMessage(Base):
    """Represents an individual message turn within a conversation."""
    __tablename__ = "chat_messages"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    conversation_id = Column(String(36), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False, index=True)
    role = Column(String(20), nullable=False)  # "user" or "assistant"
    content = Column(Text, nullable=False)
    direct_answer = Column(Text, nullable=True)
    primary_figure_json = Column(Text, nullable=True)
    primary_table_json = Column(Text, nullable=True)
    citations_json = Column(Text, nullable=True)
    refinement_metadata_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=utcnow, nullable=False)

    conversation = relationship("Conversation", back_populates="messages")


class KnowledgeConcept(Base):
    """Represents a canonical academic or technical concept synthesized within a folder."""
    __tablename__ = "knowledge_concepts"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    folder_id = Column(String(36), ForeignKey("folders.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    normalized_name = Column(String(255), nullable=False, index=True)
    summary = Column(Text, nullable=True)
    created_at = Column(DateTime, default=utcnow, nullable=False)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow, nullable=False)

    folder = relationship("Folder", back_populates="concepts")
    items = relationship("KnowledgeItem", back_populates="concept", cascade="all, delete-orphan", order_by="KnowledgeItem.created_at")


class KnowledgeItem(Base):
    """Represents an atomic factual statement, definition, formula, or table snippet under a concept."""
    __tablename__ = "knowledge_items"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    concept_id = Column(String(36), ForeignKey("knowledge_concepts.id", ondelete="CASCADE"), nullable=False, index=True)
    folder_id = Column(String(36), ForeignKey("folders.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    document_id = Column(String(36), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    chunk_id = Column(String(255), nullable=True)
    subtopic = Column(String(255), nullable=False, default="General")
    content = Column(Text, nullable=False)
    normalized_hash = Column(String(64), nullable=False, index=True)
    page_number = Column(Integer, nullable=True)
    source_filename = Column(String(255), nullable=False)
    provenance_sources_json = Column(Text, nullable=True)  # JSON-encoded list of [{"document_id", "filename", "page", "chunk_id"}]
    is_conflict = Column(Boolean, default=False, nullable=False)
    conflict_details = Column(Text, nullable=True)  # JSON-encoded description of contradiction
    created_at = Column(DateTime, default=utcnow, nullable=False)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow, nullable=False)

    concept = relationship("KnowledgeConcept", back_populates="items")
    document = relationship("Document")


class DocumentIngestionLog(Base):
    """Persists integration audit counts and summary whenever a document is ingested."""
    __tablename__ = "document_ingestion_logs"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id = Column(String(36), ForeignKey("documents.id", ondelete="CASCADE"), unique=True, nullable=False, index=True)
    folder_id = Column(String(36), ForeignKey("folders.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    items_matched = Column(Integer, default=0, nullable=False)
    concepts_enriched = Column(Integer, default=0, nullable=False)
    concepts_created = Column(Integer, default=0, nullable=False)
    duplicates_skipped = Column(Integer, default=0, nullable=False)
    conflicts_flagged = Column(Integer, default=0, nullable=False)
    summary_message = Column(String(500), nullable=True)
    details_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=utcnow, nullable=False)

    document = relationship("Document", back_populates="ingestion_log")

