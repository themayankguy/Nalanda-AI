"""High-level semantic search orchestration service for Nalanda AI.

CRITICAL REQUIREMENTS:
- Strict authorization: every search query MUST be scoped to an authorized folder
  belonging to the authenticated user.
- Extracted text, chunk content, and raw disk paths are hidden and NEVER returned.
- Returns only approved source metadata: document_id, filename, page_number,
  section_title, and calibrated relevance_score.
"""
import logging
from pathlib import Path
from typing import List, Optional
from sqlalchemy.orm import Session
from fastapi import HTTPException, status

from backend.models import User, Folder, Document, DocumentExtraction
from backend.services.storage_service import BASE_STORAGE_DIR
from backend.services.document_chunking_service import chunk_document_data
from backend.services.vector_index_service import (
    get_folder_index_dir,
    FolderVectorIndex,
    get_embedding_model,
)
from backend.schemas_search import (
    SemanticSearchRequest,
    SearchResultItem,
    SemanticSearchResponse,
)

logger = logging.getLogger("nalanda.semantic_search")


def get_or_create_folder_index(user_id: str, folder_id: str) -> FolderVectorIndex:
    """Retrieve isolated vector index instance for a specific user and folder."""
    index_dir = get_folder_index_dir(BASE_STORAGE_DIR, user_id, folder_id)
    return FolderVectorIndex(index_dir)


def index_single_document(db: Session, document: Document, index: Optional[FolderVectorIndex] = None, model=None) -> int:
    """Index an extracted document into its folder's vector index.

    Reads structured JSON from extractor output or fallback text from DB.
    """
    if document.status != "completed":
        logger.info("Skipping document %s with non-completed status '%s'", document.id, document.status)
        return 0

    extraction = (
        db.query(DocumentExtraction)
        .filter(DocumentExtraction.document_id == document.id)
        .first()
    )

    json_path = None
    fallback_text = None
    if extraction:
        if extraction.json_storage_path:
            p = Path(extraction.json_storage_path)
            if p.exists():
                json_path = p
        fallback_text = extraction.full_text

    chunks = chunk_document_data(
        document_id=document.id,
        user_id=document.user_id,
        folder_id=document.folder_id,
        source_filename=document.original_filename,
        json_path=json_path,
        fallback_text=fallback_text,
    )

    if not chunks:
        logger.info("No chunks generated for document %s", document.id)
        return 0

    if index is None:
        index = get_or_create_folder_index(document.user_id, document.folder_id)

    added_count = index.add_document_chunks(chunks, model=model)
    logger.info("Indexed %d chunks for document %s in folder %s", added_count, document.id, document.folder_id)
    return added_count


def sync_folder_index(db: Session, user_id: str, folder_id: str, model=None) -> FolderVectorIndex:
    """Ensure all completed documents in the folder are indexed in its vector index."""
    index = get_or_create_folder_index(user_id, folder_id)

    # Get all completed documents for this user and folder
    docs = (
        db.query(Document)
        .filter(
            Document.folder_id == folder_id,
            Document.user_id == user_id,
            Document.status == "completed",
        )
        .all()
    )

    indexed_doc_ids = {c.get("document_id") for c in index.chunks if c.get("document_id")}

    for doc in docs:
        if doc.id not in indexed_doc_ids:
            index_single_document(db, doc, index=index, model=model)

    return index


def execute_semantic_search(
    db: Session,
    current_user: User,
    request: SemanticSearchRequest,
) -> SemanticSearchResponse:
    """Execute natural-language semantic retrieval strictly scoped to the user's folder.

    CRITICAL ISOLATION:
    1. Rejects missing or blank folder_id.
    2. Validates that the target folder belongs to the authenticated user.
    3. Searches ONLY the candidate vectors belonging to that specific folder.
    4. Strips any internal extracted text from the output.
    """
    clean_query = request.query.strip()
    if not clean_query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Search query cannot be empty.",
        )

    folder_id = request.folder_id.strip()
    if not folder_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A target folder_id is required. Global searches are not permitted.",
        )

    # Validate folder existence and user ownership
    folder = (
        db.query(Folder)
        .filter(Folder.id == folder_id, Folder.user_id == current_user.id)
        .first()
    )
    if not folder:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Target folder not found or does not belong to you.",
        )

    # Sync and search folder's isolated vector index
    try:
        index = sync_folder_index(db, current_user.id, folder.id)
        raw_matches = index.search(query=clean_query, top_k=request.top_k)
    except RuntimeError as exc:
        logger.error("Embedding or indexing failure during search: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Semantic search service is temporarily unavailable.",
        ) from exc

    results: List[SearchResultItem] = []
    for chunk, relevance in raw_matches:
        results.append(
            SearchResultItem(
                document_id=chunk["document_id"],
                source_filename=chunk["source_filename"],
                page_number=chunk.get("page_number"),
                section_title=chunk.get("section_title"),
                relevance_score=relevance,
            )
        )

    return SemanticSearchResponse(
        query=clean_query,
        folder_id=folder.id,
        total_results=len(results),
        results=results,
    )
