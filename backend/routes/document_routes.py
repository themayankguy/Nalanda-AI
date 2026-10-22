"""Document management and upload endpoints.

CRITICAL REQUIREMENTS:
- Strict user authorization: Users can only upload and view documents in their own folders.
- Extracted text, OCR output, raw Markdown, JSON extraction results, and intermediate
  processing files must remain internal to the backend.
- Endpoints return document-level metadata and processing status ONLY.
"""
import uuid
import shutil
from typing import List, Optional
from pathlib import Path

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    UploadFile,
    File,
    BackgroundTasks,
    status,
    Query,
)
from fastapi.security import HTTPAuthorizationCredentials
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

import logging
from backend.database import get_db
from backend.models import User, Folder, Document, DocumentIngestionLog
from backend.schemas import (
    DocumentMetadataResponse,
    DocumentUploadResponse,
    DocumentStatusResponse,
    DocumentIngestionSummaryResponse,
)
from backend.auth import get_current_user, security, verify_token_string
from backend.services.storage_service import (
    validate_file,
    get_document_paths,
    get_figure_file_path,
    MAX_FILE_SIZE_BYTES,
)
from backend.services.nalanda_extraction_service import process_document_extraction

logger = logging.getLogger("nalanda.document_routes")

router = APIRouter(tags=["documents"])



@router.post(
    "/api/folders/{folder_id}/documents/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def upload_document(
    folder_id: str,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Upload a PDF or supported image document into a folder and trigger extraction."""
    # 1. Verify folder exists and belongs to current user
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

    # 2. Validate filename and read file contents to verify size
    filename = file.filename or "uploaded_file"
    file_content = await file.read()
    file_size = len(file_content)

    is_valid, file_type, err_msg = validate_file(filename, file_size, file.content_type)
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=err_msg,
        )

    # 3. Generate unique document identifier and storage paths
    doc_id = str(uuid.uuid4())
    paths = get_document_paths(
        user_id=current_user.id,
        folder_id=folder.id,
        document_id=doc_id,
        filename=filename,
    )

    # 4. Save original file securely
    try:
        with open(paths["original_file_path"], "wb") as f:
            f.write(file_content)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save uploaded file to disk storage.",
        )

    # 5. Create database record
    document = Document(
        id=doc_id,
        folder_id=folder.id,
        user_id=current_user.id,
        original_filename=paths["safe_filename"],
        file_type=file_type,
        file_size_bytes=file_size,
        storage_path=str(paths["original_file_path"]),
        status="pending",
        error_message=None,
    )
    db.add(document)
    db.commit()
    db.refresh(document)

    # 6. Schedule extraction in background task only for formats with extraction support
    # HEIC/HEIF and PPTX are accepted inputs without implementing decoding/OCR yet.
    ext = Path(paths["safe_filename"]).suffix.lower()
    if ext in {".pdf", ".jpg", ".jpeg", ".png"}:
        background_tasks.add_task(process_document_extraction, document.id)
        msg = "Document uploaded successfully. Extraction started in background."
    else:
        msg = "Document uploaded successfully."

    return DocumentUploadResponse(
        id=document.id,
        folder_id=folder.id,
        original_filename=document.original_filename,
        status="pending",
        message=msg,
    )


@router.get(
    "/api/folders/{folder_id}/documents",
    response_model=List[DocumentMetadataResponse],
)
def list_folder_documents(
    folder_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List document metadata for documents inside a specific folder."""
    # Verify folder ownership
    folder = (
        db.query(Folder)
        .filter(Folder.id == folder_id, Folder.user_id == current_user.id)
        .first()
    )
    if not folder:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Folder not found or does not belong to you.",
        )

    documents = (
        db.query(Document)
        .filter(Document.folder_id == folder_id, Document.user_id == current_user.id)
        .order_by(Document.created_at.desc())
        .all()
    )

    # Return metadata only
    return [DocumentMetadataResponse.model_validate(d) for d in documents]


@router.get(
    "/api/documents/{document_id}/status",
    response_model=DocumentStatusResponse,
)
def get_document_status(
    document_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Retrieve current extraction status of a document."""
    document = (
        db.query(Document)
        .filter(Document.id == document_id, Document.user_id == current_user.id)
        .first()
    )
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found or does not belong to you.",
        )

    return DocumentStatusResponse.model_validate(document)


@router.post(
    "/api/documents/{document_id}/retry",
    response_model=DocumentStatusResponse,
)
def retry_document_extraction(
    document_id: str,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Retry extraction for a failed document."""
    document = (
        db.query(Document)
        .filter(Document.id == document_id, Document.user_id == current_user.id)
        .first()
    )
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found or does not belong to you.",
        )

    document.status = "pending"
    document.error_message = None
    db.commit()
    db.refresh(document)

    background_tasks.add_task(process_document_extraction, document.id)
    return DocumentStatusResponse.model_validate(document)


@router.get(
    "/api/documents/{document_id}/figures/{figure_filename}",
    summary="Retrieve extracted document diagram or figure",
)
def get_document_figure(
    document_id: str,
    figure_filename: str,
    token: Optional[str] = Query(None),
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: Session = Depends(get_db),
):
    """Serve extracted figure image with user ownership verification."""
    token_str = None
    if credentials and credentials.credentials:
        token_str = credentials.credentials
    elif token:
        token_str = token

    if not token_str:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication credentials were not provided.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    current_user = verify_token_string(token_str, db)
    document = (
        db.query(Document)
        .filter(Document.id == document_id, Document.user_id == current_user.id)
        .first()
    )
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found or does not belong to you.",
        )

    fig_path = get_figure_file_path(
        user_id=current_user.id,
        folder_id=document.folder_id,
        document_id=document.id,
        filename=figure_filename,
    )
    if not fig_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Figure not found for this document.",
        )

    media_type = "image/png"
    if fig_path.suffix.lower() in [".jpg", ".jpeg"]:
        media_type = "image/jpeg"
    elif fig_path.suffix.lower() == ".webp":
        media_type = "image/webp"

    return FileResponse(str(fig_path), media_type=media_type)


@router.get(
    "/api/documents/{document_id}/file",
    summary="View and stream original uploaded PDF or document",
)
def get_document_file(
    document_id: str,
    token: Optional[str] = Query(None),
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: Session = Depends(get_db),
):
    """Serve the original uploaded PDF with user ownership verification and inline disposition."""
    token_str = None
    if credentials and credentials.credentials:
        token_str = credentials.credentials
    elif token:
        token_str = token

    if not token_str:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication credentials were not provided.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    current_user = verify_token_string(token_str, db)
    document = (
        db.query(Document)
        .filter(Document.id == document_id, Document.user_id == current_user.id)
        .first()
    )
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found or does not belong to you.",
        )

    if not document.storage_path or not Path(document.storage_path).exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Original document file not found on disk storage.",
        )

    file_path = Path(document.storage_path)
    ext = file_path.suffix.lower()
    media_type = "application/pdf"
    if ext in [".jpg", ".jpeg"]:
        media_type = "image/jpeg"
    elif ext == ".png":
        media_type = "image/png"
    elif ext == ".pptx":
        media_type = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
    elif ext == ".heic":
        media_type = "image/heic"
    elif ext == ".heif":
        media_type = "image/heif"

    return FileResponse(
        str(file_path),
        media_type=media_type,
        filename=document.original_filename,
        content_disposition_type="inline",
    )


@router.get(
    "/api/documents/{document_id}/ingestion-summary",
    response_model=DocumentIngestionSummaryResponse,
)
def get_document_ingestion_summary(
    document_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Retrieve the knowledge integration audit summary for a document."""
    doc = (
        db.query(Document)
        .filter(Document.id == document_id, Document.user_id == current_user.id)
        .first()
    )
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found or does not belong to you.",
        )
    log = (
        db.query(DocumentIngestionLog)
        .filter(DocumentIngestionLog.document_id == document_id)
        .first()
    )
    if not log:
        return DocumentIngestionSummaryResponse(
            document_id=doc.id,
            folder_id=doc.folder_id,
            items_matched=0,
            concepts_enriched=0,
            concepts_created=0,
            duplicates_skipped=0,
            conflicts_flagged=0,
            summary_message="No ingestion log available for this document.",
        )
    return DocumentIngestionSummaryResponse.model_validate(log)


@router.delete(
    "/api/documents/{document_id}",
    status_code=status.HTTP_200_OK,
)
def delete_document(
    document_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Safely delete a document, updating provenance and preserving independently supported concepts."""
    document = (
        db.query(Document)
        .filter(Document.id == document_id, Document.user_id == current_user.id)
        .first()
    )
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found or does not belong to you.",
        )

    folder_id = document.folder_id
    filename = document.original_filename

    # 1. Update knowledge provenance and concepts
    try:
        from backend.services.NLP.knowledge_integration_service import KnowledgeIntegrationService
        KnowledgeIntegrationService.handle_document_deletion(db, document.id, folder_id, current_user.id)
    except Exception as exc:
        logger.warning("Error updating knowledge on document deletion: %s", exc)

    # 2. Update vector index
    try:
        from backend.services.storage_service import BASE_STORAGE_DIR
        from backend.services.NLP.vector_index_service import get_folder_index_dir, FolderVectorIndex
        index_dir = get_folder_index_dir(BASE_STORAGE_DIR, current_user.id, folder_id)
        f_idx = FolderVectorIndex(index_dir)
        f_idx.remove_document(document.id)
    except Exception as exc:
        logger.warning("Error removing document from vector index: %s", exc)


    # 4. Remove physical storage if exists
    try:
        if document.storage_path and Path(document.storage_path).exists():
            p = Path(document.storage_path)
            doc_dir = p.parent.parent
            if doc_dir.exists() and doc_dir.name == document.id:
                shutil.rmtree(doc_dir, ignore_errors=True)
            elif p.exists():
                p.unlink(missing_ok=True)
    except Exception as exc:
        logger.warning("Error deleting document files: %s", exc)

    # 5. Delete from DB
    db.delete(document)
    db.commit()

    return {"message": f"Document '{filename}' successfully deleted.", "document_id": document_id}


