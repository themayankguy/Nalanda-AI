"""Document management and upload endpoints.

CRITICAL REQUIREMENTS:
- Strict user authorization: Users can only upload and view documents in their own folders.
- Extracted text, OCR output, raw Markdown, JSON extraction results, and intermediate
  processing files must remain internal to the backend.
- Endpoints return document-level metadata and processing status ONLY.
"""
import uuid
import shutil
from typing import List
from pathlib import Path

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    UploadFile,
    File,
    BackgroundTasks,
    status,
)
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models import User, Folder, Document
from backend.schemas import (
    DocumentMetadataResponse,
    DocumentUploadResponse,
    DocumentStatusResponse,
)
from backend.auth import get_current_user
from backend.services.storage_service import (
    validate_file,
    get_document_paths,
    MAX_FILE_SIZE_BYTES,
)
from backend.services.nalanda_extraction_service import process_document_extraction

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

    is_valid, file_type, err_msg = validate_file(filename, file_size)
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

    # 6. Schedule extraction in background task
    background_tasks.add_task(process_document_extraction, document.id)

    return DocumentUploadResponse(
        id=document.id,
        folder_id=folder.id,
        original_filename=document.original_filename,
        status="pending",
        message="Document uploaded successfully. Extraction started in background.",
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
