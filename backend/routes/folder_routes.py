from datetime import datetime, timedelta, timezone
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import func

from backend.database import get_db
from backend.models import User, Folder, Document
from backend.schemas import FolderCreateRequest, FolderResponse
from backend.auth import get_current_user

router = APIRouter(prefix="/api/folders", tags=["folders"])


@router.post("", response_model=FolderResponse, status_code=status.HTTP_201_CREATED)
def create_folder(
    request: FolderCreateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Create a new folder for the authenticated user."""
    name = request.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Folder name cannot be blank.")

    # Deduplicate rapid repeated submissions (< 3 seconds) for identical folder name
    recent_cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(seconds=3)
    existing_recent = (
        db.query(Folder)
        .filter(
            Folder.user_id == current_user.id,
            Folder.name == name,
            Folder.created_at >= recent_cutoff,
        )
        .order_by(Folder.created_at.desc())
        .first()
    )
    if existing_recent:
        return FolderResponse(
            id=existing_recent.id,
            name=existing_recent.name,
            user_id=existing_recent.user_id,
            document_count=0,
            created_at=existing_recent.created_at,
            updated_at=existing_recent.updated_at,
        )

    folder = Folder(name=name, user_id=current_user.id)
    db.add(folder)
    db.commit()
    db.refresh(folder)

    return FolderResponse(
        id=folder.id,
        name=folder.name,
        user_id=folder.user_id,
        document_count=0,
        created_at=folder.created_at,
        updated_at=folder.updated_at,
    )


@router.get("", response_model=List[FolderResponse])
def list_folders(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List all folders belonging to the authenticated user with document counts."""
    # Query folders with document counts
    results = (
        db.query(Folder, func.count(Document.id).label("doc_count"))
        .outerjoin(Document, Document.folder_id == Folder.id)
        .filter(Folder.user_id == current_user.id)
        .group_by(Folder.id)
        .order_by(Folder.created_at.desc())
        .all()
    )

    folders_out = []
    for folder, doc_count in results:
        folders_out.append(
            FolderResponse(
                id=folder.id,
                name=folder.name,
                user_id=folder.user_id,
                document_count=doc_count,
                created_at=folder.created_at,
                updated_at=folder.updated_at,
            )
        )
    return folders_out


@router.get("/{folder_id}", response_model=FolderResponse)
def get_folder(
    folder_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get single folder details after verifying user ownership."""
    folder = (
        db.query(Folder)
        .filter(Folder.id == folder_id, Folder.user_id == current_user.id)
        .first()
    )
    if not folder:
        raise HTTPException(status_code=404, detail="Folder not found.")

    doc_count = db.query(func.count(Document.id)).filter(Document.folder_id == folder.id).scalar() or 0

    return FolderResponse(
        id=folder.id,
        name=folder.name,
        user_id=folder.user_id,
        document_count=doc_count,
        created_at=folder.created_at,
        updated_at=folder.updated_at,
    )


@router.delete("/{folder_id}", status_code=status.HTTP_200_OK)
def delete_folder(
    folder_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Permanently delete a knowledge folder, all associated documents, and storage on disk."""
    import shutil
    import logging
    from backend.services.storage_service import BASE_STORAGE_DIR
    from backend.models import FolderNoteCache, Conversation

    logger = logging.getLogger("nalanda.folder_routes")
    folder = (
        db.query(Folder)
        .filter(Folder.id == folder_id, Folder.user_id == current_user.id)
        .first()
    )
    if not folder:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Folder not found.")

    folder_name = folder.name

    # Explicitly clean up related caches and conversations before deleting folder
    try:
        db.query(FolderNoteCache).filter(FolderNoteCache.folder_id == folder_id).delete()
        db.query(Conversation).filter(Conversation.folder_id == folder_id).delete()
    except Exception as exc:
        logger.debug("Cascade pre-cleanup non-blocking: %s", exc)

    db.delete(folder)
    db.commit()

    # Clean up physical storage on disk if present
    folder_storage_dir = BASE_STORAGE_DIR / "users" / str(current_user.id) / "folders" / str(folder_id)
    if folder_storage_dir.exists():
        try:
            shutil.rmtree(folder_storage_dir, ignore_errors=True)
        except Exception as exc:
            logger.warning("Could not delete folder storage dir %s: %s", folder_storage_dir, exc)

    return {"message": f"Folder '{folder_name}' successfully deleted.", "folder_id": folder_id}
