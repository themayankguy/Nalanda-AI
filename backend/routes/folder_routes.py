"""Folder management API endpoints."""
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
