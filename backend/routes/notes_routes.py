"""API endpoint for ATML Automated Comprehensive Notes and Study Guide Generation."""
import logging
from typing import Optional
from pydantic import BaseModel
from fastapi import APIRouter, Depends, HTTPException, status, Response
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models import User, Folder
from backend.auth import get_current_user
from backend.schemas_notes import NotesResponseSchema
from backend.services.atml.notes_generator import NotesGenerator
from backend.services.atml.pdf_exporter import NotesPdfExporter

logger = logging.getLogger("nalanda.notes_routes")

router = APIRouter(prefix="/api/folders", tags=["notes"])

_notes_generator = NotesGenerator()
_pdf_exporter = NotesPdfExporter()


class ExportPdfRequest(BaseModel):
    markdown_content: Optional[str] = None


@router.get(
    "/{folder_id}/notes",
    response_model=NotesResponseSchema,
    status_code=status.HTTP_200_OK,
    summary="Generate Comprehensive Academic Study Notes for Folder",
)
def get_folder_notes(
    folder_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Synthesize a complete, multi-section comprehensive academic study guide grounded in the folder's lecture materials.

    ATML Features Applied:
    - Executive overview & learning scope synthesis via FLAN-T5.
    - Pedagogical fact-anchor extraction with exact page citations.
    - Comparative analysis matrix (structured comparison tables).
    - Sequential workflows & procedural architectures.
    - High-yield examination revision points.
    """
    try:
        notes_result = _notes_generator.generate_notes_for_folder(
            db=db,
            current_user=current_user,
            folder_id=folder_id,
        )
        return notes_result.to_dict()
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except Exception as exc:
        logger.error("Error generating notes for folder %s: %s", folder_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate study guide: {exc}",
        )


@router.post(
    "/{folder_id}/notes/export-pdf",
    status_code=status.HTTP_200_OK,
    summary="Download Academic Study Notes as PDF Document",
)
def export_folder_notes_pdf(
    folder_id: str,
    payload: Optional[ExportPdfRequest] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Export the study guide for a folder directly as a formatted, paginated academic PDF download."""
    folder = (
        db.query(Folder)
        .filter(Folder.id == folder_id, Folder.user_id == current_user.id)
        .first()
    )
    if not folder:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Folder '{folder_id}' not found.",
        )

    markdown_text = payload.markdown_content if payload and payload.markdown_content else None
    if not markdown_text:
        notes_result = _notes_generator.generate_notes_for_folder(
            db=db,
            current_user=current_user,
            folder_id=folder_id,
        )
        markdown_text = notes_result.markdown_content

    try:
        pdf_bytes = _pdf_exporter.export_markdown_to_pdf_bytes(
            markdown_text=markdown_text,
            folder_name=folder.name,
        )
        safe_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in folder.name)
        filename = f"Study_Guide_{safe_name}.pdf"

        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Content-Type": "application/pdf",
            },
        )
    except Exception as exc:
        logger.error("Error exporting PDF for folder %s: %s", folder_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate PDF document: {exc}",
        )


@router.get(
    "/{folder_id}/notes/pdf",
    status_code=status.HTTP_200_OK,
    summary="Direct Download of Academic Study Notes as PDF",
)
def download_folder_notes_pdf(
    folder_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Direct GET endpoint to download study guide PDF."""
    return export_folder_notes_pdf(
        folder_id=folder_id,
        payload=None,
        current_user=current_user,
        db=db,
    )

