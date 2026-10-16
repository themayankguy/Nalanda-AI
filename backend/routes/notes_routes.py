"""API endpoint for ATML Automated Comprehensive Notes and Study Guide Generation."""
import logging
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models import User
from backend.auth import get_current_user
from backend.schemas_notes import NotesResponseSchema
from backend.services.atml.notes_generator import NotesGenerator

logger = logging.getLogger("nalanda.notes_routes")

router = APIRouter(prefix="/api/folders", tags=["notes"])

_notes_generator = NotesGenerator()


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
