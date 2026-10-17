"""API endpoint for ATML Automated Quiz and Question Generation."""
import logging
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models import User
from backend.auth import get_current_user
from typing import Optional
from backend.schemas_quiz import QuizResponseSchema
from backend.services.atml.quiz_generator import QuizGenerator

logger = logging.getLogger("nalanda.quiz_routes")

router = APIRouter(prefix="/api/folders", tags=["quiz"])

_quiz_generator = QuizGenerator()


@router.get(
    "/{folder_id}/quiz",
    response_model=QuizResponseSchema,
    status_code=status.HTTP_200_OK,
    summary="Generate Academic Practice Quiz for Folder",
)
def get_folder_quiz(
    folder_id: str,
    num_questions: int = Query(default=10, ge=1, le=25, description="Number of questions to synthesize"),
    doc_ids: Optional[str] = Query(default=None, description="Comma-separated document IDs to filter by"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Generate multiple-choice practice questions strictly grounded in the folder's course materials.

    ATML Features Applied:
    - Selective unit/document generation when doc_ids is specified.
    - Inverted Seq2Seq Question Generation with FLAN-T5.
    - Vector Manifold Semantic Distractor Optimization with all-MiniLM-L6-v2 embeddings.
    - Verifiable page provenance and source-grounded explanations.
    """
    selected_doc_ids = [d.strip() for d in doc_ids.split(",") if d.strip()] if doc_ids else None
    try:
        quiz_result = _quiz_generator.generate_quiz_for_folder(
            db=db,
            current_user=current_user,
            folder_id=folder_id,
            num_questions=num_questions,
            document_ids=selected_doc_ids,
        )
        return quiz_result.to_dict()
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except Exception as exc:
        logger.error("Error generating quiz for folder %s: %s", folder_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate practice quiz: {exc}",
        )
