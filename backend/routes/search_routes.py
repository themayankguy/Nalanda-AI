"""Semantic search API endpoints for Nalanda AI."""
from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models import User
from backend.auth import get_current_user
from backend.schemas_search import SemanticSearchRequest, SemanticSearchResponse
from backend.services.semantic_search_service import execute_semantic_search

router = APIRouter(prefix="/api/search", tags=["search"])


@router.post(
    "/semantic",
    response_model=SemanticSearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Semantic Search Scoped to User Folder",
)
def semantic_search(
    request: SemanticSearchRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Execute natural-language semantic retrieval strictly scoped to the user's folder.

    CRITICAL ISOLATION:
    - User identity is server-verified from the session token.
    - Requires an explicit folder_id belonging to the authenticated user.
    - Returns source metadata only (filename, page, section, relevance score).
    - Extracted text and raw contents are NEVER exposed.
    """
    return execute_semantic_search(db, current_user, request)
