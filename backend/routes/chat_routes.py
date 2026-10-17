"""Chat and conversation management routes for Nalanda AI."""
from typing import List
from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models import User
from backend.auth import get_current_user
from backend.schemas_chat import (
    ConversationCreateRequest,
    ConversationSummary,
    ConversationDetail,
    ChatMessageRequest,
    ChatMessageResponse,
)
from backend.services.NLP.chat_service import (
    create_conversation,
    list_conversations,
    get_conversation_detail,
    delete_conversation,
    post_user_message_and_respond,
)

router = APIRouter(tags=["chat"])


@router.get(
    "/api/folders/{folder_id}/conversations",
    response_model=List[ConversationSummary],
    status_code=status.HTTP_200_OK,
    summary="List saved conversations for a folder",
)
def get_folder_conversations(
    folder_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Retrieve all saved conversations for the authenticated user in the specified folder."""
    return list_conversations(db, current_user, folder_id)


@router.post(
    "/api/folders/{folder_id}/conversations",
    response_model=ConversationSummary,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new conversation session",
)
def create_new_conversation(
    folder_id: str,
    request: ConversationCreateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Start a new persistent conversation session scoped to the folder."""
    return create_conversation(db, current_user, folder_id, title=request.title)


@router.get(
    "/api/conversations/{conversation_id}",
    response_model=ConversationDetail,
    status_code=status.HTTP_200_OK,
    summary="Get conversation history",
)
def get_conversation(
    conversation_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Retrieve messages and metadata for a specific conversation session."""
    return get_conversation_detail(db, current_user, conversation_id)


@router.post(
    "/api/conversations/{conversation_id}/messages",
    response_model=ChatMessageResponse,
    status_code=status.HTTP_200_OK,
    summary="Send message and generate grounded assistant response",
)
def send_message(
    conversation_id: str,
    request: ChatMessageRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Post a user message, resolve multi-turn context, retrieve evidence, and store assistant reply."""
    return post_user_message_and_respond(db, current_user, conversation_id, request.query)


@router.delete(
    "/api/conversations/{conversation_id}",
    status_code=status.HTTP_200_OK,
    summary="Delete conversation session",
)
def remove_conversation(
    conversation_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Delete a conversation and all its message history."""
    return delete_conversation(db, current_user, conversation_id)
