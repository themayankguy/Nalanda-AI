"""Persistent multi-turn chat orchestration service for Nalanda AI.

Coordinates:
- User and folder access control.
- Conversation creation, listing, retrieval, and deletion.
- Multi-turn query resolution via ChatContextService.
- Multi-chunk semantic retrieval and hierarchical concept synthesis.
- Message history persistence in SQLite database.
"""
import json
import logging
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from fastapi import HTTPException, status

from backend.models import User, Folder, Conversation, ChatMessage
from backend.schemas_chat import (
    ConversationSummary,
    ConversationDetail,
    ChatMessageResponse,
)
from backend.schemas_search import (
    SemanticSearchRequest,
    FigureItem,
    TableItem,
    SearchResultItem,
)
from backend.services.NLP.chat_context_service import ChatContextService
from backend.services.NLP.semantic_search_service import execute_semantic_search

logger = logging.getLogger("nalanda.chat_service")


def get_user_conversation(db: Session, user: User, conversation_id: str) -> Conversation:
    """Retrieve conversation and enforce strict user ownership."""
    conversation = (
        db.query(Conversation)
        .filter(Conversation.id == conversation_id, Conversation.user_id == user.id)
        .first()
    )
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found or access denied.",
        )
    return conversation


def create_conversation(
    db: Session,
    user: User,
    folder_id: str,
    title: Optional[str] = None,
) -> ConversationSummary:
    """Create a new conversation session strictly scoped to an authorized folder."""
    folder = (
        db.query(Folder)
        .filter(Folder.id == folder_id, Folder.user_id == user.id)
        .first()
    )
    if not folder:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Target folder not found or does not belong to you.",
        )

    conv_title = (title or "").strip() or f"Chat - {folder.name}"
    conversation = Conversation(
        user_id=user.id,
        folder_id=folder.id,
        title=conv_title,
    )
    db.add(conversation)
    db.commit()
    db.refresh(conversation)

    return ConversationSummary(
        id=conversation.id,
        user_id=conversation.user_id,
        folder_id=conversation.folder_id,
        title=conversation.title,
        message_count=0,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
    )


def list_conversations(
    db: Session,
    user: User,
    folder_id: str,
) -> List[ConversationSummary]:
    """List all saved conversations for a user in a specific folder."""
    # Verify folder ownership
    folder = (
        db.query(Folder)
        .filter(Folder.id == folder_id, Folder.user_id == user.id)
        .first()
    )
    if not folder:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Target folder not found or does not belong to you.",
        )

    conversations = (
        db.query(Conversation)
        .filter(Conversation.user_id == user.id, Conversation.folder_id == folder_id)
        .order_by(Conversation.updated_at.desc())
        .all()
    )

    summaries = []
    for conv in conversations:
        msg_count = db.query(ChatMessage).filter(ChatMessage.conversation_id == conv.id).count()
        summaries.append(
            ConversationSummary(
                id=conv.id,
                user_id=conv.user_id,
                folder_id=conv.folder_id,
                title=conv.title,
                message_count=msg_count,
                created_at=conv.created_at,
                updated_at=conv.updated_at,
            )
        )
    return summaries


def get_conversation_detail(
    db: Session,
    user: User,
    conversation_id: str,
) -> ConversationDetail:
    """Retrieve full conversation history with message turns and source metadata."""
    conversation = get_user_conversation(db, user, conversation_id)
    messages = (
        db.query(ChatMessage)
        .filter(ChatMessage.conversation_id == conversation.id)
        .order_by(ChatMessage.created_at.asc())
        .all()
    )

    msg_responses = []
    for m in messages:
        # Deserialize JSON payloads
        fig_obj = FigureItem(**json.loads(m.primary_figure_json)) if m.primary_figure_json else None
        tab_obj = TableItem(**json.loads(m.primary_table_json)) if m.primary_table_json else None
        cits = [SearchResultItem(**c) for c in json.loads(m.citations_json)] if m.citations_json else []
        ref_meta = json.loads(m.refinement_metadata_json) if m.refinement_metadata_json else None

        msg_responses.append(
            ChatMessageResponse(
                id=m.id,
                conversation_id=m.conversation_id,
                role=m.role,
                content=m.content,
                direct_answer=m.direct_answer,
                primary_figure=fig_obj,
                primary_table=tab_obj,
                citations=cits,
                refinement_metadata=ref_meta,
                created_at=m.created_at,
            )
        )

    return ConversationDetail(
        id=conversation.id,
        user_id=conversation.user_id,
        folder_id=conversation.folder_id,
        title=conversation.title,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        messages=msg_responses,
    )


def delete_conversation(
    db: Session,
    user: User,
    conversation_id: str,
) -> Dict[str, str]:
    """Delete a conversation and all its associated messages."""
    conversation = get_user_conversation(db, user, conversation_id)
    db.delete(conversation)
    db.commit()
    return {"status": "deleted", "conversation_id": conversation_id}


def post_user_message_and_respond(
    db: Session,
    user: User,
    conversation_id: str,
    query_text: str,
) -> ChatMessageResponse:
    """Process a user message turn:

    1. Store user message in database.
    2. Retrieve recent bounded turns for context.
    3. Resolve follow-up references (pronouns, ordinals).
    4. Execute multi-chunk retrieval and synthesis on the folder.
    5. Store assistant response with metadata in database.
    6. Return structured response.
    """
    clean_query = query_text.strip()
    if not clean_query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query cannot be empty.",
        )

    conversation = get_user_conversation(db, user, conversation_id)

    # 1. Fetch recent bounded history (up to last 6 messages)
    recent_messages = (
        db.query(ChatMessage)
        .filter(ChatMessage.conversation_id == conversation.id)
        .order_by(ChatMessage.created_at.desc())
        .limit(6)
        .all()
    )
    # Reverse to chronological order
    history_dicts = [
        {"role": m.role, "content": m.direct_answer or m.content}
        for m in reversed(recent_messages)
    ]

    # 2. Persist user turn
    user_msg = ChatMessage(
        conversation_id=conversation.id,
        role="user",
        content=clean_query,
    )
    db.add(user_msg)
    db.commit()

    # 3. Resolve follow-up query context (pronouns, ordinals)
    search_query, antecedent = ChatContextService.resolve_followup_query(clean_query, history_dicts)

    # 4. Execute multi-chunk semantic retrieval on folder
    search_req = SemanticSearchRequest(
        query=search_query,
        folder_id=conversation.folder_id,
        top_k=5,
    )
    search_res = execute_semantic_search(db, user, search_req)

    # 5. Determine direct answer content
    answer_text = search_res.direct_answer
    if not answer_text:
        if search_res.total_results == 0:
            answer_text = "No relevant sources or answers were found for this query in the selected folder."
        else:
            top_res = search_res.results[0]
            from backend.services.atml.answer_structurer import AnswerStructurer
            from backend.services.atml.nlg_reconstructor import GeneralizedNLGReconstructor
            structured_snippet = AnswerStructurer().structure(clean_query, top_res.text_snippet)
            lead = GeneralizedNLGReconstructor.synthesize_conversational_intro(clean_query)
            if structured_snippet.startswith("#"):
                answer_text = f"{structured_snippet}"
            else:
                answer_text = f"{lead}\n\n{structured_snippet}"

    # 6. Serialize multimodal assets and metadata
    fig_json = search_res.primary_figure.json() if search_res.primary_figure else None
    tab_json = search_res.primary_table.json() if search_res.primary_table else None
    cits_json = json.dumps([r.dict() for r in search_res.results]) if search_res.results else None
    ref_json = json.dumps(search_res.refinement_metadata) if search_res.refinement_metadata else None

    # 7. Update conversation title if it is default
    if conversation.title.startswith("New Conversation") or conversation.title.startswith("Chat -"):
        # Create concise title from clean_query
        title_words = clean_query.split()[:5]
        new_title = " ".join(title_words)
        if len(new_title) > 40:
            new_title = new_title[:37] + "..."
        conversation.title = new_title

    # 8. Persist assistant turn
    assistant_msg = ChatMessage(
        conversation_id=conversation.id,
        role="assistant",
        content=answer_text,
        direct_answer=answer_text,
        primary_figure_json=fig_json,
        primary_table_json=tab_json,
        citations_json=cits_json,
        refinement_metadata_json=ref_json,
    )
    db.add(assistant_msg)
    db.commit()
    db.refresh(assistant_msg)

    return ChatMessageResponse(
        id=assistant_msg.id,
        conversation_id=assistant_msg.conversation_id,
        role=assistant_msg.role,
        content=assistant_msg.content,
        direct_answer=assistant_msg.direct_answer,
        primary_figure=search_res.primary_figure,
        primary_table=search_res.primary_table,
        citations=search_res.results,
        refinement_metadata=search_res.refinement_metadata,
        created_at=assistant_msg.created_at,
    )
