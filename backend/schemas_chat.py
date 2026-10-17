"""Pydantic schemas for multi-turn persistent chat."""
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict

from backend.schemas_search import FigureItem, TableItem, SearchResultItem


class ConversationCreateRequest(BaseModel):
    folder_id: str = Field(..., description="Target folder ID for the conversation")
    title: Optional[str] = Field(None, description="Optional custom conversation title")


class ConversationSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    user_id: str
    folder_id: str
    title: str
    message_count: int = 0
    created_at: datetime
    updated_at: datetime


class ChatMessageRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="User's question or message")


class ChatMessageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    conversation_id: str
    role: str
    content: str
    direct_answer: Optional[str] = None
    primary_figure: Optional[FigureItem] = None
    primary_table: Optional[TableItem] = None
    citations: List[SearchResultItem] = []
    refinement_metadata: Optional[Dict[str, Any]] = None
    created_at: datetime


class ConversationDetail(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    user_id: str
    folder_id: str
    title: str
    created_at: datetime
    updated_at: datetime
    messages: List[ChatMessageResponse] = []
