"""Pydantic schemas for ATML Automated Quiz and Question Generation."""
from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict


class QuizQuestionItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    question: str
    options: List[str]
    correct_option_index: int
    correct_answer: str
    explanation: str
    source_document: str
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    concept: str
    prerequisite_review: Optional[dict] = None


class QuizResponseSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    folder_id: str
    folder_name: str
    total_questions: int
    questions: List[QuizQuestionItem]
    note: Optional[str] = None
