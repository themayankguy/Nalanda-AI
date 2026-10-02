"""Pydantic schemas for local semantic search and RAG retrieval.

CRITICAL REQUIREMENTS:
- Every RAG query must be scoped to an explicitly selected folder_id.
- Extracted text, OCR output, and raw chunk content remain strictly internal
  to the backend and are NEVER returned in search API responses.
- Responses contain only approved source metadata (filename, page, section, score).
"""
from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict


class SemanticSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="Natural-language search query")
    folder_id: str = Field(..., min_length=1, description="Target folder ID for user- and folder-scoped retrieval")
    top_k: int = Field(default=5, ge=1, le=20, description="Maximum number of relevant source passages to return")


class SearchResultItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    document_id: str
    source_filename: str
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    relevance_score: float = Field(..., description="Cosine similarity score (0.0 to 1.0)")


class SemanticSearchResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    query: str
    folder_id: str
    total_results: int
    results: List[SearchResultItem]
